# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2022/11/1 4:11 PM
# @File: modeling_layoutlmv2
# @Email: mlshenkai@163.com
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, LayoutLMv2Config, add_start_docstrings
from .layers.layout_embedding_layer import LayoutLMv2Embeddings
from libs.utils.import_utils import requires_backends
from .encoder.layoutlmv2_encoder import LayoutLMv2Encoder
from .encoder.layoutlmv2_visual_encoder import LayoutLMv2VisualEncoder
from .layers.layoutlmv2_pool_layer import LayoutLMv2PoolerLayer

LAYOUTLMV2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/layoutlmv2-base-uncased",
    "microsoft/layoutlmv2-large-uncased",
]


class LayoutLMv2PreTrainModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LayoutLMv2Config
    pretrained_model_archive_map = LAYOUTLMV2_PRETRAINED_MODEL_ARCHIVE_LIST
    base_model_prefix = "layoutlmv2"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LayoutLMv2Encoder):
            module.gradient_checkpointing = value


@add_start_docstrings(
    "The bare LayoutLMv2 Model transformer outputting raw hidden-states without any specific head on top.",
    "LAYOUTLMV2_START_DOCSTRING",
)
class LayoutLMv2Model(LayoutLMv2PreTrainModel):
    def __init__(self, config):
        requires_backends(self, "detectron2")
        super(LayoutLMv2Model, self).__init__(config)
        self.config = config
        self.has_visual_segment_embedding = config.has_visual_segment_embedding
        self.embeddings = LayoutLMv2Embeddings(config)
        self.visual = LayoutLMv2VisualEncoder(config)
        self.visual_proj = nn.Linear(
            config.image_feature_pool_shape[-1], config.hidden_size
        )
        if self.has_visual_segment_embedding:
            self.visual_segment_embedding = nn.Parameter(
                nn.Embedding(1, config.hidden_size).weight[0]
            )
        self.visual_LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.visual_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.encode = LayoutLMv2Encoder(config)
        self.pooler = LayoutLMv2PoolerLayer(config)

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value: nn.Module):
        self.embeddings.word_embeddings = value

    def _calc_text_embeddings(
        self, input_ids, bbox, position_ids, token_type_ids, inputs_embeds=None
    ):
        """

        Args:
            input_ids: (tensor) (B, N, seq_len)
            bbox:  (tensor) (B, N, 4)
            position_ids: (tensor) (B, N, seq_len)
            token_type_ids: (tensor) (B, N, seq_len)
            inputs_embeds: (tensor) (B, N, seq, D)

        Returns:

        """
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        #  B, N, D

        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if inputs_embeds is None:
            inputs_embeds = self.embeddings.word_embeddings(input_ids)
        position_embeddings = self.embeddings.position_embeddings(
            position_ids
        )  # (B, N, seq_len, D)
        spatial_position_embeddings = self.embeddings._cal_spatial_position_embeddings(
            bbox
        )  # (B, N, D)
        token_type_embeddings = self.embeddings.token_type_embeddings(token_type_ids)

    def post_init(self):
        self.init_weights()
        self._backward_compatibility_gradient_checkpointing()

    def _backward_compatibility_gradient_checkpointing(self):
        if self.supports_gradient_checkpointing and getattr(
            self.config, "gradient_checkpointing", False
        ):
            self.gradient_checkpointing_enable()
            # Remove the attribute now that is has been consumed, so it's no saved in the config.
            delattr(self.config, "gradient_checkpointing")
