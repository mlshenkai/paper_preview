# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2022/11/1 4:11 PM
# @File: modeling_layoutlmv2
# @Email: mlshenkai@163.com
from typing import Optional, Union, Tuple
from transformers.models.layoutlmv2 import modeling_layoutlmv2
from transformers.models.layoutlmv2 import configuration_layoutlmv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, LayoutLMv2Config, add_start_docstrings
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.pytorch_utils import torch_int_div

from document_intelligent.utils.transformers_pretrain_model import TransformersPreTrainModel
from libs.utils.doc import *
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
class LayoutLMv2Model(TransformersPreTrainModel):
    def __init__(self, config):
        requires_backends(self, "detectron2")
        super(LayoutLMv2Model, self).__init__()
        model_config = config.model
        visual_config = config.visual_model
        self.model_config = model_config
        self.visual_config = visual_config
        self.config = model_config
        self.has_visual_segment_embedding = self.model_config.has_visual_segment_embedding
        self.embeddings = LayoutLMv2Embeddings(self.model_config)
        self.visual = LayoutLMv2VisualEncoder(self.visual_config, self.model_config)
        self.visual_proj = nn.Linear(
            model_config.image_feature_pool_shape[-1], model_config.hidden_size
        )
        if self.has_visual_segment_embedding:
            self.visual_segment_embedding = nn.Parameter(
                nn.Embedding(1, model_config.hidden_size).weight[0]
            )
        self.visual_LayerNorm = nn.LayerNorm(
            model_config.hidden_size, eps=model_config.layer_norm_eps
        )
        self.visual_dropout = nn.Dropout(model_config.hidden_dropout_prob)

        self.encode = LayoutLMv2Encoder(model_config)
        self.pooler = LayoutLMv2PoolerLayer(model_config)

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
            input_ids: (tensor) (B, N)
            bbox:  (tensor) (B, N, 4)
            position_ids: (tensor) (B, N)
            token_type_ids: (tensor) (B, N)
            inputs_embeds: (tensor) (B, N, D)

        Returns:

        """
        if input_ids is not None:
            input_shape = input_ids.size()  # B, N
        else:
            input_shape = inputs_embeds.size()[:-1]  # B. N

        seq_length = input_shape[1]  # N
        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if inputs_embeds is None:
            inputs_embeds = self.embeddings.word_embeddings(input_ids)  # (B, N, D)
        position_embeddings = self.embeddings.position_embeddings(
            position_ids
        )  # (B, N, D)
        spatial_position_embeddings = self.embeddings._cal_spatial_position_embeddings(
            bbox
        )  # (B, N, D)
        token_type_embeddings = self.embeddings.token_type_embeddings(token_type_ids)

        embeddings = (
            inputs_embeds
            + position_embeddings
            + spatial_position_embeddings
            + token_type_embeddings
        )

        embeddings = self.embeddings.LayerNorm(embeddings)
        embeddings = self.embeddings.dropout(embeddings)
        return embeddings

    def _calc_img_embeddings(self, image, bbox, position_ids):
        visual_embeddings = self.visual_proj(self.visual(image))
        position_embeddings = self.embeddings.position_embeddings(position_ids)
        spatial_position_embeddings = self.embeddings._cal_spatial_position_embeddings(
            bbox
        )

        embeddings = (
            visual_embeddings + position_embeddings + spatial_position_embeddings
        )
        if self.has_visual_segment_embedding:
            embeddings += self.visual_segment_embedding
        embeddings = self.visual_LayerNorm(embeddings)
        embeddings = self.visual_dropout(embeddings)
        return embeddings

    def _cals_visual_bbox(self, image_feature_pool_shape, bbox, device, final_shape):
        """
        将图片进行切分 每个
        Args:
            image_feature_pool_shape:
            bbox:
            device:
            final_shape:

        Returns:

        """
        visual_bbox_x = torch_int_div(
            torch.arange(
                0,
                1000 * (image_feature_pool_shape[1] + 1),
                1000,
                device=device,
                dtype=bbox.dtype,
            ),
            self.config.image_feature_pool_shape[1],
        )
        visual_bbox_y = torch_int_div(
            torch.arange(
                0,
                1000 * (self.config.image_feature_pool_shape[0] + 1),
                1000,
                device=device,
                dtype=bbox.dtype,
            ),
            self.config.image_feature_pool_shape[0],
        )
        visual_bbox = torch.stack(
            [
                visual_bbox_x[:-1].repeat(image_feature_pool_shape[0], 1),
                visual_bbox_y[:-1]
                .repeat(image_feature_pool_shape[1], 1)
                .transpose(0, 1),
                visual_bbox_x[1:].repeat(image_feature_pool_shape[0], 1),
                visual_bbox_y[1:]
                .repeat(image_feature_pool_shape[1], 1)
                .transpose(0, 1),
            ],
            dim=-1,
        ).view(-1, bbox.size(-1))

        visual_bbox = visual_bbox.repeat(final_shape[0], 1, 1)  # B，49, 4

        return visual_bbox

    def _get_input_shape(self, input_ids=None, inputs_embeds=None):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            return input_ids.size()
        elif inputs_embeds is not None:
            return inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        image: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        """
        Return:

        Examples:

         from transformers import LayoutLMv2Processor, LayoutLMv2Model, set_seed
         from PIL import Image
         import torch
         from datasets import load_dataset

         set_seed(88)

         processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")
         model = LayoutLMv2Model.from_pretrained("microsoft/layoutlmv2-base-uncased")


         dataset = load_dataset("hf-internal-testing/fixtures_docvqa")
         image_path = dataset["test"][0]["file"]
         image = Image.open(image_path).convert("RGB")

         encoding = processor(image, return_tensors="pt")

         outputs = model(**encoding)
         last_hidden_states = outputs.last_hidden_state

         last_hidden_states.shape
        torch.Size([1, 342, 768])
        Args:
            input_ids:
            bbox:
            image:
            attention_mask:
            token_type_ids:
            position_ids:
            head_mask:
            inputs_embeds:
            output_attentions:
            output_hidden_states:
            return_dict:

        Returns:

        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )
        input_shape = self._get_input_shape(
            input_ids, inputs_embeds
        )  # (B, N) N is seq_len
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        visual_shape = list(input_shape)  # [B, N]
        visual_shape[1] = (
            self.config.image_feature_pool_shape[0]
            * self.config.image_feature_pool_shape[1]
        )  # w*h
        visual_shape = torch.Size(visual_shape)  # [B, w*h]
        final_shape = list(self._get_input_shape(input_ids, inputs_embeds))  # [B,N]
        final_shape[1] += visual_shape[1]  # [B, N+w*h]
        final_shape = torch.Size(final_shape)

        visual_bbox = self._cals_visual_bbox(
            self.config.image_feature_pool_shape, bbox, device, final_shape
        )
        final_bbox = torch.cat(
            [bbox, visual_bbox], dim=1
        )  # box->(B, N, 4), visual_bbox->(B, 49, 4) =cat=> (B, N+49, 4)
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        visual_attention_mask = torch.ones(visual_shape, device=device)
        final_attention_mask = torch.cat([attention_mask, visual_attention_mask], dim=1)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if position_ids is None:
            seq_len = input_shape[1]  # input_shape->(B, seq_len) seq_len=N
            position_ids = self.embeddings.position_ids[:, :seq_len]  # B, seq_len
            position_ids = position_ids.expand(input_shape)

        visual_position_ids = torch.arange(
            0, visual_shape[1], dtype=torch.long, device=device
        ).repeat(input_shape[0], 1)  # (B, w*h)
        final_position_ids = torch.cat([position_ids, visual_position_ids], dim=1)

        if bbox is None:
            bbox = torch.zeros(tuple(list(input_shape)+[4]), dtype=torch.long, device=device)  # (B, N, 4)

        text_layout_emb = self._calc_text_embeddings(
            input_ids=input_ids,
            bbox=bbox,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds
        )

        visual_emb = self._calc_img_embeddings(image, bbox=visual_bbox, position_ids=visual_position_ids)

        final_emb = torch.cat([text_layout_emb, visual_emb], dim=1)
        extended_attention_mask = final_attention_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (1, 1, N, 1, 1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)  # (num_head, 1, N, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers
        encoder_outputs = self.encode(
            final_emb,
            extended_attention_mask,
            bbox=final_bbox,
            position_ids=final_position_ids,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        sequence_output = encoder_outputs[0]  # (B, N, D)
        pooled_output = self.pooler(sequence_output)  # (B, N, D)

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions
        )


