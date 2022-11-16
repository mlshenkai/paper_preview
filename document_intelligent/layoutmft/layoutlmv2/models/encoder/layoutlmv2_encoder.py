# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2022/11/9 9:36 PM
# @File: layoutlmv2_encoder
# @Email: mlshenkai@163.com
import torch
import torch.nn as nn
from torch.utils import checkpoint as utils_checkpoints
import torch.nn.functional as F
from ..layers.layoutlmv2_layer import LayoutLMv2Layer
from document_intelligent.utils.bucket_utils import relative_position_bucket
from libs.model.model_outputs import BaseModelOutput


class LayoutLMv2Encoder(nn.Module):
    def __init__(self, config):
        super(LayoutLMv2Encoder, self).__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [LayoutLMv2Layer(config) for _ in range(config.num_hidden_layers)]
        )

        self.has_relation_attention_bias = config.has_relation_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias

        if self.has_relative_attention_bias:
            self.rel_pos_bins = config.rel_pos_bins
            self.max_rel_pos = config.max_rel_pos
            self.rel_pos_onehot_size = config.rel_pos_bins
            self.rel_pos_bias = nn.Linear(
                self.rel_pos_onehot_size, config.num_attention_heads, bias=False
            )

        if self.has_spatial_attention_bias:
            self.max_rel_2d_pos = config.max_rel_2d_pos
            self.rel_2d_pos_bins = config.rel_2d_pos_bins
            self.rel_2d_pos_onehot_size = config.rel_2d_pos_bins
            self.rel_pos_x_bias = nn.Linear(
                self.rel_2d_pos_onehot_size, config.num_attention_heads, bias=False
            )
            self.rel_pos_y_bias = nn.Linear(
                self.rel_2d_pos_onehot_size, config.num_attention_heads, bias=False
            )

        self.gradient_checkpointing = False

    def _calculate_1d_position_embeddings(self, hidden_states, position_ids):
        rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
        rel_pos = relative_position_bucket(
            rel_pos_mat,
            num_buckets=self.rel_pos_bins,
            max_distance=self.max_rel_pos,
        )
        rel_pos = nn.functional.one_hot(
            rel_pos, num_classes=self.rel_pos_onehot_size
        ).type_as(hidden_states)
        rel_pos = self.rel_pos_bias(rel_pos).permute(0, 3, 1, 2)
        rel_pos = rel_pos.contiguous()
        return rel_pos

    def _calculate_2d_position_embeddings(self, hidden_states, bbox):
        position_coord_x = bbox[:, :, 0]
        position_coord_y = bbox[:, :, 3]
        rel_pos_x_2d_mat = position_coord_x.unsqueeze(-2) - position_coord_x.unsqueeze(
            -1
        )
        rel_pos_y_2d_mat = position_coord_y.unsqueeze(-2) - position_coord_y.unsqueeze(
            -1
        )
        rel_pos_x = relative_position_bucket(
            rel_pos_x_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos,
        )
        rel_pos_y = relative_position_bucket(
            rel_pos_y_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos,
        )
        rel_pos_x = F.one_hot(
            rel_pos_x, num_classes=self.rel_2d_pos_onehot_size
        ).type_as(hidden_states)
        rel_pos_y = F.one_hot(
            rel_pos_y, num_classes=self.rel_2d_pos_onehot_size
        ).type_as(hidden_states)
        rel_pos_x = self.rel_pos_x_bias(rel_pos_x).permute(0, 3, 1, 2)
        rel_pos_y = self.rel_pos_y_bias(rel_pos_y).permute(0, 3, 1, 2)
        rel_pos_x = rel_pos_x.contiguous()
        rel_pos_y = rel_pos_y.contiguous()
        rel_2d_pos = rel_pos_x + rel_pos_y
        return rel_2d_pos

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        bbox=None,
        position_ids=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attention = () if output_attentions else None
        rel_pos = (
            self._calculate_1d_position_embeddings(hidden_states, position_ids)
            if self.has_relation_attention_bias
            else None
        )
        rel_2d_pos = (
            self._calculate_2d_position_embeddings(hidden_states, bbox)
            if self.has_spatial_attention_bias
            else None
        )
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = utils_checkpoints.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    rel_pos,
                    rel_2d_pos,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                    rel_pos,
                    rel_2d_pos,
                )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attention = all_self_attention + (hidden_states,)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attention]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attention,
        )
