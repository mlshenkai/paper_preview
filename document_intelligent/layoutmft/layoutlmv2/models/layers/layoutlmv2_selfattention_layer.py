# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2022/11/9 8:36 PM
# @File: layoutlmv2_selfattention
# @Email: mlshenkai@163.com
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from .self_attention_layer import SelfAttentionLayer
from .layoutlmv2_output_layer import LayoutLMv2SelfOutput


class LayoutLMv2SelfAttention(SelfAttentionLayer):
    def __init__(self, config):
        super(LayoutLMv2SelfAttention, self).__init__(config)
        self.has_relation_attention_bias = config.has_relation_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        q, k, v = self.compute_qkv(hidden_states)
        query_layer = self.transpose_for_scores(q)
        key_layer = self.transpose_for_scores(k)
        value_layer = self.transpose_for_scores(v)

        query_layer = query_layer / math.sqrt(self.attention_head_size)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if self.has_relation_attention_bias:
            attention_scores += rel_pos
        if self.has_spatial_attention_bias:
            attention_scores += rel_2d_pos

        if attention_mask:
            attention_scores = attention_scores.float().masked_fill_(
                attention_mask.to(torch.bool), torch.finfo(attention_scores.dtype).min
            )
        attention_probs = F.softmax(
            attention_scores, dim=-1, dtype=torch.float32
        ).type_as(value_layer)
        attention_probs = self.dropout(attention_probs)
        if head_mask:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )
        return outputs


class LayoutLMv2Attention(nn.Module):
    def __init__(self, config):
        super(LayoutLMv2Attention, self).__init__()
        self.config = config
        self.self_attention = LayoutLMv2SelfAttention(config)
        self.output = LayoutLMv2SelfOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        self_attention_output = self.self_attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions,
            rel_pos,
            rel_2d_pos,
        )
        attention_output = self.output(self_attention_output[0], hidden_states)
        outputs = (attention_output,) + self_attention_output[1:]
        return outputs
