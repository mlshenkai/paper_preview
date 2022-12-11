# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2022/11/9 7:27 PM
# @File: self_attention_layer
# @Email: mlshenkai@163.com
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionLayer(nn.Module):
    def __init__(self, config):
        super(SelfAttentionLayer, self).__init__()
        self.config = config
        self.fast_qkv = config.fast_qkv
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if self.fast_qkv:
            self.qkv_linear = nn.Linear(
                config.hidden_size, 3 * self.all_head_size, bias=False
            )
            self.q_bias = nn.Parameter(torch.zeros(1, 1, self.all_head_size))
            self.v_bias = nn.Parameter(torch.zeros(1, 1, self.all_head_size))
        else:
            self.query = nn.Linear(config.hidden_size, self.all_head_size)
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_prob_dropout_prob)

    def transpose_for_scores(self, x):
        """

        Args:
            x: (tensor) (B, N, D)

        Returns:

        """
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )  # [B, N, n_h, h_d]
        x = x.view(*new_x_shape)  # (B, N, n_h, h_d)
        return x

    def compute_qkv(self, hidden_states):
        if self.fast_qkv:
            qkv = self.qkv_linear(hidden_states)
            q, k, v = torch.chunk(qkv, 3, dim=-1)
            if q.ndimension() == self.q_bias.ndimension():
                q = q + self.q_bias
                v = v + self.v_bias
            else:
                _sz = (1,) + (q.ndimension() - 1) + (-1,)
                q = q + self.q_bias.view(*_sz)
                v = v + self.v_bias.view(*_sz)
        else:
            q = self.query(hidden_states)
            k = self.key(hidden_states)
            v = self.value(hidden_states)
        return q, k, v

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        q, k, v = self.compute_qkv(hidden_states)
        query_layer = self.transpose_for_scores(q)
        key_layer = self.transpose_for_scores(k)
        value_layer = self.transpose_for_scores(v)

        query_layer = query_layer / math.sqrt(self.attention_head_size)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
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
