# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2022/11/9 8:57 PM
# @File: layoutlmv2_layer
# @Email: mlshenkai@163.com
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import apply_chunking_to_forward

from .layoutlmv2_selfattention_layer import LayoutLMv2Attention
from .layoutlmv2_intermediate import LayoutLMv2Intermediate
from .layoutlmv2_output_layer import LayoutLMv2SelfOutput


class LayoutLMv2Layer(nn.Module):
    def __init__(self, config):
        super(LayoutLMv2Layer, self).__init__()
        self.config = config
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = LayoutLMv2Attention(config)
        self.intermediate = LayoutLMv2Intermediate(config)
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
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions,
            rel_pos,
            rel_2d_pos,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )
        outputs = (layer_output,) + outputs
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
