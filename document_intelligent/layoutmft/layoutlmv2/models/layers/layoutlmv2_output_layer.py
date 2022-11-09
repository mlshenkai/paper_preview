# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2022/11/9 8:46 PM
# @File: layoutlmv2_output_layer
# @Email: mlshenkai@163.com
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayoutLMv2SelfOutput(nn.Module):
    def __init__(self, config):
        super(LayoutLMv2SelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
