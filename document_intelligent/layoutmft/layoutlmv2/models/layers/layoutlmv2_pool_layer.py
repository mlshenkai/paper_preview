# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2022/11/10 4:19 PM
# @File: layoutlmv2_pool_layer
# @Email: mlshenkai@163.com
import torch
import torch.nn as nn


class LayoutLMv2PoolerLayer(nn.Module):
    def __init__(self, config):
        super(LayoutLMv2PoolerLayer, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)

        return pooled_output
