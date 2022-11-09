# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2022/11/9 9:23 PM
# @File: layoutlmv2_intermediate
# @Email: mlshenkai@163.com
import torch
import torch.nn as nn
from libs.utils.activations import ACT2FN


class LayoutLMv2Intermediate(nn.Module):
    def __init__(self, config):
        super(LayoutLMv2Intermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
