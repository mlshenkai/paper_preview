# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2022/11/10 1:33 PM
# @File: model_outputs
# @Email: mlshenkai@163.com
from libs.utils.generic import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple
import torch


@dataclass
class BaseModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
