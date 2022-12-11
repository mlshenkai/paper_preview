# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2022/12/7 2:37 PM
# @File: modeling_layoutlm
# @Email: mlshenkai@163.com
import torch
import math
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from libs.utils.activations import ACT2FN
from libs.utils import logging


logger = logging.get_logger(__name__)

LayoutLMLayerNormal = torch.nn.LayerNorm


class LayoutLMEmbeddings(nn.Module):
    """
    包含 word embedding+position embeddings
    """
    def __init__(self, config):
        super(LayoutLMEmbeddings, self).__init__()
        self.config = config

        self.word_embeddings = nn.Embedding(config.vocab_size)





