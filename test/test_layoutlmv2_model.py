# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2022/12/7 2:46 PM
# @File: test_layoutlmv2_model
# @Email: mlshenkai@163.com
from document_intelligent.layoutmft.layoutlmv2.models.modeling_layoutlmv2 import LayoutLMv2Model

from model_config_loader.load_config import Config

config = Config("../document_intelligent/layoutmft/layoutlmv2/config/layoutlmv2_config.yaml")
detectron2_config = Config("../document_intelligent/layoutmft/layoutlmv2/config/detectron2_config.yaml")
config.add("visual_model", detectron2_config)
print(config)
layout_model = LayoutLMv2Model(config)

print(layout_model)