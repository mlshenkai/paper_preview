# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2022/11/10 1:45 PM
# @File: layoutlmv2_visual_encoder
# @Email: mlshenkai@163.com
import math

import detectron2.modeling.backbone
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.utils.registry import Registry
from libs.utils.import_utils import is_detectron2_available
from libs.utils.sync_batchnorm_utils import my_convert_sync_batchnorm
import torch.distributed

META_ARCH_REGISTRY = Registry("META_ARCH")
META_ARCH_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


class LayoutLMv2VisualEncoder(nn.Module):
    def __init__(self, config):
        super(LayoutLMv2VisualEncoder, self).__init__()
        self.config = config.get_detectron2_config()
        model = self.build_model(self.config)
        assert isinstance(model, detectron2.modeling.backbone.FPN)
        self.backbone = model.backbone
        assert len(self.config.model.pixel_mean) == len(self.config.model.pixel_std)
        num_channels = len(self.config.model.pixel_mean)
        self.register_buffer(
            "pixel_mean",
            torch.Tensor(self.config.model.pixel_mean).view(num_channels, 1, 1),
        )
        self.register_buffer(
            "pixel_std",
            torch.Tensor(self.config.model.pixel_std).view(num_channels, 1, 1),
        )
        self.out_feature_key = "p2"
        if torch.are_deterministic_algorithms_enabled():
            print("use AveragePool instead of adaptiveAveragePool")
            input_shape = (224, 224)
            backbone_stride = self.backbone.output_shape()[self.out_feature_key].stride
            self.pool = nn.AvgPool2d(
                kernel_size=(
                    math.ceil(
                        math.ceil(input_shape[0] / backbone_stride)
                        / config.image_feature_pool_shape[0]
                    ),
                    math.ceil(
                        math.ceil(input_shape[1] / backbone_stride)
                        / config.image_feature_pool_shape[1]
                    ),
                )
            )
        else:
            self.pool = nn.AdaptiveAvgPool2d(config.image_feature_pool_shape[:2])
        if len(config.image_feature_pool_shape) == 2:
            config.image_feature_pool_shape.append(
                self.backbone.output_shape()[self.out_feature_key].channels
            )
        assert (
            self.backbone.output_shape()[self.out_feature_key].channels
            == config.image_feature_pool_shape[2]
        )

    @staticmethod
    def build_model(cfg):
        meta_arch = cfg.model.meta_architecture
        model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
        model.to(torch.device(cfg.model.device))
        return model

    def forward(self, images):
        image_input = (
            (images if torch.is_tensor(images) else images.tensor()) - self.pixel_mean
        ) / self.pixel_std
        features = self.backbone(image_input)
        features = features[self.out_feature_key]
        features = self.pool(features).flatten(start_dim=2).transpose(1, 2).contiguous()
        return features

    def synchronize_batch_norm(self):
        if not (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_rank() > -1
        ):
            raise RuntimeError("不能使用分布训练")

        self_rank = torch.distributed.get_rank()
        node_size = torch.cuda.device_count()
        world_size = torch.distributed.get_world_size()

        if not (world_size & node_size == 0):
            raise RuntimeError(
                "Make sure the number of processes can be divided by the number of nodes"
            )
        node_global_ranks = [
            list(range(i * node_size, (i + 1) * node_size))
            for i in range(world_size // node_size)
        ]

        sync_bn_groups = [
            torch.distributed.new_group(ranks=node_global_ranks[i])
            for i in range(world_size // node_size)
        ]

        node_rank = self_rank // node_size

        self.backbone = my_convert_sync_batchnorm(
            self.backbone, process_group=sync_bn_groups[node_rank]
        )
