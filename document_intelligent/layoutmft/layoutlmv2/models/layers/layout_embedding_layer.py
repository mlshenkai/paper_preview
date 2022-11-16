# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2022/11/9 7:28 PM
# @File: layout_embedding_layer
# @Email: mlshenkai@163.com
import torch
import torch.nn as nn


class LayoutLMv2Embeddings(nn.Module):
    """
    layoutlmv2 layout embedding
    """

    def __init__(self, config):
        super(LayoutLMv2Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )

        self.x_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.coordinate_size
        )
        self.y_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.coordinate_size
        )
        self.h_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.coordinate_size
        )
        self.w_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.coordinate_size
        )
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer(
            "position_ids",
            torch.arange(config.max_2d_position_embeddings).expand((-1, 1)),
        )

    def _cal_spatial_position_embeddings(self, bbox):
        """
        绝对坐标 embedding
        Args:
            bbox:  (tensor) (B,N, 4)

        Returns:

        """

        left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
        right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
        upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
        lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])

        h_position_embeddings = self.h_position_embeddings(
            bbox[:, :, 3] - bbox[:, :, 1]
        )
        w_position_embeddings = self.w_position_embeddings(
            bbox[:, :, 2] - bbox[:, :, 0]
        )

        spatial_position_embeddings = torch.cat(
            [
                left_position_embeddings,
                upper_position_embeddings,
                right_position_embeddings,
                lower_position_embeddings,
                h_position_embeddings,
                w_position_embeddings,
            ],
            dim=-1,
        )
        return spatial_position_embeddings
