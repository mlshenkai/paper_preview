# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2022/11/1 4:11 PM
# @File: modeling_layoutlmv2
# @Email: mlshenkai@163.com
import detectron2
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from libs.model.extractor import RoIPool
from detectron2.modeling import META_ARCH_REGISTRY
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.layoutlm.modeling_layoutlm import (
    LayoutLMOutput as LayoutLMv2Output,
)
from transformers.modeling_utils import (
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.models.layoutlm.modeling_layoutlm import (
    LayoutLMIntermediate as LayoutLMv2Intermediate,
)
from transformers.models.layoutlmv2 import modeling_layoutlmv2


class LayoutLMv2Embedding(nn.Module):
    """
    refuse text_embedding, position_embedding
    """

    def __init__(self, config):
        super(LayoutLMv2Embedding, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        self.x_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.coordinate_size
        )
        self.y_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.coordinate_size
        )
        self.h_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.shape_size
        )
        self.w_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.shape_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

        self.LayerNormal = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1))
        )

    def _cal_spatial_position_embeddings(self, bbox):
        """
        1维 绝对position embedding
        Args:
            bbox: tensor, (B,N,4)

        Returns:

        """

        left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
        upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
        right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
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


class LayoutLmv2SelfAttention(nn.Module):
    def __init__(self, config):
        super(LayoutLmv2SelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.fast_qkv = config.fast_qkv
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.has_relative_attention_bias = config.has_relative_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias

        if config.fast_qkv:
            self.qkv_linear = nn.Linear(
                config.hidden_size, 3 * self.all_head_size, bias=False
            )
            self.q_bias = nn.Parameter(torch.zeros(1, 1, self.all_head_size))
            self.v_bias = nn.Parameter(torch.zeros(1, 1, self.all_head_size))
        else:
            self.query = nn.Linear(config.hidden_size, self.all_head_size)
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_score(self, x):
        """

        Args:
            x: tensor (B, N, H*D)

        Returns:
            tensor (B, H, N, D)
        """
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )  # (B, N, H*D) -> (B,N)+(H,D)-> (B,N,H,D)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (B, H, N, D)

    def compute_qkv(self, hidden_states):
        """

        Args:
            hidden_states: (B,N, H*D)

        Returns:

        """
        if self.fast_qkv:
            qkv = self.qkv_linear(hidden_states)  # (B, N, 3*all_head_size)
            q, k, v = torch.chunk(qkv, 3, dim=-1)
            if q.ndimension() == self.q_bias.ndimension():
                q = q + self.q_bias
                v = v + self.v_bias
            else:
                _sz = (1,) * (q.ndimension() - 1) + (-1,)
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
        encoder_hidden_states=None,
        encoder_attention_states=None,
        past_key_value=None,
        output_attention=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        q, k, v = self.compute_qkv(hidden_states)

        #  B,N, H*D -> B, H, N, D
        query_layer = self.transpose_for_score(q)
        key_layer = self.transpose_for_score(k)
        value_layer = self.transpose_for_score(v)

        query_layer = query_layer / math.sqrt(self.attention_head_size)

        # (B, H, N, D) * (B, H, D, N) -> (B, H, N, N)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if self.has_relative_attention_bias:
            attention_scores += rel_pos
        if self.has_spatial_attention_bias:
            attention_scores += rel_2d_pos

        attention_scores = attention_scores.float().mask_fill_(
            attention_mask.to(torch.bool), float(1e-8)
        )
        attention_probs = F.softmax(
            attention_scores, dim=-1, dtype=torch.float32
        ).type_as(value_layer)

        attention_probs = self.dropout(attention_probs)

        #  (B, H, N, N) * (B, H, N, D) -> (B, H, N, D)
        context_layer = torch.matmul(attention_probs, value_layer)

        #  (B, H, N, D) -> (B, N, H, D)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        #  (B, N, H, D) -> (B, N)+(all_size, ) -> (B, N, all_size, )
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)

        #  (B, N, all_size, )
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attention else (context_layer,)
        )  # ((B,N,H,D),(B,H,N,N))
        return outputs


class LayoutLmv2Attention(nn.Module):
    def __init__(self, config):
        super(LayoutLmv2Attention, self).__init__()
        self.self_attention = LayoutLmv2SelfAttention(config)
        self.output = LayoutLMv2Output(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads,
            self.self_attention.num_attention_heads,
            self.self_attention.attention_head_size,
            self.pruned_heads,
        )

        #  Prune linear layers
        self.self_attention.query = prune_linear_layer(self.self_attention.query, index)
        self.self_attention.key = prune_linear_layer(self.self_attention.key, index)
        self.self_attention.value = prune_linear_layer(self.self_attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        #  Update hyper params and store pruned heads
        self.self_attention.num_attention_heads = (
            self.self_attention.num_attention_heads - len(heads)
        )
        self.self_attention.all_head_size = (
            self.self_attention.attention_head_size
            * self.self_attention.num_attention_heads
        )
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        self_attention_output = self.self_attention(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
        )

        #  self_attention_output[0]: context_layer_output (B,N,H,D)
        attention_output = self.output(self_attention_output[0], hidden_states)
        outputs = (attention_output,) + self_attention_output[
            1:
        ]  # add attention if we output them
        return outputs


class LayoutLMv2Layer(nn.Module):
    def __init__(self, config):
        super(LayoutLMv2Layer, self).__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = LayoutLmv2Attention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            self.cross_attention = LayoutLmv2Attention(config)
        self.intermediate = LayoutLMv2Intermediate(config)
        self.output = LayoutLMv2Output(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
        )
        attention_output = self_attention_outputs[0]
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attn_past_key_value[-1]
        else:
            outputs = self_attention_outputs[1:]
        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            cross_attn_present_key_value = (
                past_key_value[-2:] if past_key_value is not None else None
            )
            cross_attention_outputs = self.cross_attention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_present_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]

            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )
        outputs = (layer_output,) + outputs
        if self.is_decoder:
            outputs = outputs + (present_key_value,)
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


def relative_position_bucket(
    relative_position, bidirectional=True, num_buckets=32, max_distance=128
):
    ret = 0
    if bidirectional:
        num_buckets //= 2
        ret += (relative_position > 0).long() * num_buckets
        n = torch.abs(relative_position)
    else:
        n = torch.max(-relative_position, torch.zeros_like(relative_position))
    # now n is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = n < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    val_if_large = max_exact + (
        torch.log(n.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    val_if_large = torch.min(
        val_if_large, torch.full_like(val_if_large, num_buckets - 1)
    )

    ret += torch.where(is_small, n, val_if_large)
    return ret


class LayoutLMv2Encoder(nn.Module):
    def __init__(self, config):
        super(LayoutLMv2Encoder, self).__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [LayoutLMv2Layer(self.config) for _ in range(self.config.num_hidden_layers)]
        )
        self.has_relative_attention_bias = self.config.has_relative_attention_bias
        self.has_spatial_attention_bias = self.config.has_spatial_attention_bias
        if self.has_relative_attention_bias:
            self.rel_pos_bins = self.config.config.rel_pos_bins
            self.max_rel_pos = self.config.max_rel_pos
            self.rel_pos_onehot_size = self.config.rel_pos_bins
            self.rel_pos_bias = nn.Linear(
                self.rel_pos_onehot_size, self.config.num_attention_heads, bias=False
            )
        if self.has_spatial_attention_bias:
            self.max_rel_2s_pos = self.config.max_rel_2d_pos
            self.rel_2d_pos_bin = self.config.rel_2d_pos_bins
            self.rel_2d_onehot_size = self.config.rel_2d_pos_bins
            self.rel_pos_x_bias = nn.Linear(
                self.rel_2d_onehot_size, self.config.num_attention_heads, bias=False
            )
            self.rel_pos_y_bias = nn.Linear(
                self.rel_2d_onehot_size, self.config.num_attention_heads, bias=False
            )

    def _cal_1d_pos_emb(self, hidden_states, position_ids):
        rel_pos_mat = position_ids.unqueeze(-2) - position_ids.unsqueeze(-1)
        rel_pos = relative_position_bucket(
            rel_pos_mat,
            num_buckets=self.rel_pos_bins,
            max_distance=self.max_rel_pos,
        )
        rel_pos = F.one_hot(rel_pos, num_classes=self.rel_pos_onehot_size).type_as(
            hidden_states
        )
        rel_pos = self.rel_pos_bias(rel_pos).permute(0, 3, 1, 2)
        rel_pos = rel_pos.contiguous()
        return rel_pos

    def _cal_2d_pos_emb(self, hidden_states, bbox):
        position_coord_x = bbox[:, :, 0]
        position_coord_y = bbox[:, :, 3]
        rel_pos_x_2d_mat = position_coord_x.unsqueeze(-2) - position_coord_x.unsqueeze(
            -1
        )
        rel_pos_y_2d_mat = position_coord_y.unsqueeze(-2) - position_coord_y.unsqueeze(
            -1
        )
        rel_pos_x = relative_position_bucket(
            rel_pos_x_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos,
        )
        rel_pos_y = relative_position_bucket(
            rel_pos_y_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos,
        )
        rel_pos_x = F.one_hot(
            rel_pos_x, num_classes=self.rel_2d_pos_onehot_size
        ).type_as(hidden_states)
        rel_pos_y = F.one_hot(
            rel_pos_y, num_classes=self.rel_2d_pos_onehot_size
        ).type_as(hidden_states)
        rel_pos_x = self.rel_pos_x_bias(rel_pos_x).permute(0, 3, 1, 2)
        rel_pos_y = self.rel_pos_y_bias(rel_pos_y).permute(0, 3, 1, 2)
        rel_pos_x = rel_pos_x.contiguous()
        rel_pos_y = rel_pos_y.contiguous()
        rel_2d_pos = rel_pos_x + rel_pos_y
        return rel_2d_pos

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        bbox=None,
        position_ids=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )

        next_decoder_cache = () if use_cache else None

        rel_pos = (
            self._cal_1d_pos_emb(hidden_states, position_ids)
            if self.has_relative_attention_bias
            else None
        )
        rel_2d_pos = (
            self._cal_2d_pos_emb(hidden_states, bbox)
            if self.has_spatial_attention_bias
            else None
        )

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    rel_pos=rel_pos,
                    rel_2d_pos=rel_2d_pos,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    rel_pos=rel_pos,
                    rel_2d_pos=rel_2d_pos,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


def my_convert_sync_batchnorm(module, process_group=None):
    # same as `nn.modules.SyncBatchNorm.convert_sync_batchnorm` but allowing converting from `detectron2.layers.FrozenBatchNorm2d`
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        return nn.modules.SyncBatchNorm.convert_sync_batchnorm(module, process_group)
    module_output = module
    if isinstance(module, detectron2.layers.FrozenBatchNorm2d):
        module_output = torch.nn.SyncBatchNorm(
            num_features=module.num_features,
            eps=module.eps,
            affine=True,
            track_running_stats=True,
            process_group=process_group,
        )
        module_output.weight = torch.nn.Parameter(module.weight)
        module_output.bias = torch.nn.Parameter(module.bias)
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = torch.tensor(
            0, dtype=torch.long, device=module.running_mean.device
        )
    for name, child in module.named_children():
        module_output.add_module(name, my_convert_sync_batchnorm(child, process_group))
    del module
    return module_output


class VisualBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = detectron2.config.get_cfg()
        config.add(self.cfg)
        meta_arch = self.cfg.MODEL.META_ARCHITECTURE
        model = META_ARCH_REGISTRY.get(meta_arch)(self.cfg)
        assert isinstance(model.backbone, detectron2.modeling.backbone.FPN)
        self.backbone = model.backbone
        if (
            config.convert_sync_batchnorm
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_rank() > -1
        ):
            self_rank = torch.distributed.get_rank()
            node_size = torch.cuda.device_count()
            world_size = torch.distributed.get_world_size()
            assert world_size % node_size == 0

            node_global_ranks = [
                list(range(i * node_size, (i + 1) * node_size))
                for i in range(world_size // node_size)
            ]
            sync_bn_groups = [
                torch.distributed.new_group(ranks=node_global_ranks[i])
                for i in range(world_size // node_size)
            ]
            node_rank = self_rank // node_size
            assert self_rank in node_global_ranks[node_rank]

            self.backbone = my_convert_sync_batchnorm(
                self.backbone, process_group=sync_bn_groups[node_rank]
            )

        assert len(self.cfg.MODEL.PIXEL_MEAN) == len(self.cfg.MODEL.PIXEL_STD)
        num_channels = len(self.cfg.MODEL.PIXEL_MEAN)
        self.register_buffer(
            "pixel_mean",
            torch.Tensor(self.cfg.MODEL.PIXEL_MEAN).view(num_channels, 1, 1),
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(self.cfg.MODEL.PIXEL_STD).view(num_channels, 1, 1)
        )
        self.out_feature_key = "p2"
        # if torch.is_deterministic():
        #     logger.warning("using `AvgPool2d` instead of `AdaptiveAvgPool2d`")
        #     input_shape = (224, 224)
        #     backbone_stride = self.backbone.output_shape()[self.out_feature_key].stride
        #     self.pool = nn.AvgPool2d(
        #         (
        #             math.ceil(math.ceil(input_shape[0] / backbone_stride) / config.image_feature_pool_shape[0]),
        #             math.ceil(math.ceil(input_shape[1] / backbone_stride) / config.image_feature_pool_shape[1]),
        #         )
        #     )
        # else:
        #     self.pool = nn.AdaptiveAvgPool2d(config.image_feature_pool_shape[:2])
        self.pool = RoIPool(config.image_feature_pool_shape[:2])
        if len(config.image_feature_pool_shape) == 2:
            config.image_feature_pool_shape.append(
                self.backbone.output_shape()[self.out_feature_key].channels
            )
        assert (
            self.backbone.output_shape()[self.out_feature_key].channels
            == config.image_feature_pool_shape[2]
        )

    def forward(self, images):
        images_input = (images.tensor - self.pixel_mean) / self.pixel_std
        features = self.backbone(images_input)
        features = features[self.out_feature_key]
        # features = self.pool(features).flatten(start_dim=2).transpose(1, 2).contiguous()
        features = self.pool(features)  # notice that self.pool has been modified
        return features
