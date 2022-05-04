# ------------------------------------------------------------------------------------
# Sparse DETR
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------


import numpy as np
import torch


def idx_to_flat_grid(temporal_shapes, idx):
    flat_grid_shape = (idx.shape[0], int(torch.sum(temporal_shapes)))
    flat_grid = torch.zeros(flat_grid_shape, device=idx.device, dtype=torch.float32)
    flat_grid.scatter_(1, idx.to(torch.int64), 1)

    return flat_grid

# TODO - check temporal_shapes usage
def attn_map_to_flat_grid(temporal_shapes, level_start_index, sampling_locations, attention_weights):
    '''
    Params:
        temporal_shapes (tensor:int): multiscale feature level dimension eg. [400, 200, 100, 50]
        level_start_index (tensor:int): starting index for corresponding level eg. [0, 400, 600, 700]
        sampling_locations (tensor): batch_size, num_layers, num_queries, num_heads, num_feature_levels, num_points, 1
        attention_weights (tensor): batch_size, num_layers, num_queries, num_heads, num_feature_levels, num_points
    Return:
        flat_grid (tensor): batch_size, num_layers, num_heads, sum of num_token in all level eg. [3, 6, 8, 750]
        
    '''
    
    # sampling_locations: [batch_size, num_layers, num_queries, num_heads, num_feature_levels, num_points, 1]
    # attention_weights: [batch_size, num_layers, num_queries, num_heads, num_feature_levels, num_points]
    batch_size, num_layers, _, num_heads, *_ = sampling_locations.shape

    # [batch_size * num_layers * num_heads, num_queries * num_points, num_feature_levels, 1]
    sampling_locations = sampling_locations.permute(0, 1, 3, 2, 5, 4, 6).flatten(0, 2).flatten(1, 2)
    
    # [batch_size * num_layers * num_heads, num_queries * num_points, num_feature_levels]
    attention_weights = attention_weights.permute(0, 1, 3, 2, 5, 4).flatten(0, 2).flatten(1, 2)
    
    unsqueezed_temporal_shapes = temporal_shapes.unsqueeze(-1)  # (num_feature_levels, 1)

    # [batch_size * num_layers * num_heads, num_queries * num_points, num_feature_levels, 1]
    tid_float = sampling_locations * unsqueezed_temporal_shapes
    tid_start = tid_float.floor().to(torch.int64)
    tid_end = tid_start + 1
    
    # [batch_size * num_layers * num_heads, num_queries * num_points, num_feature_levels]
    margin_start = (tid_float - tid_start).prod(dim=-1)
    margin_end = (tid_float - tid_end).prod(dim=-1)
    
    # [batch_size * num_layers * num_heads, sum of num_token in all level] eg. 750
    flat_grid_shape = (attention_weights.shape[0], int(torch.sum(temporal_shapes)))
    flat_grid = torch.zeros(flat_grid_shape, dtype=torch.float32, device=attention_weights.device)

    zipped = [(tid_start, margin_end), (tid_end, margin_start)]
    for tid, margin in zipped:
        # [batch_size * num_layers * num_heads, num_queries * num_points, num_feature_levels]
        valid_mask = torch.logical_and(tid[..., 0] >= 0, tid[..., 0] < temporal_shapes)

        # [batch_size * num_layers * num_heads, num_queries * num_points, num_feature_levels]
        idx = tid[..., 0] + level_start_index    # 144, 80, 4
        
        # [batch_size * num_layers * num_heads, num_queries * num_points * num_feature_levels]
        idx = (idx * valid_mask).flatten(1, 2)  # 144, 320

        # [batch_size * num_layers * num_heads, num_queries * num_points * num_feature_levels]
        weights = (attention_weights * valid_mask * margin).flatten(1)  # 144, 320
        
        flat_grid.scatter_add_(1, idx, weights)

    return flat_grid.reshape(batch_size, num_layers, num_heads, -1)


def compute_corr(flat_grid_topk, flat_grid_attn_map, temporal_shapes):
    if len(flat_grid_topk.shape) == 1:
        flat_grid_topk = flat_grid_topk.unsqueeze(0)
        flat_grid_attn_map = flat_grid_attn_map.unsqueeze(0)
        
    tot = flat_grid_attn_map.sum(-1)
    hit = (flat_grid_topk * flat_grid_attn_map).sum(-1)

    corr = [hit / tot]
    flat_grid_idx = 0

    for shape in temporal_shapes:
        level_range = np.arange(int(flat_grid_idx), int(flat_grid_idx + shape))
        tot = (flat_grid_attn_map[:, level_range]).sum(-1)
        hit = (flat_grid_topk[:, level_range] * flat_grid_attn_map[:, level_range]).sum(-1)
        flat_grid_idx += shape
        corr.append(hit / tot)
    return corr

