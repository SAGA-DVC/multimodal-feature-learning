# ------------------------------------------------------------------------------------
# Sparse DETR
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------


from pathlib import Path

from pyrsistent import v

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# from utils.box_ops import box_cxcywh_to_xyxy
# from utils.misc import unwrap


def idx_to_flat_grid(spatial_shapes, idx):
    flat_grid_shape = (idx.shape[0], int(torch.sum(spatial_shapes)))
    flat_grid = torch.zeros(flat_grid_shape, device=idx.device, dtype=torch.float32)
    flat_grid.scatter_(1, idx.to(torch.int64), 1)

    return flat_grid

# TODO - check spatial_shapes usage
def attn_map_to_flat_grid(spatial_shapes, level_start_index, sampling_locations, attention_weights):
    print('*'*80)
    print(spatial_shapes, level_start_index, sampling_locations.shape, attention_weights.shape)
    # # tensor([400, 200, 100,  50]) tensor([  0, 400, 600, 700]) torch.Size([3, 6, 20, 8, 4, 4, 1]) torch.Size([3, 6, 20, 8, 4, 4])
    
    # sampling_locations: [N, n_layers, Len_q, n_heads, n_levels, n_points, 1]
    # attention_weights: [N, n_layers, Len_q, n_heads, n_levels, n_points]
    N, n_layers, _, n_heads, *_ = sampling_locations.shape
    sampling_locations = sampling_locations.permute(0, 1, 3, 2, 5, 4, 6).flatten(0, 2).flatten(1, 2)
    # [N * n_layers * n_heads, Len_q * n_points, n_levels, 1]
    attention_weights = attention_weights.permute(0, 1, 3, 2, 5, 4).flatten(0, 2).flatten(1, 2)
    # [N * n_layers * n_heads, Len_q * n_points, n_levels]
    
    print(sampling_locations.shape, attention_weights.shape)
    # # torch.Size([144, 80, 4, 1]) torch.Size([144, 80, 4])    # 3*6*8 = 144, 20*4 = 80, 4, 2

    # rev_spatial_shapes = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], dim=-1) # hw -> wh (xy)
    rev_spatial_shapes = torch.stack([torch.sqrt(spatial_shapes).ceil(), torch.sqrt(spatial_shapes).ceil()], dim=-1) # hw -> wh (xy) (4,2)
    print("Rev spatial: ", rev_spatial_shapes, sampling_locations.max())

    # rev_spatial_shapes = spatial_shapes
    col_row_float = sampling_locations * rev_spatial_shapes # 144, 80, 4, 2

    col_row_ll = col_row_float.floor().to(torch.int64)  # 144, 80, 4, 2
    zero = torch.zeros(*col_row_ll.shape[:-1], dtype=torch.int64, device=col_row_ll.device) # 144, 80, 4
    one = torch.ones(*col_row_ll.shape[:-1], dtype=torch.int64, device=col_row_ll.device)   # 144, 80, 4
    col_row_lh = col_row_ll + torch.stack([zero, one], dim=-1)  # 144, 80, 4, 2
    col_row_hl = col_row_ll + torch.stack([one, zero], dim=-1)  # 144, 80, 4, 2
    col_row_hh = col_row_ll + 1 # 144, 80, 4, 2
    
    # 144, 80, 4
    margin_ll = (col_row_float - col_row_ll).prod(dim=-1)
    margin_lh = -(col_row_float - col_row_lh).prod(dim=-1)
    margin_hl = -(col_row_float - col_row_hl).prod(dim=-1)
    margin_hh = (col_row_float - col_row_hh).prod(dim=-1)
    # print("Margin Max: ", margin_ll.max(), margin_lh.max(), margin_hl.max(), margin_hh.max())
    
    flat_grid_shape = (attention_weights.shape[0], int(torch.sum(spatial_shapes)))  # 144, 750
    flat_grid = torch.zeros(flat_grid_shape, dtype=torch.float32, device=attention_weights.device)

    zipped = [(col_row_ll, margin_hh), (col_row_lh, margin_hl), (col_row_hl, margin_lh), (col_row_hh, margin_ll)]
    for col_row, margin in zipped:
        valid_mask = torch.logical_and(
            torch.logical_and(col_row[..., 0] >= 0, col_row[..., 0] < rev_spatial_shapes[..., 0]),
            torch.logical_and(col_row[..., 1] >= 0, col_row[..., 1] < rev_spatial_shapes[..., 1]),
        )   # [144, 80, 4]

        print('-'*10)
        print(col_row[..., 1].max(), spatial_shapes, col_row[..., 0].max(), level_start_index)
        print(col_row[..., 1].shape, spatial_shapes, col_row[..., 0].shape, level_start_index)
        print('-'*10)

        idx = col_row[..., 1] * torch.sqrt(spatial_shapes).ceil().to(torch.int64) + col_row[..., 0] + level_start_index    # 144, 80, 4

        print("Prev IDX max: ", idx.max())
        idx = (idx * valid_mask).flatten(1, 2)  # 144, 320
        print("After IDX max: ", idx.max())

        weights = (attention_weights * valid_mask * margin).flatten(1)  # 144, 320
        
        print("SC: ", flat_grid.shape, idx.shape, weights.shape, idx.max())
        flat_grid.scatter_add_(1, idx, weights)

    print('*'*80)

    return flat_grid.reshape(N, n_layers, n_heads, -1)


def compute_corr(flat_grid_topk, flat_grid_attn_map, spatial_shapes):
    if len(flat_grid_topk.shape) == 1:
        flat_grid_topk = flat_grid_topk.unsqueeze(0)
        flat_grid_attn_map = flat_grid_attn_map.unsqueeze(0)
        
    tot = flat_grid_attn_map.sum(-1)
    hit = (flat_grid_topk * flat_grid_attn_map).sum(-1)

    corr = [hit / tot]
    flat_grid_idx = 0

    for shape in spatial_shapes:
        level_range = np.arange(int(flat_grid_idx), int(flat_grid_idx + shape))
        tot = (flat_grid_attn_map[:, level_range]).sum(-1)
        hit = (flat_grid_topk[:, level_range] * flat_grid_attn_map[:, level_range]).sum(-1)
        flat_grid_idx += shape
        corr.append(hit / tot)
    return corr

