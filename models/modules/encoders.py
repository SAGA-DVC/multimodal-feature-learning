""" Modules for DVC """

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from typing import Optional
import math
import warnings

import numpy as np

import torch
import torchvision
import torch.nn as nn
from torch import Tensor
from torch.nn.init import xavier_uniform_, constant_
import torch.nn.functional as F
from torch.autograd import Function

try:
    import MultiScaleDeformableAttention as MSDA
except:
    pass
from torch.autograd.function import once_differentiable
# needed due to empty tensor bug in pytorch and torchvision 0.5

from .encoders import *
 



class VivitEncoder(nn.Module):
    def __init__(self, model_name, num_frames, num_patches, d_model, depth, temporal_depth, num_heads, 
                mlp_ratio=4., qkv_bias=False, attention_dropout=0., projection_dropout=0., dropout_1=0., dropout_2=0., pre_norm=True):
        
        """
        ViViT Encoder for spatio temporal attention, factorised attention, factorised self attention and factorised dot product attention.
  
        Parameters:
            `model_name` (string): One of 'spatio temporal attention', 'factorised encoder', 'factorised self attention' or 'factorised dot product attention'
            `num_frames` (int): Number of frames in the input video
            `num_patches` (int): Number of patches per frame in the input video
            `d_model` (int): Dimension of the tensors used to compute attention
            `depth` (int): number of encoder blocks. 
            `temporal_depth` (int): number of temporal encoder blocks (for factorised encoder model only)
            `num_heads` (int): Number of attention heads.
            `mlp_ratio` (int): Used to determine the hidden layer dimension of the MLP. (default 4)
            `qkv_bias` (boolean): Determines whether to use bias as part of the query/key/value linear layers in the attention block (default True)
            `attention_dropout` (float): Dropout probability for the layer after the multi-head attention mechanism (default 0.0)
            `projection_dropout` (float): Dropout probability for the layer after the projection layer (default 0.0)
            `dropout_1` (float): Dropout probability for the MLP block (default 0.0)
            `dropout_2` (float): Dropout probability for the MLP block (default 0.0)
            `pre_norm` (boolean): If True, the normalisation layer would be placed before the attention and mlp blocks. Else, after them. (default True)

        """

        super(VivitEncoder, self).__init__()

        self.model_name = model_name

        if self.model_name == 'spatio temporal attention':
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model)) # [class] token

            self.encoder = nn.ModuleList(
                [
                    EncoderLayer(
                        d_model=d_model,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        attention_dropout=attention_dropout,
                        projection_dropout=projection_dropout,
                        dropout_1=dropout_1,
                        dropout_2=dropout_2,
                        pre_norm=pre_norm
                    )
                    for _ in range(depth)
                ]
            )

        elif self.model_name == 'factorised encoder':
            self.spacial_token = nn.Parameter(torch.zeros(1, 1, d_model)) # spatial [class] token
            self.temporal_token = nn.Parameter(torch.zeros(1, 1, d_model)) # temporal [class] token
            
            self.spatialEncoder = nn.ModuleList(
                [
                    EncoderLayer(
                        d_model=d_model,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        attention_dropout=attention_dropout,
                        projection_dropout=projection_dropout,
                        dropout_1=dropout_1,
                        dropout_2=dropout_2,
                        pre_norm=pre_norm
                    )
                    for _ in range(depth)
                ]
            )

            self.temporalEncoder = nn.ModuleList(
                [
                    EncoderLayer(
                        d_model=d_model,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        attention_dropout=attention_dropout,
                        projection_dropout=projection_dropout,
                        dropout_1=dropout_1,
                        dropout_2=dropout_2,
                        pre_norm=pre_norm
                    )
                    for _ in range(temporal_depth)
                ]
            )

        elif self.model_name == 'factorised self attention':
            self.encoder = nn.ModuleList(
                [
                    FactorisedSelfAttentionEncoderLayer(
                        d_model=d_model,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        attention_dropout=attention_dropout,
                        projection_dropout=projection_dropout,
                        dropout_1=dropout_1,
                        dropout_2=dropout_2
                    )
                    for _ in range(depth)
                ]
            )
        elif self.model_name == 'factorised dot product attention':
            self.encoder = nn.ModuleList(
                [
                    FactorisedDotProductAttentionEncoderLayer(
                        d_model=d_model,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        attention_dropout=attention_dropout,
                        projection_dropout=projection_dropout,
                        dropout_1=dropout_1,
                        dropout_2=dropout_2
                    )
                    for _ in range(depth)
                ]
            )
        else:
            raise ValueError(f'Unrecognized model: {model_name}. Choose between factorised encoder, \
                            factorised self attention or factorised dot product attention')


    def forward(self, x, positional_embedding_layer=None, spatial_positional_embedding_layer=None):

        """
        Performs a forward pass over the specified attention architecture for all layers of the ViViT encoder.
  
        Parameters:
            x (tensor): Tensor of dimension (batch_size, num_frames, num_patches, d_model)
        
        Returns:
            x (tensor): if model_name is 'spatio temporal attention', Tensor of dimension (batch_size, num_frames * num_patches + 1, d_model)
                        if model_name is 'factorised encoder', Tensor of dimension (batch_size, num_frames + 1, d_model) 
                        if model _name is 'factorised self attention' or 'factorised dot product attention', 
                        Tensor of dimension (batch_size, num_frames, num_patches, d_model)  

        """
        
        batch_size, num_frames, num_patches, d_model = x.shape
        
        if positional_embedding_layer == None:
            positional_embedding_layer = nn.Identity()

        if spatial_positional_embedding_layer == None:
            spatial_positional_embedding_layer = nn.Identity()

        # change pos embed for this model?
        if self.model_name == 'spatio temporal attention':

            x = x.reshape(batch_size, -1, d_model) # (batch_size, num_frames * num_patches, d_model)

            cls_token = self.cls.expand(batch_size, 1, -1) # (1, 1, d_model) -> (batch_size, 1, d_model)

            x = torch.cat((cls_token, x), dim=1) # (batch_size, num_frames * num_patches + 1, d_model)
            
            for layer in self.encoder:
                x = positional_embedding_layer(x) # (batch_size, num_frames * num_patches + 1, d_model)
                x = layer(x) # (batch_size, num_frames * num_patches + 1, d_model)


        elif self.model_name == 'factorised encoder':

            x = x.reshape(-1, num_patches, d_model) # (batch_size * num_frames, num_patches, d_model)
        
            cls_token_spatial = self.spacial_token.expand(batch_size * num_frames, 1, -1) # (1, 1, d_model) -> (batch_size * num_frames, 1, d_model)
            x = torch.cat((cls_token_spatial, x), dim=1) # (batch_size * num_frames, num_patches + 1, d_model)
        
            for layer in self.spatialEncoder:
                x = spatial_positional_embedding_layer(x) # (batch_size * num_frames, num_patches + 1, d_model)
                x = layer(x) # (batch_size * num_frames, num_patches + 1, d_model)
            
            x = x.reshape(batch_size, num_frames, num_patches + 1, d_model)

            x = x[:, :, 0] # (batch_size, num_frames, d_model)

            cls_token_temporal = self.temporal_token.expand(batch_size, -1, -1) # (1, 1, d_model) -> (batch_size, 1, d_model)
            x = torch.cat((cls_token_temporal, x), dim=1) # (batch_size, num_frames + 1, d_model)

            for layer in self.temporalEncoder:
                x = positional_embedding_layer(x) # (batch_size, num_frames + 1, d_model)
                x = layer(x) # (batch_size, num_frames + 1, d_model)

        elif self.model_name == 'factorised self attention' or self.model_name == 'factorised dot product attention': 
            for layer in self.encoder:
                x = positional_embedding_layer(x) # (batch_size, num_frames, num_patches, d_model)
                x = layer(x) # (batch_size, num_frames, num_patches, d_model)

        return x
