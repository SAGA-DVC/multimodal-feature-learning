""" Modules for DVC """

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
import math
import warnings
from torch.nn.init import xavier_uniform_, constant_
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np

try:
    import MultiScaleDeformableAttention as MSDA
except:
    pass
from torch.autograd.function import once_differentiable
# needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision


# --------------------------------------------------------------
# Modules used by ViViT

class TokenEmbedding(nn.Module):
    def __init__(self, img_size=224, spatial_patch_size=16, temporal_patch_size=1, in_channels=3, d_model=768, layer_norm=None):
        
        """
        Converts video into token embeddings based on specified patches that convolve over the video. Based on the
        temporal patch size, these embeddings can follow 'uniform frame sampling' or 'tubelet embedding' schemes.

        Parameters:
            `img_size` (int): dimension of one frame of the video (should be a square i.e. height=width) (default 224)
            `spatial_patch_size` (int): dimension of the patch that will be used to convolve over a frame (default 16)
            `temporal_patch_size` (int): dimension of the patch that will be used to convolve over multiple frames (default 1)
            `in_channels` (int): number of channels of the each frame in the video. e.g. RGB. (default 3)
            `num_classes` (int): number of classes for the prediction task (default 1000)
            `d_model` (int): Dimension of the tensors used to compute attention

        """
        
        super(TokenEmbedding, self).__init__()

        self.temporal_patch_size = temporal_patch_size

        self.project_to_patch_embeddings = nn.Conv3d(in_channels, d_model, 
                                                    kernel_size=(temporal_patch_size, spatial_patch_size, spatial_patch_size), 
                                                    stride=(temporal_patch_size, spatial_patch_size, spatial_patch_size))
        self.layer_norm = layer_norm(d_model) if layer_norm else nn.Identity()

    def forward(self, x):

        """
        3D Convolutions are used to get the token embeddings. 
  
        Parameters:
           x (tensor): Tensor of dimension (batch_size, in_channels, num_frames, img_size, img_size)

        Returns:
            x (tensor): Tensor of dimension (batch_size, num_frames, num_patches, d_model)

        """

        x = self.project_to_patch_embeddings(x) # (batch_size, d_model, num_frames, num_patches ** 0.5, num_patches ** 0.5)
        x = x.flatten(3)  # (batch_size, d_model, num_frames, num_patches)
        x = x.permute(0, 2, 3, 1)  # (batch_size, num_frames, num_patches, d_model)
        x = self.layer_norm(x)
        
        return x

# TODO - pos emebd to accomomdate variable length input
class PositionalEmbedding(nn.Module):
        
    def __init__(self, shape, positional_embedding_dropout=0.):

        """
        Positional embeddings are initialzed and added to the input tensors. 
  
        Parameters:
            `shape` (tuple of ints): Shape of the Positional Embedding
            `positional_embedding_dropout` (float): dropout probability for the positional embeddings (default 0.0)
          
        """

        super(PositionalEmbedding, self).__init__()

        self.pos_shape = shape
        self.positional_embedding = nn.Parameter(torch.zeros(*shape)) 
        self.positional_embedding_dropout = nn.Dropout(p=positional_embedding_dropout)
    
    def forward(self, x):

        """
        Adds positional embeddings to the input tensors. 
  
        Parameters:
           x (tensor): Tensor of dimension (batch_size, num_tokens, d_model) OR (batch_size, num_frames, num_patches, d_model)

        Returns:
            x (tensor): Tensor of dimension (batch_size, num_tokens, d_model) OR (batch_size, num_frames, num_patches, d_model)

        """

        _, num_tokens, d_model = x.shape
        
        assert self.pos_shape[1] >= num_tokens, f"Dimension 1 (num_tokens) of positional embedding {self.pos_shape[1]} set at initialisation time should be >= the dimension 1 (num_tokens) of input {num_tokens}."
        assert self.pos_shape[2] == d_model, f"Dimension 2 (d_model) of positional embedding {self.pos_shape[2]} set at initialisation time does not match the dimension 2 (d_model) of input {d_model}."
        
        x = self.positional_embedding_dropout(x + self.positional_embedding[:, :num_tokens]) 

        return x
        

class Attention(nn.Module):

    def __init__(self, d_model, num_heads=12, qkv_bias=False, attention_dropout=0., projection_dropout=0., init=''):

        """
        Initialises all the attributes of the for the multi-headed attention block. 
  
        Parameters:
            `d_model` (int): Dimension of the tensors used to compute attention
            `num_heads` (int): Number of attention heads. (default 12)
            `qkv_bias` (boolean): Determines whether to use bias as part of the query/key/value linear layers in the attention block (default True)
            `attention_dropout` (float): Dropout probability for the layer after the multi-head attention mechanism (default 0.0)
            `projection_dropout` (float): Dropout probability for the layer after the projection layer (default 0.0)
        """

        super(Attention, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert d_model == self.head_dim * num_heads, "The model dimension must be divisible by the number of heads."

        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, d_model * 3, bias=qkv_bias)
        self.attention_dropout = nn.Dropout(attention_dropout)

        self.projection_layer = nn.Linear(d_model, d_model)
        self.projection_dropout = nn.Dropout(projection_dropout)

    # src_mask not yet added
    def forward(self, x, mask=None):

        """
        Performs a forward pass on the multi-headed attention block followed by a linear (projection) layer.
  
        Parameters:
            x (tensor): Tensor of dimension (batch_size, num_tokens, d_model)
        
        Returns:
            x (tensor): Tensor of dimension (batch_size, num_tokens, d_model)

        """

        batch_size, num_tokens, d_model = x.shape
        
        qkv = self.qkv(x) # (batch_size, num_tokens, dim * 3)
        qkv = qkv.reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, batch_size, num_heads, num_tokens, head_dim)

        query, key, value = qkv.unbind(0) 
        
        # (batch_size, num_heads, num_tokens, head_dim) * (batch_size, num_heads, head_dim, num_tokens) 
        # -> (batch_size, num_heads, num_tokens, num_tokens)
        self_attention = torch.matmul(query, key.transpose(-2, -1))

        if mask is not None:
            self_attention = self_attention.masked_fill(mask == 0, float("-1e20"))
            
        self_attention = (self_attention * self.scale).softmax(dim=-1)
        self_attention = self.attention_dropout(self_attention)

        weighted_attention = torch.matmul(self_attention, value) # (batch_size, num_heads, num_tokens, head_dim)
        weighted_attention = weighted_attention.transpose(1, 2).flatten(2) # (batch_size, num_tokens, d_model)
        
        x = self.projection_layer(weighted_attention) # (batch_size, num_tokens, d_model)
        x = self.projection_dropout(x)

        return x

# TODO - add mask
class DotProductAttention(nn.Module):
    def __init__(self, d_model, num_heads=12, qkv_bias=False, attention_dropout=0., projection_dropout=0.):

        """
        Initialises all the attributes for the Dot Product Attention architecture. 
  
        Parameters:
            `d_model` (int): Dimension of the tensors used to compute attention
            `num_heads` (int): Number of attention heads. (default 12)
            `qkv_bias` (boolean): Determines whether to use bias as part of the query/key/value linear layers in the attention block (default True)
            `attention_dropout` (float): Dropout probability for the layer after the multi-head attention mechanism (default 0.0)
            `projection_dropout` (float): Dropout probability for the layer after the projection layer (default 0.0)
          
        """

        super(DotProductAttention, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = self.d_model // self.num_heads
        assert d_model == self.head_dim * self.num_heads, "The model dimension must be divisible by the number of heads."

        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, d_model * 3, bias=qkv_bias)
        self.attention_dropout = nn.Dropout(attention_dropout)

        self.projection_layer = nn.Linear(d_model, d_model)
        self.projection_dropout = nn.Dropout(projection_dropout)

    # masks not yet added
    def forward(self, x):

        """
        Performs a forward pass on the Dot Product Attention block which fuses the spatial and temporal attention outputs.
  
        Parameters:
            x (tensor): Tensor of dimension (batch_size, num_frames, num_patches, d_model)
        
        Returns:
            x (tensor): Tensor of dimension (batch_size, num_frames, num_patches, d_model)

        """

        batch_size, num_frames, num_patches, d_model = x.shape
        x = x.reshape(batch_size, -1, d_model) # (batch_size, num_frames * num_patches, d_model)
        
        qkv = self.qkv(x) # (batch_size, num_frames * num_patches, d_model * 3)
        qkv = qkv.reshape(batch_size, num_frames * num_patches, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, batch_size, num_heads, num_frames * num_patches, head_dim)

        query, key, value = qkv.unbind(0) # (batch_size, num_heads, num_frames * num_patches, head_dim)

        query_spatial, query_temporal = query.chunk(2, dim=1) # (batch_size, num_heads/2, num_frames * num_patches, head_dim)
        key_spatial, key_temporal = key.chunk(2, dim=1) # (batch_size, num_heads/2, num_frames * num_patches, head_dim)
        value_spatial, value_temporal = value.chunk(2, dim=1) # (batch_size, num_heads/2, num_frames * num_patches, head_dim)

        query_spatial = query_spatial.reshape(batch_size * num_frames, self.num_heads // 2, 
                                            num_patches, -1)
        key_spatial = key_spatial.reshape(batch_size * num_frames, self.num_heads // 2, 
                                            num_patches, -1)
        value_spatial = value_spatial.reshape(batch_size * num_frames, self.num_heads // 2, 
                                            num_patches, -1)

        query_temporal = query_temporal.reshape(batch_size * num_patches, self.num_heads // 2, 
                                            num_frames, -1)
        key_temporal = key_temporal.reshape(batch_size * num_patches, self.num_heads // 2, 
                                            num_frames, -1)
        value_temporal = value_temporal.reshape(batch_size * num_patches, self.num_heads // 2, 
                                            num_frames, -1)
        
        # (batch_size * num_frames, num_heads/2, num_patches, head_dim) * (batch_size * num_frames, num_heads/2, head_dim, num_patches) 
        # -> (batch_size * num_frames, num_heads/2, num_patches, num_patches)
        self_attention_spatial = torch.matmul(query_spatial, key_spatial.transpose(-2, -1)) * self.scale
        self_attention_spatial = self_attention_spatial.softmax(dim=-1)
        self_attention_spatial = self.attention_dropout(self_attention_spatial)

        weighted_attention_spatial = torch.matmul(self_attention_spatial, value_spatial) # (batch_size * num_frames, num_heads/2, num_patches, head_dim)
        
        # (batch_size * num_patches, num_heads/2, num_frames, head_dim) * (batch_size * num_patches, num_heads/2, head_dim, num_frames) 
        # -> (batch_size * num_patches, num_heads/2, num_frames, num_frames)
        self_attention_temporal = torch.matmul(query_temporal, key_temporal.transpose(-2, -1)) * self.scale
        self_attention_temporal = self_attention_temporal.softmax(dim=-1)
        self_attention_temporal = self.attention_dropout(self_attention_temporal)

        weighted_attention_temporal = torch.matmul(self_attention_temporal, value_temporal) # (batch_size * num_patches, num_heads/2, num_frames , head_dim)

        weighted_attention_spatial = weighted_attention_spatial.reshape(batch_size, self.num_heads // 2, num_frames * num_patches, -1)
        weighted_attention_temporal = weighted_attention_temporal.reshape(batch_size, self.num_heads // 2, num_frames * num_patches, -1)

        weighted_attention = torch.cat((weighted_attention_spatial, weighted_attention_temporal), dim=1) # (batch_size, num_heads, num_frames * num_patches, head_dim)

        weighted_attention = weighted_attention.transpose(1, 2).flatten(2) # (batch_size, num_frames * num_patches, d_model)
        
        x = self.projection_layer(weighted_attention) # (batch_size, num_frames * num_patches, d_model)
        x = self.projection_dropout(x)

        x = x.reshape(batch_size, num_frames, num_patches, d_model)

        return x


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout_1=0., dropout_2=0.):

        """
        Multi-layer perceptron which consists of 2 fully connected layers.
  
        Parameters:
            `in_dim` (int): Input dimension of the MLP block
            `hidden_dim` (int): Dimension of the intermediate layer
            `out_dim` (int): Output dimension of the MLP block
            `drouput_1` (float): Dropout probability applied after the first fully connected layer in the MLP block (default 0.0)
            `drouput_2` (float): Dropout probability applied after the second fully connected layer in the MLP block (default 0.0)
            
        """

        super(MLP, self).__init__()

        self.fully_connected_1 = nn.Linear(in_dim, hidden_dim)
        self.activation_layer = nn.GELU()
        self.dropout_1 = nn.Dropout(dropout_1)
        self.fully_connected_2 = nn.Linear(hidden_dim, out_dim)
        self.dropout_2 = nn.Dropout(dropout_2)

    def forward(self, x):

        """
        Performs a forward pass on the Multi-layer perceptron.

        Parameters:
            x (tensor): Tensor of dimension (batch_size, num_tokens, d_model)
        
        Returns:
            x (tensor): Tensor of dimension (batch_size, num_tokens, d_model)

        """

        x = self.fully_connected_1(x) # (batch_size, num_tokens, hidden_dim)
        x = self.activation_layer(x)
        x = self.dropout_1(x) 
        x = self.fully_connected_2(x)  # (batch_size, num_tokens, out_dim)
        x = self.dropout_2(x) 

        return x

class FFN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout=0.):

        """
        Feed Forward Network with 'n' layers
  
        Parameters:
            `in_dim` (int): Input dimension of the MLP block
            `hidden_dim` (int): Dimension of the intermediate layer
            `out_dim` (int): Output dimension of the MLP block
            `num_layers` (int): Depth of FFN
            `drouput` (float): Dropout probability applied after the first fully connected layer in the MLP block (default 0.0)
            
        """

        super(FFN, self).__init__()

        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([in_dim] + h, h + [out_dim]))
        self.relu = nn.ReLU()

    def forward(self, x):

        """
        Performs a forward pass on the Feed Forward Network.
        Parameters:
            x (tensor): Tensor of dimension (batch_size, num_tokens, d_model)
        
        Returns:
            x (tensor): Tensor of dimension (batch_size, num_tokens, d_model)
        """

        for i, layer in enumerate(self.layers):
            x = self.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# TODO- dropout before/after each layer_norm?
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio=4., qkv_bias=False, dropout_1=0., dropout_2=0., 
                attention_dropout=0., projection_dropout=0., pre_norm=True):

        """
        EncoderLayer consisting of the basic attention architecture.
  
        Parameters:
            `d_model` (int): Dimension of the tensors used to compute attention
            `num_heads` (int): Number of attention heads. 
            `mlp_ratio` (int): Used to determine the hidden layer dimension of the MLP. (default 4)
            `qkv_bias` (boolean): Determines whether to use bias as part of the query/key/value linear layers in the attention block (default True)
            `dropout_1` (float): Dropout probability for the MLP block (default 0.0)
            `dropout_2` (float): Dropout probability for the MLP block (default 0.0)
            `attention_dropout` (float): Dropout probability for the layer after the multi-head attention mechanism (default 0.0)
            `projection_dropout` (float): Dropout probability for the layer after the projection layer (default 0.0)
            `pre_norm` (boolean): If True, the normalisation layer would be placed before the attention and mlp blocks. Else, after them. (default True)
        """

        super(EncoderLayer, self).__init__()

        self.pre_norm = pre_norm

        #eps for compatibility with ViT pretrained weights??
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6) 

        self.attention = Attention(d_model, num_heads=num_heads, qkv_bias=qkv_bias, 
                                   attention_dropout=attention_dropout, projection_dropout=projection_dropout)

        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)

        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mlp = MLP(in_dim=d_model, hidden_dim=mlp_hidden_dim, out_dim=d_model, 
                       dropout_1=dropout_1, dropout_2=dropout_2)

    def forward(self, x):

        """
        Performs a forward pass on the Factorised Encoder block.
  
        Parameters:
            x (tensor): Tensor of dimension (batch_size, num_tokens, d_model)
        
        Returns:
            x (tensor): Tensor of dimension (batch_size, num_tokens, d_model)

        """

        if self.pre_norm:
            return self.forward_pre(x)
        
        else:
            return self.forward_post(x)


    def forward_pre(self, x):

        """
        Performs a forward pass with pre-norm on the Factorised Encoder block.
  
        Parameters:
            x (tensor): Tensor of dimension (batch_size, num_tokens, d_model)
        
        Returns:
            x (tensor): Tensor of dimension (batch_size, num_tokens, d_model)

        """

        x = x + self.attention(self.layer_norm_1(x)) # (batch_size, num_tokens, d_model)
        x = x + self.mlp(self.layer_norm_2(x)) # (batch_size, num_tokens, d_model)

        return x
    
    
    def forward_post(self, x):

        """
        Performs a forward pass with post-norm on the Factorised Encoder block.
  
        Parameters:
            x (tensor): Tensor of dimension (batch_size, num_tokens, d_model)
        
        Returns:
            x (tensor): Tensor of dimension (batch_size, num_tokens, d_model)

        """

        x = self.layer_norm_1(x + self.attentio(x)) # (batch_size, num_tokens, d_model)
        x = self.layer_norm_2(x + self.mlp(x)) # (batch_size, num_tokens, d_model)

        return x


# TODO - forward pre-post
class FactorisedSelfAttentionEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio=4., qkv_bias=False, 
                attention_dropout=0., projection_dropout=0., dropout_1=0., dropout_2=0.):
        
        """
        Attention architecture consisting of spatial attention followed by temporal attention within one block.
    
        Parameters:
            `d_model` (int): Dimension of the tensors used to compute attention
            `num_heads` (int): Number of attention heads.
            `mlp_ratio` (int): Used to determine the hidden layer dimension of the MLP. (default 4)
            `qkv_bias` (boolean): Determines whether to use bias as part of the query/key/value linear layers in the attention block (default True)
            `attention_dropout` (float): Dropout probability for the layer after the multi-head attention mechanism (default 0.0)
            `projection_dropout` (float): Dropout probability for the layer after the projection layer (default 0.0)
            `dropout_1` (float): Dropout probability for the MLP block (default 0.0)
            `dropout_2` (float): Dropout probability for the MLP block (default 0.0)
    
        """

        super(FactorisedSelfAttentionEncoderLayer, self).__init__()

        #eps for compatibility with ViT pretrained weights??
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        
        self.spatial_attention = Attention(d_model, num_heads=num_heads, qkv_bias=qkv_bias, 
                                   attention_dropout=attention_dropout, projection_dropout=projection_dropout)

        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)

        self.temporal_attention = Attention(d_model, num_heads=num_heads, qkv_bias=qkv_bias, 
                                   attention_dropout=attention_dropout, projection_dropout=projection_dropout)

        self.layer_norm_3 = nn.LayerNorm(d_model, eps=1e-6)

        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mlp = MLP(in_dim=d_model, hidden_dim=mlp_hidden_dim, out_dim=d_model, 
                       dropout_1=dropout_1, dropout_2=dropout_2)

    def forward(self, x):

        """
        Performs a forward pass on the Factorised Self-Attention block.
  
        Parameters:
            x (tensor): Tensor of dimension (batch_size, num_frames, num_patches, d_model)
        
        Returns:
            x (tensor): Tensor of dimension (batch_size, num_frames, num_patches, d_model)

        """

        batch_size, num_frames, num_patches, d_model = x.shape
        
        x = x.reshape(-1, num_patches, d_model) # (batch_size * num_frames, num_patches, d_model)

        x = x + self.spatial_attention(self.layer_norm_1(x)) # (batch_size * num_frames, num_patches, d_model)

        x = x.reshape(-1, num_frames, d_model) # (batch_size * num_patches, num_frames, d_model)

        x = x + self.temporal_attention(self.layer_norm_2(x)) # (batch_size * num_patches, num_frames, d_model)

        x = x + self.mlp(self.layer_norm_3(x)) # (batch_size * num_patches, num_frames, d_model)

        x = x.reshape(batch_size, num_frames, num_patches, d_model) # (batch_size, num_frames, num_patches, d_model)

        return x


# TODO - forward pre-post
class FactorisedDotProductAttentionEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio=4., qkv_bias=False, 
                attention_dropout=0., projection_dropout=0., dropout_1=0., dropout_2=0.):
        
        """
        Attention architecture consisting of spatial attention fused with temporal attention within one block.
  
        Parameters:
            `d_model` (int): Dimension of the tensors used to compute attention
            `depth` (int): number of encoder blocks. 
            `num_heads` (int): Number of attention heads.
            `mlp_ratio` (int): Used to determine the hidden layer dimension of the MLP. (default 4)
            `qkv_bias` (boolean): Determines whether to use bias as part of the query/key/value linear layers in the attention block (default True)
            `attention_dropout` (float): Dropout probability for the layer after the multi-head attention mechanism (default 0.0)
            `projection_dropout` (float): Dropout probability for the layer after the projection layer (default 0.0)
            `dropout_1` (float): Dropout probability for the MLP block (default 0.0)
            `dropout_2` (float): Dropout probability for the MLP block (default 0.0)
    
        """

        super(FactorisedDotProductAttentionEncoderLayer, self).__init__()
        
        #eps for compatibility with ViT pretrained weights??
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6) 

        self.attention = DotProductAttention(d_model, num_heads=num_heads, qkv_bias=qkv_bias, 
                                   attention_dropout=attention_dropout, projection_dropout=projection_dropout)
                                   
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)

        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mlp = MLP(in_dim=d_model, hidden_dim=mlp_hidden_dim, out_dim=d_model, 
                       dropout_1=dropout_1, dropout_2=dropout_2)

    def forward(self, x):

        """
        Performs a forward pass on the Factorised Dot Product Attention block.
  
        Parameters:
            x (tensor): Tensor of dimension (batch_size, num_frames, num_patches, d_model)
        
        Returns:
            x (tensor): Tensor of dimension (batch_size, num_frames, num_patches, d_model)

        """

        batch_size, num_frames, num_patches, d_model = x.shape

        x = x + self.attention(self.layer_norm_1(x)) # (batch_size, num_frames, num_patches, d_model)

        x = x.reshape(batch_size, -1, d_model)

        x = x + self.mlp(self.layer_norm_2(x)) # (batch_size, num_frames * num_patches, d_model)

        x = x.reshape(batch_size, num_frames, num_patches, d_model)

        return x


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


# --------------------------------------------------------------
# Modules used by the Decoder and Bi-modal encoder

class CrossAttention(nn.Module):

    def __init__(self, d_model, num_heads=12, qkv_bias=False, attention_dropout=0., projection_dropout=0.):

        """
        Initialises all the attributes of the for the cross attention block which involves different modalities (eg. video and audio). 
  
        Parameters:
            `d_model` (int): Dimension of the tensors used to compute attention
            `num_heads` (int): Number of attention heads. (default 12)
            `qkv_bias` (boolean): Determines whether to use bias as part of the query/key/value linear layers in the attention block (default True)
            `attention_dropout` (float): Dropout probability for the layer after the multi-head attention mechanism (default 0.0)
            `projection_dropout` (float): Dropout probability for the layer after the projection layer (default 0.0)
        """

        super(CrossAttention, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert d_model == self.head_dim * num_heads, "The model dimension must be divisible by the number of heads."

        self.scale = self.head_dim ** -0.5

        self.q_linear = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.v_linear = nn.Linear(d_model, d_model, bias=qkv_bias)

        self.attention_dropout = nn.Dropout(attention_dropout)

        self.projection_layer = nn.Linear(d_model, d_model)
        self.projection_dropout = nn.Dropout(projection_dropout)

    # src_mask not yet added
    def forward(self, q, k, v, mask=None):

        """
        Performs a forward pass on the cross attention block followed by a linear (projection) layer.
  
        Parameters:
            q (tensor): Tensor of dimension (batch_size, num_tokens_q, d_model) represeting a query vector
            k (tensor): Tensor of dimension (batch_size, num_tokens_k, d_model) represeting a key vector
            v (tensor): Tensor of dimension (batch_size, num_tokens_v, d_model) represeting a value vector

        Returns:
            x (tensor): Tensor of dimension (batch_size, num_tokens_q, d_model)

        """

        assert k.shape == v.shape, f"The keys and values inputted to the cross attention module should have the same shape. However, key has {k.shape} and value has {v.shape}."

        batch_size, num_tokens_q, _ = q.shape
        _, num_tokens_k, _ = k.shape
        
        q = self.q_linear(q) # (batch_size, num_tokens_q, d_model)
        k = self.k_linear(k) # (batch_size, num_tokens_k, d_model)
        v = self.v_linear(v) # (batch_size, num_tokens_k, d_model)

        q = q.reshape(batch_size, num_tokens_q, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3) # (batch_size, num_heads, num_tokens_q, head_dim)

        k = k.reshape(batch_size, num_tokens_k, self.num_heads, self.head_dim)
        k = k.permute(0, 2, 1, 3) # (batch_size, num_heads, num_tokens_k, head_dim)

        v = v.reshape(batch_size, num_tokens_k, self.num_heads, self.head_dim)
        v = v.permute(0, 2, 1, 3) # (batch_size, num_heads, num_tokens_k, head_dim)
        
        # (batch_size, num_heads, num_tokens_q, head_dim) * (batch_size, num_heads, head_dim, num_tokens_k) 
        # -> (batch_size, num_heads, num_tokens_q, num_tokens_k)
        cross_attention = torch.matmul(q, k.transpose(-2, -1))

        if mask is not None:
            cross_attention = cross_attention.masked_fill(mask == 0, float("-1e20"))

        cross_attention = (cross_attention * self.scale).softmax(dim=-1)
        cross_attention = self.attention_dropout(cross_attention)

        weighted_cross_attention = torch.matmul(cross_attention, v) # (batch_size, num_heads, num_tokens_q, head_dim)
        weighted_cross_attention = weighted_cross_attention.transpose(1, 2).flatten(2) # (batch_size, num_tokens_q, d_model)
        
        x = self.projection_layer(weighted_cross_attention) # (batch_size, num_tokens_q, d_model)
        x = self.projection_dropout(x)

        return x

# TODO- dropout before/after each layer_norm?
# TODO- check forward_pre sequence for self_attention (which one of q, k, v should have layer_norm?)
class BiModalEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio=4., qkv_bias=False,
                attention_dropout=0., projection_dropout=0., dropout_1=0., dropout_2=0., pre_norm=True):

        """
        Bi-modal encoder block for cross attention b/w video and audio features.
  
        Parameters:
            `d_model` (int): Dimension of the tensors used to compute attention
            `num_heads` (int): Number of attention heads.
            `mlp_ratio` (int): Used to determine the hidden layer dimension of the MLP. (default 4)
            `qkv_bias` (boolean): Determines whether to use bias as part of the query/key/value linear layers in the attention block (default True)
            `attention_dropout` (float): Dropout probability for the layer after the multi-head attention mechanism (default 0.0)
            `projection_dropout` (float): Dropout probability for the layer after the projection layer (default 0.0)
            `dropout_1` (float): Dropout probability for the MLP block (default 0.0)
            `dropout_2` (float): Dropout probability for the MLP block (default 0.0)
            `pre_norm` (boolean): If True, the normalisation layer would be placed before the attention and mlp blocks. Else, after them. (default True)
    
        """
        
        super(BiModalEncoderLayer, self).__init__()

        self.pre_norm = pre_norm

        self.attention_av = CrossAttention(d_model, num_heads=num_heads, qkv_bias=qkv_bias, 
                                   attention_dropout=attention_dropout, projection_dropout=projection_dropout)

        self.attention_va = CrossAttention(d_model, num_heads=num_heads, qkv_bias=qkv_bias, 
                                   attention_dropout=attention_dropout, projection_dropout=projection_dropout)

        self.layer_norm_av_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_va_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_av_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_va_2 = nn.LayerNorm(d_model, eps=1e-6)

        mlp_hidden_dim = int(d_model * mlp_ratio)

        self.mlp_av = MLP(in_dim=d_model, hidden_dim=mlp_hidden_dim, out_dim=d_model, 
                       dropout_1=dropout_1, dropout_2=dropout_2)

        self.mlp_va = MLP(in_dim=d_model, hidden_dim=mlp_hidden_dim, out_dim=d_model, 
                       dropout_1=dropout_1, dropout_2=dropout_2)



    def forward(self, vid, aud):

        """
        Performs a forward pass over the Bi-modal encoder with video and audio features.
  
        Parameters:
            vid (tensor): Tensor of dimension (batch_size, num_frames, d_model) representing video features
            aud (tensor): Tensor of dimension (batch_size, num_tokens, d_model) representing audio features
        
        Returns:
            vid (tensor): Tensor of dimension (batch_size, num_frames, d_model) representing video features after cross attention with audio features 
            aud (tensor): Tensor of dimension (batch_size, num_tokens, d_model) representing audio features after cross attention with video features 

        """
        
        if self.pre_norm:
            return self.forward_pre(vid, aud)
        
        else:
            return self.forward_post(vid, aud)
        
    

    def forward_pre(self, vid, aud):

        """
        Performs a forward pass over the Bi-modal encoder with video and audio features and normalisation layers before attention and mlp blocks.
  
        Parameters:
            vid (tensor): Tensor of dimension (batch_size, num_frames, d_model) representing video features
            aud (tensor): Tensor of dimension (batch_size, num_tokens, d_model) representing audio features
        
        Returns:
            vid (tensor): Tensor of dimension (batch_size, num_frames, d_model) representing video features after cross attention with audio features 
            aud (tensor): Tensor of dimension (batch_size, num_tokens, d_model) representing audio features after cross attention with video features 

        """

        vid_after_norm = self.layer_norm_av_1(vid)
        aud_after_norm = self.layer_norm_va_1(aud)

        vid = vid + self.attention_av(vid_after_norm, aud_after_norm, aud_after_norm) # (batch_size, num_tokens, d_model)
        aud = aud + self.attention_va(aud_after_norm, vid_after_norm, vid_after_norm) # (batch_size, num_tokens, d_model)

        vid = vid + self.mlp_av(self.layer_norm_av_2(vid)) # (batch_size, num_tokens, d_model)
        aud = aud + self.mlp_va(self.layer_norm_va_2(aud)) # (batch_size, num_tokens, d_model)

        return vid, aud

    
    def forward_post(self, vid, aud):

        """
        Performs a forward pass over the Bi-modal encoder with video and audio features and normalisation layers after attention and mlp blocks.
  
        Parameters:
            vid (tensor): Tensor of dimension (batch_size, num_frames, d_model) representing video features
            aud (tensor): Tensor of dimension (batch_size, num_tokens, d_model) representing audio features
        
        Returns:
            vid (tensor): Tensor of dimension (batch_size, num_frames, d_model) representing video features after cross attention with audio features 
            aud (tensor): Tensor of dimension (batch_size, num_tokens, d_model) representing audio features after cross attention with video features 

        """

        vid = self.layer_norm_av_1(vid + self.attention_av(vid, aud, aud)) # (batch_size, num_tokens, d_model)
        aud = self.layer_norm_va_1(aud + self.attention_va(aud, vid, vid)) # (batch_size, num_tokens, d_model)

        vid = self.layer_norm_av_2(vid + self.mlp_av(vid)) # (batch_size, num_tokens, d_model)
        aud = self.layer_norm_va_2(aud + self.mlp_va(aud)) # (batch_size, num_tokens, d_model)

        return vid, aud


# --------------------------------------------------------------
# Modules used by the Decoder

# TODO - dropout before/after each layer_norm?
# TODO- check forward_pre sequence for self_attention (which one of q, k, v should have layer_norm?)
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio=4., qkv_bias=False,  
                attention_dropout=0., projection_dropout=0., dropout_1=0., dropout_2=0., pre_norm=True):

        """
        Decoder consisting of the basic attention architecture.
  
        Parameters:
            `d_model` (int): Dimension of the tensors used to compute attention
            `num_heads` (int): Number of attention heads. 
            `mlp_ratio` (int): Used to determine the hidden layer dimension of the MLP. (default 4)
            `qkv_bias` (boolean): Determines whether to use bias as part of the query/key/value linear layers in the attention block (default True)
            `attention_dropout` (float): Dropout probability for the layer after the multi-head attention mechanism (default 0.0)
            `projection_dropout` (float): Dropout probability for the layer after the projection layer (default 0.0)
            `dropout_1` (float): Dropout probability for the MLP block (default 0.0)
            `dropout_2` (float): Dropout probability for the MLP block (default 0.0)
            `pre_norm` (boolean): If True, the normalisation layer would be placed before the attention and mlp blocks. Else, after them. (default True)
        """

        super(DecoderLayer, self).__init__()

        self.pre_norm=pre_norm
        
        self.self_attention = CrossAttention(d_model=d_model, num_heads=num_heads, qkv_bias=qkv_bias, 
                                attention_dropout=attention_dropout, projection_dropout=projection_dropout)

        self.cross_attention = CrossAttention(d_model=d_model, num_heads=num_heads, qkv_bias=qkv_bias, 
                                attention_dropout=attention_dropout, projection_dropout=projection_dropout)

        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_3 = nn.LayerNorm(d_model, eps=1e-6)

        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mlp = MLP(in_dim=d_model, hidden_dim=mlp_hidden_dim, out_dim=d_model, 
                       dropout_1=dropout_1, dropout_2=dropout_2)

    
    def forward(self, target, memory, positional_embedding_layer, query_embedding, mask):

        """
        Performs a forward pass on the Decoder block. Calls either forward_pre() or forward_post() based on the value of self.pre_nrom
  
        Parameters:
            target (tensor): the sequence to the decoder layer, Tensor of dimension (batch_size, num_queries, d_model)
            memory: the sequence from the last layer of the encoder
            positional_embedding_layer: position embedding layer for encoder inputs
            query_embedding: event queries
        
        Returns:
            target (tensor): Tensor of dimension (batch_size, num_queries, d_model)
        """

        if self.pre_norm:
            return self.forward_pre(target, memory, positional_embedding_layer, query_embedding, mask) # (batch_size, num_queries, d_model)
        else:
            return self.forward_post(target, memory, positional_embedding_layer, query_embedding, mask) # (batch_size, num_queries, d_model)

    
    def forward_pre(self, target, memory, positional_embedding_layer, query_embedding, mask):
        
        """
        Performs a forward pass on the Decoder block with normalisation layers before attention and mlp blocks.
  
        Parameters:
            target (tensor): the sequence to the decoder layer, Tensor of dimension (batch_size, num_queries, d_model)
            memory: the sequence from the last layer of the encoder
            positional_embedding_layer: position embedding layer for encoder inputs
            query_embedding: event queries
        
        Returns:
            target (tensor): Tensor of dimension (batch_size, num_queries, d_model)
        """

        target_after_norm = self.layer_norm_1(target) 
        q = k = target_after_norm + query_embedding
        target = target + self.self_attention(q=q, k=k, v=target_after_norm, mask=mask) # (batch_size, num_queries, d_model)

        target_after_norm = self.layer_norm_2(target)
        q = target_after_norm + query_embedding
        k = positional_embedding_layer(memory)
        target = target + self.cross_attention(q=q, k=k, v=memory) # (batch_size, num_queries, d_model)
        
        target_after_norm = self.layer_norm_3(target)
        target = target + self.mlp(target_after_norm)

        return target


    def forward_post(self, target, memory, positional_embedding_layer, query_embedding, mask):

        """
        Performs a forward pass on the Decoder block with normalisation layers after attention and mlp blocks.
  
        Parameters:
            target (tensor): the sequence to the decoder layer, Tensor of dimension (batch_size, num_queries, d_model)
            memory: the sequence from the last layer of the encoder
            positional_embedding_layer: position embedding layer for encoder inputs
            query_embedding: event queries
        
        Returns:
            target (tensor): Tensor of dimension (batch_size, num_queries, d_model)
        """
       
        q = k = target + query_embedding
        target = self.layer_norm_1(target + self.self_attention(q=q, k=k, v=target, mask=mask)) # (batch_size, num_queries, d_model)

        q = target + query_embedding
        k = positional_embedding_layer(memory)
        target = self.layer_norm_2(target + self.cross_attention(q=q, k=k, v=memory)) # (batch_size, num_queries, d_model)

        target = target + self.mlp(target)
        target = self.layer_norm_3(target)

        return target


# --------------------------------------------------------------
# Modules used by the CaptionDecoder
# Taken from https://github.com/v-iashin/BMT/blob/master/model/blocks.py

# --------------------------------------------------------------
# Modules used by the Decoder

# TODO - dropout before/after each layer_norm?
# TODO - check forward_pre sequence for self_attention (which one of q, k, v should have layer_norm?)
# TODO - check pos embed for encoder -- how??
class CaptionDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio=4., qkv_bias=False,  
                attention_dropout=0., projection_dropout=0., dropout_1=0., dropout_2=0., pre_norm=True):

        """
        Decoder consisting of the basic attention architecture.
  
        Parameters:
            `d_model` (int): Dimension of the tensors used to compute attention
            `num_heads` (int): Number of attention heads. 
            `mlp_ratio` (int): Used to determine the hidden layer dimension of the MLP. (default 4)
            `qkv_bias` (boolean): Determines whether to use bias as part of the query/key/value linear layers in the attention block (default True)
            `attention_dropout` (float): Dropout probability for the layer after the multi-head attention mechanism (default 0.0)
            `projection_dropout` (float): Dropout probability for the layer after the projection layer (default 0.0)
            `dropout_1` (float): Dropout probability for the MLP block (default 0.0)
            `dropout_2` (float): Dropout probability for the MLP block (default 0.0)
            `pre_norm` (boolean): If True, the normalisation layer would be placed before the attention and mlp blocks. Else, after them. (default True)
        """

        super(CaptionDecoderLayer, self).__init__()

        self.pre_norm=pre_norm
        
        self.self_attention = CrossAttention(d_model=d_model, num_heads=num_heads, qkv_bias=qkv_bias, 
                                attention_dropout=attention_dropout, projection_dropout=projection_dropout)

        self.cross_attention = CrossAttention(d_model=d_model, num_heads=num_heads, qkv_bias=qkv_bias, 
                                attention_dropout=attention_dropout, projection_dropout=projection_dropout)

        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_3 = nn.LayerNorm(d_model, eps=1e-6)

        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mlp = MLP(in_dim=d_model, hidden_dim=mlp_hidden_dim, out_dim=d_model, 
                       dropout_1=dropout_1, dropout_2=dropout_2)

    
    def forward(self, target, memory, word_positional_embedding_layer, positional_embedding_layer, tgt_mask, memory_mask):

        """
        Performs a forward pass on the Decoder block. Calls either forward_pre() or forward_post() based on the value of self.pre_nrom
  
        Parameters:
            target (tensor): Tensor of dimension (batch_size, seq_len, vocab_size). The sequence to the decoder layer. 
            memory: Tensor of dimension (batch_size, num_tokens, d_model). The sequence from the last layer of the encoder
            word_positional_embedding_layer (nn.Module): position embedding layer for captions
            positional_embedding_layer (nn.Module): position embedding layer for encoder inputs
            tgt_mask (Tensor): Tensor of dimension (batch_size, 1, seq_len, seq_len). Target mask for the captions to be used in the self attention block
            memory_mask (Tensor): Tensor of dimension (batch_size, 1, 1, num_tokens). Memory padding mask to be used in the cross attention block
        
        Returns:
            target (tensor): Tensor of dimension (batch_size, seq_len, vocab_size)
        """

        if self.pre_norm:
            return self.forward_pre(target, memory, word_positional_embedding_layer, positional_embedding_layer, tgt_mask, memory_mask) # (batch_size, num_queries, d_model)
        else:
            return self.forward_post(target, memory, word_positional_embedding_layer, positional_embedding_layer, tgt_mask, memory_mask) # (batch_size, num_queries, d_model)

    
    def forward_pre(self, target, memory, word_positional_embedding_layer, positional_embedding_layer, tgt_mask, memory_mask):
        
        """
        Performs a forward pass on the Decoder block with normalisation layers before attention and mlp blocks.
  
        Parameters:
            target (tensor): Tensor of dimension (batch_size, seq_len, vocab_size). The sequence to the decoder layer. 
            memory: Tensor of dimension (batch_size, num_tokens, d_model). The sequence from the last layer of the encoder
            word_positional_embedding_layer (nn.Module): position embedding layer for captions
            positional_embedding_layer (nn.Module): position embedding layer for encoder inputs
            tgt_mask (Tensor): Tensor of dimension (batch_size, 1, seq_len, seq_len). Target mask for the captions to be used in the self attention block
            memory_mask (Tensor): Tensor of dimension (batch_size, 1, 1, num_tokens). Memory padding mask to be used in the cross attention block
        
        Returns:
            target (tensor): Tensor of dimension (batch_size, seq_len, vocab_size)
        """

        target_after_norm = self.layer_norm_1(target) 
        q = k = word_positional_embedding_layer(target_after_norm)
        target = target + self.self_attention(q=q, k=k, v=target_after_norm, mask=tgt_mask) # (batch_size, num_queries, d_model)

        target_after_norm = self.layer_norm_2(target)
        q = word_positional_embedding_layer(target_after_norm)
        k = positional_embedding_layer(memory)
        target = target + self.cross_attention(q=q, k=k, v=memory, mask=memory_mask) # (batch_size, num_queries, d_model)
        
        target_after_norm = self.layer_norm_3(target)
        target = target + self.mlp(target_after_norm)

        return target


    def forward_post(self, target, memory, word_positional_embedding_layer, positional_embedding_layer, tgt_mask, memory_mask):

        """
        Performs a forward pass on the Decoder block with normalisation layers after attention and mlp blocks.
  
        Parameters:
            target (tensor): Tensor of dimension (batch_size, seq_len, vocab_size). The sequence to the decoder layer. 
            memory: Tensor of dimension (batch_size, num_tokens, d_model). The sequence from the last layer of the encoder
            word_positional_embedding_layer (nn.Module): position embedding layer for captions
            positional_embedding_layer (nn.Module): position embedding layer for encoder inputs
            tgt_mask (Tensor): Tensor of dimension (batch_size, 1, seq_len, seq_len). Target mask for the captions to be used in the self attention block
            memory_mask (Tensor): Tensor of dimension (batch_size, 1, 1, num_tokens). Memory padding mask to be used in the cross attention block
        
        Returns:
            target (tensor): Tensor of dimension (batch_size, seq_len, vocab_size)
        """
       
        q = k = word_positional_embedding_layer(target)
        target = self.layer_norm_1(target + self.self_attention(q=q, k=k, v=target, mask=matgt_masksk)) # (batch_size, num_queries, d_model)

        q = word_positional_embedding_layer(target)
        k = positional_embedding_layer(memory)
        target = self.layer_norm_2(target + self.cross_attention(q=q, k=k, v=memory, mask=memory_mask)) # (batch_size, num_queries, d_model)

        target = target + self.mlp(target)
        target = self.layer_norm_3(target)

        return target


class VocabularyEmbedder(nn.Module):

    def __init__(self, vocab_size, d_model):
        super(VocabularyEmbedder, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        # replaced if pretrained weights are used
        self.embedder = nn.Embedding(vocab_size, d_model)

    def forward(self, x):  # x - tokens (B, seq_len)
        x = self.embedder(x)
        x = x * np.sqrt(self.d_model)
        return x  # (B, seq_len, d_model)

    def init_word_embeddings(self, embedding_matrix, emb_weights_req_grad=True):
        if embedding_matrix is None:
            print('Training word embeddings from scratch')
        else:
            _, pretrained_embed_dim = embedding_matrix.shape
            if self.d_model == pretrained_embed_dim:
                self.embedder = self.embedder.from_pretrained(embedding_matrix)
                self.embedder.weight.requires_grad = emb_weights_req_grad
                print(f'Glove embedding dimension is of the same size as d_model i.e. {self.d_model}')
            else:
                self.embedder = nn.Sequential(
                    nn.Embedding(self.vocab_size, pretrained_embed_dim).from_pretrained(embedding_matrix),
                    nn.Linear(pretrained_embed_dim, self.d_model),
                    nn.ReLU()
                )
                self.embedder[0].weight.requires_grad = emb_weights_req_grad
                print(f'Glove embedding dimension is {pretrained_embed_dim} and d_model is {self.d_model}')



# --------------------------------------------------------
# Modules for deformable transformer

class MSDeformAttnFunction(Function):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        # sampling_locations:(...,2), the first item of last dim means x axis corresponding to w, and second item of the last dim means y, corresponding to h.
        ctx.im2col_step = im2col_step
        output = MSDA.ms_deform_attn_forward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = \
            MSDA.ms_deform_attn_backward(
                value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output, ctx.im2col_step)

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights, return_value=False):
    # for debug and test only,
    # need to use cuda version instead
    '''
    :param value (batch_size, sum of num_token in all level, nhead, dmodel/nhead)  
    :param value_spatial_shapes (num_feature_levels, 2)
    :param sampling_locations (batch_size, sum of num_token in all level, n_heads, n_levels, n_points, 2) 
    :param attention_weights (batch_size, sum of num_token in all level, n_heads, n_levels, n_points)
    :param return_value
    '''
    N_, S_, M_, D_ = value.shape    # N_: batch size , S_: \sum_H*W, M_ : head number, D_: feature dim of each head

    _, Lq_, M_, L_, P_, _ = sampling_locations.shape  # Lq_: \sum H*W, L_: multi-scale number, P_: number of sampled key points

    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)

    sampling_grids = 2 * sampling_locations - 1 # convert value from range[0,1] to [-1, 1]
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # sampling_grid_l_: (...,2), the first item of last dim means x axis corresponding to w, and second item of the last dim means y, corresponding to h.
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='border', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)

    if return_value:
        return torch.stack(sampling_value_list, dim=-2)
    #(N_ * M_, D_, Lq_, L_* P_) * (N_*M_, 1, Lq_, L_*P_) --> (N_*M_, D_, Lq_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    return output.transpose(1, 2).contiguous()



def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points )
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2)
        grid_init = grid_init[..., 0].repeat(1, self.n_levels, self.n_points)
        for i in range(self.n_points):
            grid_init[:, :, i] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 1), range in [0, 1], including padding area
                                        or (N, Length_{query}, n_levels, 2), add additional (c, l) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} T_l, C)
        :param input_spatial_shapes        (n_levels ), [T_0, T_1, ..., T_{L-1}]
        :param input_level_start_index     (n_levels ), [0, 1_0, T_0+T_1, ...]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements
        :return output                     (N, Length_{query}, C)
        """
        
        N, Len_q, _ = query.shape   #   (batch_size, sum of num_token in all level, dmodel)
        N, Len_in, _ = input_flatten.shape  #   (batch_size, sum of num_token in all level, dmodel)
        assert input_spatial_shapes.sum() == Len_in

        value = self.value_proj(input_flatten)  # linear transformation --> nn.Linear(d_model, d_model) shape->(batch_size, sum of num_token in all level, dmodel)
        
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)   #   (batch_size, sum of num_token in all level, nhead, dmodel/nhead)    

        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)  # linear transformation --> nn.Linear(d_model, n_heads * n_levels * n_points ) shape->(batch_size, sum of num_token in all level, n_heads, n_levels, n_points)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)   #   linear transformation --> nn.Linear(d_model, n_heads * n_levels * n_points) shape->(batch_size, sum of num_token in all level, n_heads, n_levels*n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points) #   (batch_size, sum of num_token in all level, n_heads, n_levels, n_points)


        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 1:
            offset_normalizer = input_spatial_shapes
            sampling_locations = reference_points[:, :, None, :, None, 0] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None]  #   (batch_size, sum of num_token in all level, n_heads, n_levels, n_points)                   
        elif reference_points.shape[-1] == 2:
            sampling_locations = reference_points[:, :, None, :, None, 0] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 1] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 1 or 2, but get {} instead.'.format(reference_points.shape[-1]))

        if True:
            sampling_locations = torch.stack((sampling_locations, 0.5 * sampling_locations.new_ones(sampling_locations.shape)), -1)  #   (batch_size, sum of num_token in all level, n_heads, n_levels, n_points, 2) 
            input_spatial_shapes = torch.stack([input_spatial_shapes.new_ones(input_spatial_shapes.shape), input_spatial_shapes], -1)   # (num_feature_levels, 2)
            
        if query.device.type == 'cuda':
            output = MSDeformAttnFunction.apply(
                value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights,
                self.im2col_step)
        else:
            output = ms_deform_attn_core_pytorch(value, input_spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)
        return output


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor], duration=None):
        self.tensors = tensors
        self.mask = mask
        self.duration = duration

    def to(self, device, non_blocking=False):
        '''# type: (Device) -> NestedTensor # noqa'''

        cast_tensor = self.tensors.to(device, non_blocking=non_blocking)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device, non_blocking=non_blocking)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def record_stream(self, *args, **kwargs):
        self.tensors.record_stream(*args, **kwargs)
        if self.mask is not None:
            self.mask.record_stream(*args, **kwargs)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.max_duration = 384
        self.duration_embed_layer = nn.Linear(self.max_duration, self.max_duration)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        duration = tensor_list.duration
        assert mask is not None
        not_mask = ~mask
        x_embed = not_mask.cumsum(1, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            x_embed = (x_embed - 0.5) / (x_embed[:, -1:] + eps) * self.scale
        
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)
        pos_x = x_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        dur_embed = self.duration_embedding(duration).reshape(-1,1,self.max_duration).expand_as(pos_x)
        pos = torch.cat((pos_x, dur_embed), dim=2).permute(0, 2, 1)
        return pos

    def duration_embedding(self, durations):
        out = torch.zeros(len(durations), self.max_duration, device=durations.device)
        durations = durations.int()
        for ii in range(len(durations)):
            out[ii, :durations[ii]] = 1
        out = self.duration_embed_layer(out)
        return out


def build_position_encoding(position_embedding, N_steps):
    if position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    else:
        raise ValueError(f"not supported {position_embedding}")

    return 
    

def decide_two_stage(transformer_input_type, gt_boxes, gt_boxes_mask, criterion):
    if transformer_input_type == 'gt_proposals':
        two_stage = True
        proposals = gt_boxes
        proposals_mask = gt_boxes_mask
        criterion.matcher.cost_caption = 0
        for q_k in ['loss_length', 'loss_ce', 'loss_bbox', 'loss_giou']:
            for key in criterion.weight_dict.keys():
                if q_k in key:
                    criterion.weight_dict[key] = 0
        disable_iterative_refine = True
    elif transformer_input_type == 'queries':  #
        two_stage = False
        proposals = None
        proposals_mask = None
        disable_iterative_refine = False
    else:
        raise ValueError('Wrong value of transformer_input_type, got {}'.format(transformer_input_type))
    return two_stage, disable_iterative_refine, proposals, proposals_mask