""" Encoder and Decoder layers """

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
from torch.nn import MultiheadAttention
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

from .attention import *



# TODO- dropout before/after each layer_norm?
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio=4., qkv_bias=False, mlp_dropout_1=0., mlp_dropout_2=0., 
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

        self.self_attention = CrossAttention(d_model=d_model, num_heads=num_heads, qkv_bias=qkv_bias, 
                                attention_dropout=attention_dropout, projection_dropout=projection_dropout)

        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)

        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mlp = MLP(in_dim=d_model, hidden_dim=mlp_hidden_dim, out_dim=d_model, 
                       dropout_1=mlp_dropout_1, dropout_2=mlp_dropout_2)

    def forward(self, x, positional_embedding, src_mask=None, src_padding_mask=None):

        """
        Performs a forward pass on the Factorised Encoder block.
  
        Parameters:
            x (tensor): Tensor of dimension (batch_size, num_tokens, d_model)
        
        Returns:
            x (tensor): Tensor of dimension (batch_size, num_tokens, d_model)

        """

        if self.pre_norm:
            return self.forward_pre(x, positional_embedding, src_mask, src_padding_mask)
        
        else:
            return self.forward_post(x, positional_embedding, src_mask, src_padding_mask)


    def forward_pre(self, x, positional_embedding, src_mask=None, src_padding_mask=None):

        """
        Performs a forward pass with pre-norm on the Factorised Encoder block.
  
        Parameters:
            x (tensor): Tensor of dimension (batch_size, num_tokens, d_model)
        
        Returns:
            x (tensor): Tensor of dimension (batch_size, num_tokens, d_model)

        """

        # x = x + self._sa_block(self.layer_norm_1(x), attn_mask=src_mask, key_padding_mask=src_padding_mask)    # (batch_size, num_tokens, d_model)
        # x = x + self.mlp(self.layer_norm_2(x))    # (batch_size, num_tokens, d_model)

        # return x

        q, k, v = positional_embedding(self.layer_norm_1(x)), positional_embedding(self.layer_norm_1(x)), self.layer_norm_1(x)

        x = x + self.self_attention(q=q, k=k, v=v, attn_mask=src_mask, key_padding_mask=src_padding_mask, need_weights=False)[0]    # (batch_size, num_tokens, d_model)

        x = x + self.mlp(self.layer_norm_2(x))

        return x
    
    
    def forward_post(self, x, positional_embedding, src_mask=None, src_padding_mask=None):

        """
        Performs a forward pass with post-norm on the Factorised Encoder block.
  
        Parameters:
            x (tensor): Tensor of dimension (batch_size, num_tokens, d_model)
        
        Returns:
            x (tensor): Tensor of dimension (batch_size, num_tokens, d_model)

        """

        # x = self.layer_norm_1(x + self._sa_block(self.layer_norm_1(x), attn_mask=src_mask, key_padding_mask=src_padding_mask))    # (batch_size, num_tokens, d_model)
        # x = self.layer_norm_2(x + self.mlp(x))    # (batch_size, num_tokens, d_model)

        # return x

        q, k, v = positional_embedding(x), positional_embedding(x), x
        
        x = self.layer_norm_1(x + self.self_attention(q=q, k=k, v=v, attn_mask=src_mask, key_padding_mask=src_padding_mask, need_weights=False)[0])    # (batch_size, num_tokens, d_model)

        x = self.layer_norm_2(x + self.mlp(x))

        return x



# TODO- dropout before/after each layer_norm?
# TODO- check forward_pre sequence for self_attention (which one of q, k, v should have layer_norm?)
class BiModalEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio=4., qkv_bias=False,
                attention_dropout=0., projection_dropout=0., mlp_dropout_1=0., mlp_dropout_2=0., pre_norm=True):

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
                       dropout_1=mlp_dropout_1, dropout_2=mlp_dropout_2)



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


# TODO - dropout before/after each layer_norm?
# TODO- check forward_pre sequence for self_attention (which one of q, k, v should have layer_norm?)
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio=4., qkv_bias=False,  
                attention_dropout=0., projection_dropout=0., mlp_dropout_1=0., mlp_dropout_2=0., pre_norm=True):

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
                       dropout_1=mlp_dropout_1, dropout_2=mlp_dropout_2)

    
    def forward(self, tgt, memory, positional_embedding, query_embedding, tgt_mask=None, memory_mask=None, tgt_padding_mask=None, memory_padding_mask=None):

        """
        Performs a forward pass on the Decoder block. Calls either forward_pre() or forward_post() based on the value of self.pre_nrom
  
        Parameters:
            tgt (tensor): the sequence to the decoder layer, Tensor of dimension (batch_size, num_queries, d_model)
            memory: the sequence from the last layer of the encoder
            positional_embedding: position embedding for encoder inputs
            query_embedding: event queries
        
        Returns:
            tgt (tensor): Tensor of dimension (batch_size, num_queries, d_model)
        """

        if self.pre_norm:
            return self.forward_pre(tgt, memory, positional_embedding, query_embedding, tgt_mask, memory_mask, tgt_padding_mask, memory_padding_mask)    # (batch_size, num_queries, d_model)
        else:
            return self.forward_post(tgt, memory, positional_embedding, query_embedding, tgt_mask, memory_mask, tgt_padding_mask, memory_padding_mask)    # (batch_size, num_queries, d_model)

    
    def forward_pre(self, tgt, memory, positional_embedding, query_embedding, tgt_mask=None, memory_mask=None, tgt_padding_mask=None, memory_padding_mask=None):

        """
        Performs a forward pass on the Decoder block with normalisation layers before attention and mlp blocks.
  
        Parameters:
            tgt (tensor): the sequence to the decoder layer, Tensor of dimension (batch_size, num_queries, d_model)
            memory: the sequence from the last layer of the encoder
            positional_embedding: position embedding for encoder inputs
            query_embedding: event queries
        
        Returns:
            tgt (tensor): Tensor of dimension (batch_size, num_queries, d_model)
        """

        tgt_after_norm = self.layer_norm_1(tgt) 
        q = k = tgt_after_norm + query_embedding
        tgt = tgt + self.self_attention(q=q, k=k, v=tgt_after_norm, attn_mask=tgt_mask, key_padding_mask=tgt_padding_mask, need_weights=False)[0]    # (batch_size, num_queries, d_model)

        tgt_after_norm = self.layer_norm_2(tgt)
        q = tgt_after_norm + query_embedding
        k = positional_embedding(memory)
        tgt = tgt + self.cross_attention(q=q, k=k, v=memory, attn_mask=memory_mask, key_padding_mask=memory_padding_mask, need_weights=False)[0]    # (batch_size, num_queries, d_model)) # (batch_size, num_queries, d_model)
        
        tgt_after_norm = self.layer_norm_3(tgt)
        tgt = tgt + self.mlp(tgt_after_norm)

        return tgt


    def forward_post(self, tgt, memory, positional_embedding, query_embedding, tgt_mask=None, memory_mask=None, tgt_padding_mask=None, memory_padding_mask=None):

        """
        Performs a forward pass on the Decoder block with normalisation layers after attention and mlp blocks.
  
        Parameters:
            tgt (tensor): the sequence to the decoder layer, Tensor of dimension (batch_size, num_queries, d_model)
            memory: the sequence from the last layer of the encoder
            positional_embedding: position embedding for encoder inputs
            query_embedding: event queries
        
        Returns:
            tgt (tensor): Tensor of dimension (batch_size, num_queries, d_model)
        """
       
        q = k = tgt + query_embedding
        tgt = self.layer_norm_1(tgt + self.self_attention(q=q, k=k, v=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_padding_mask, need_weights=False)[0])    # (batch_size, num_queries, d_model)

        q = tgt + query_embedding
        k = positional_embedding(memory)
        tgt = self.layer_norm_2(tgt + self.cross_attention(q=q, k=k, v=memory, attn_mask=memory_mask, key_padding_mask=memory_padding_mask, need_weights=False)[0])    # (batch_size, num_queries, d_model)

        tgt = self.layer_norm_3(tgt + self.mlp(tgt))

        return tgt


class BiModalDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio=4., qkv_bias=False,  
                attention_dropout=0., projection_dropout=0., mlp_dropout_1=0., mlp_dropout_2=0., pre_norm=True):

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
        
        self.cross_attention_video = CrossAttention(d_model=d_model, num_heads=num_heads, qkv_bias=qkv_bias, 
                                attention_dropout=attention_dropout, projection_dropout=projection_dropout)

        self.cross_attention_audio = CrossAttention(d_model=d_model, num_heads=num_heads, qkv_bias=qkv_bias, 
                                attention_dropout=attention_dropout, projection_dropout=projection_dropout)

        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_3 = nn.LayerNorm(d_model, eps=1e-6)

        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mlp = MLP(in_dim=d_model, hidden_dim=mlp_hidden_dim, out_dim=d_model, 
                       dropout_1=mlp_dropout_1, dropout_2=mlp_dropout_2)

    
    def forward(self, tgt, video_memory, audio_memory, video_positional_embedding, audio_positional_embedding, 
                query_embedding, tgt_mask=None, video_memory_mask=None, audio_memory_mask=None, 
                tgt_padding_mask=None, video_memory_padding_mask=None, audio_memory_padding_mask=None):

        """
        Performs a forward pass on the Decoder block. Calls either forward_pre() or forward_post() based on the value of self.pre_nrom
  
        Parameters:
            tgt (tensor): the sequence to the decoder layer, Tensor of dimension (batch_size, num_queries, d_model)
            memory: the sequence from the last layer of the encoder
            positional_embedding: position embedding for encoder inputs
            query_embedding: event queries
        
        Returns:
            tgt (tensor): Tensor of dimension (batch_size, num_queries, d_model)
        """

        if self.pre_norm:
            return self.forward_pre(tgt, video_memory, audio_memory, video_positional_embedding, audio_positional_embedding, 
                                    query_embedding, tgt_mask, video_memory_mask, audio_memory_mask, 
                                    tgt_padding_mask, video_memory_padding_mask, audio_memory_padding_mask)    # (batch_size, num_queries, d_model)
        else:
            return self.forward_post(tgt, video_memory, audio_memory, video_positional_embedding, audio_positional_embedding, 
                                    query_embedding, tgt_mask, video_memory_mask, audio_memory_mask, 
                                    tgt_padding_mask, video_memory_padding_mask, audio_memory_padding_mas)    # (batch_size, num_queries, d_model)

    
    def forward_pre(self, tgt, video_memory, audio_memory, video_positional_embedding, audio_positional_embedding, 
                    query_embedding, tgt_mask=None, video_memory_mask=None, audio_memory_mask=None, 
                    tgt_padding_mask=None, video_memory_padding_mask=None, audio_memory_padding_mask=None):

        """
        Performs a forward pass on the Decoder block with normalisation layers before attention and mlp blocks.
  
        Parameters:
            tgt (tensor): the sequence to the decoder layer, Tensor of dimension (batch_size, num_queries, d_model)
            memory: the sequence from the last layer of the encoder
            positional_embedding: position embedding for encoder inputs
            query_embedding: event queries
        
        Returns:
            tgt (tensor): Tensor of dimension (batch_size, num_queries, d_model)
        """

        tgt_after_norm = self.layer_norm_1(tgt) 
        q = k = tgt_after_norm + query_embedding
        tgt = tgt + self.self_attention(q=q, k=k, v=tgt_after_norm, attn_mask=tgt_mask, key_padding_mask=tgt_padding_mask, need_weights=False)[0]    # (batch_size, num_queries, d_model)

        # video
        tgt_after_norm = self.layer_norm_2(tgt)
        q = tgt_after_norm + query_embedding
        k = positional_embedding(video_memory)
        tgt = tgt + self.cross_attention_video(q=q, k=k, v=video_memory, attn_mask=video_memory_mask, key_padding_mask=video_memory_padding_mask, need_weights=False)[0]    # (batch_size, num_queries, d_model)) # (batch_size, num_queries, d_model)

        # audio
        tgt_after_norm = self.layer_norm_2(tgt)
        q = tgt_after_norm + query_embedding
        k = positional_embedding(audio_memory)
        tgt = tgt + self.cross_attention_audio(q=q, k=k, v=audio_memory, attn_mask=audio_emory_mask, key_padding_mask=audio_memory_padding_mask, need_weights=False)[0]    # (batch_size, num_queries, d_model)) # (batch_size, num_queries, d_model)
        
        tgt_after_norm = self.layer_norm_3(tgt)
        tgt = tgt + self.mlp(tgt_after_norm)

        return tgt


    def forward_post(self, tgt, video_memory, audio_memory, video_positional_embedding, audio_positional_embedding, 
                    query_embedding, tgt_mask=None, video_memory_mask=None, audio_memory_mask=None, 
                    tgt_padding_mask=None, video_memory_padding_mask=None, audio_memory_padding_mask=None):

        """
        Performs a forward pass on the Decoder block with normalisation layers after attention and mlp blocks.
  
        Parameters:
            tgt (tensor): the sequence to the decoder layer, Tensor of dimension (batch_size, num_queries, d_model)
            memory: the sequence from the last layer of the encoder
            positional_embedding: position embedding for encoder inputs
            query_embedding: event queries
        
        Returns:
            tgt (tensor): Tensor of dimension (batch_size, num_queries, d_model)
        """
       
        q = k = tgt + query_embedding
        tgt = self.layer_norm_1(tgt + self.self_attention(q=q, k=k, v=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_padding_mask, need_weights=False)[0])    # (batch_size, num_queries, d_model)

        q = tgt + query_embedding
        k = positional_embedding(memory)
        tgt = self.layer_norm_2(tgt + self.cross_attention(q=q, k=k, v=memory, attn_mask=memory_mask, key_padding_mask=memory_padding_mask, need_weights=False)[0])    # (batch_size, num_queries, d_model)

        tgt = self.layer_norm_3(tgt + self.mlp(tgt))

        return tgt



# TODO - dropout before/after each layer_norm?
class UnimodalCaptionDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio=4., qkv_bias=False,  
                attention_dropout=0., projection_dropout=0., bridge_dropout=0., mlp_dropout_1=0., mlp_dropout_2=0., pre_norm=True):

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

        super(UnimodalCaptionDecoderLayer, self).__init__()

        self.pre_norm=pre_norm
        
        self.self_attention = CrossAttention(d_model=d_model, num_heads=num_heads, qkv_bias=qkv_bias, 
                                attention_dropout=attention_dropout, projection_dropout=projection_dropout)

        self.cross_attention = CrossAttention(d_model=d_model, num_heads=num_heads, qkv_bias=qkv_bias, 
                                attention_dropout=attention_dropout, projection_dropout=projection_dropout)

        # self.self_attention = MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=attention_dropout, 
        #                                         bias=qkv_bias, batch_first=True)
        
        # self.cross_attention = MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=attention_dropout, 
        #                                         bias=qkv_bias, batch_first=True)

        self.projection_dropout_1 = nn.Dropout(projection_dropout)
        self.projection_dropout_2 = nn.Dropout(projection_dropout)

        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_3 = nn.LayerNorm(d_model, eps=1e-6)

        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mlp = MLP(in_dim=d_model, hidden_dim=mlp_hidden_dim, out_dim=d_model, 
                       dropout_1=mlp_dropout_1, dropout_2=mlp_dropout_2)

    
    def forward(self, target, memory, tgt_mask=None, memory_mask=None, tgt_padding_mask=None, memory_padding_mask=None):

        """
        Performs a forward pass on the Decoder block. Calls either forward_pre() or forward_post() based on the value of self.pre_nrom
  
        Parameters:
            target (tensor): Tensor of dimension (batch_size, seq_len, vocab_size). The sequence to the decoder layer. 
            memory: Tensor of dimension (batch_size, num_tokens, d_model). The sequence from the last layer of the encoder
            **word_positional_embedding_layer (nn.Module): position embedding layer for captions
            **positional_embedding_layer (nn.Module): position embedding layer for encoder inputs
            tgt_mask (Tensor): Tensor of dimension (batch_size, 1, seq_len, seq_len). Target mask for the captions to be used in the self attention block
            memory_mask (Tensor): Tensor of dimension (batch_size, 1, 1, num_tokens). Memory padding mask to be used in the cross attention block
        
        Returns:
            target (tensor): Tensor of dimension (batch_size, seq_len, vocab_size)
        """

        if self.pre_norm:
            return self.forward_pre(target, memory, tgt_mask, memory_mask, tgt_padding_mask, memory_padding_mask) # (batch_size, num_tokens_tgt, d_model)
        else:
            return self.forward_post(target, memory, tgt_mask, memory_mask, tgt_padding_mask, memory_padding_mask) # (batch_size, num_tokens_tgt, d_model)

    
    def forward_pre(self, target, memory, tgt_mask=None, memory_mask=None, tgt_padding_mask=None, memory_padding_mask=None):
        
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

        x = target

        x = x + self._sa_block(self.layer_norm_1(x), tgt_mask, tgt_padding_mask)
        x = x + self._ca_block(self.layer_norm_2(x), memory, memory_mask, memory_padding_mask)
        x = x + self.mlp(self.layer_norm_3(x))

        return x


    def forward_post(self, target, memory, tgt_mask=None, memory_mask=None, tgt_padding_mask=None, memory_padding_mask=None):

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

        x = target

        x = self.layer_norm_1(x + self._sa_block(x, tgt_mask, tgt_padding_mask))
        x = self.layer_norm_2(x + self._ca_block(x, memory, memory_mask, memory_padding_mask))
        x = self.layer_norm_3(x + self.mlp(x))

        return x


    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attention(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.projection_dropout_1(x)


    def _ca_block(self, x, mem, attn_mask, key_padding_mask):
        x = self.cross_attention(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.projection_dropout_2(x)


# TODO - dropout before/after each layer_norm?
class MultimodalCaptionDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio=4., qkv_bias=False,  
                attention_dropout=0., projection_dropout=0., bridge_dropout=0., mlp_dropout_1=0., mlp_dropout_2=0., pre_norm=True):

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

        super(UnimodalCaptionDecoderLayer, self).__init__()

        self.pre_norm=pre_norm
        
        self.self_attention = MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=attention_dropout, 
                                                bias=qkv_bias, batch_first=True)
        
        self.video_cross_attention = MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=attention_dropout, 
                                                bias=qkv_bias, batch_first=True)
        
        self.audio_cross_attention = MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=attention_dropout, 
                                                bias=qkv_bias, batch_first=True)

        self.projection_dropout_1 = nn.Dropout(projection_dropout)
        self.projection_dropout_2 = nn.Dropout(projection_dropout)
        self.projection_dropout_3 = nn.Dropout(projection_dropout)

        self.linear_layer = nn.Linear(2 * d_model, d_model)
        self.activation_layer = nn.GELU()

        self.dropout = nn.Dropout(bridge_dropout)

        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_3 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_4 = nn.LayerNorm(d_model, eps=1e-6)

        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mlp = MLP(in_dim=d_model, hidden_dim=mlp_hidden_dim, out_dim=d_model, 
                       dropout_1=mlp_dropout_1, dropout_2=mlp_dropout_2)

    
    def forward(self, target, video_memory, audio_memory, 
                tgt_mask=None, video_memory_mask=None, audio_memory_mask=None, 
                tgt_padding_mask=None, video_memory_padding_mask=None, audio_memory_padding_mask=None):

        """
        Performs a forward pass on the Decoder block. Calls either forward_pre() or forward_post() based on the value of self.pre_nrom
  
        Parameters:
            target (tensor): Tensor of dimension (batch_size, seq_len, vocab_size). The sequence to the decoder layer. 
            memory: Tensor of dimension (batch_size, num_tokens, d_model). The sequence from the last layer of the encoder
            **word_positional_embedding_layer (nn.Module): position embedding layer for captions
            **positional_embedding_layer (nn.Module): position embedding layer for encoder inputs
            tgt_mask (Tensor): Tensor of dimension (batch_size, 1, seq_len, seq_len). Target mask for the captions to be used in the self attention block
            memory_mask (Tensor): Tensor of dimension (batch_size, 1, 1, num_tokens). Memory padding mask to be used in the cross attention block
        
        Returns:
            target (tensor): Tensor of dimension (batch_size, seq_len, vocab_size)
        """

        if self.pre_norm:
            return self.forward_pre(target, video_memory, audio_memory, 
                                    tgt_mask, video_memory_mask, audio_memory_mask, 
                                    tgt_padding_mask, video_memory_padding_mask, audio_memory_padding_mask) # (batch_size, num_tokens_tgt, d_model)
        else:
            return self.forward_post(target, video_memory, audio_memory, 
                                    tgt_mask, video_memory_mask, audio_memory_mask, 
                                    tgt_padding_mask, video_memory_padding_mask, audio_memory_padding_mask) # (batch_size, num_tokens_tgt, d_model)

    
    def forward_pre(self, target, video_memory, audio_memory, 
                    tgt_mask=None, video_memory_mask=None, audio_memory_mask=None, 
                    tgt_padding_mask=None, video_memory_padding_mask=None, audio_memory_padding_mask=None):
        
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

        x = target

        x = x + self._sa_block(self.layer_norm_1(x), tgt_mask, tgt_padding_mask)

        x = self.layer_norm_2(x)
        vid_x = x + self._ca_video_block(x, video_memory, video_memory_mask, video_memory_padding_mask)
        aud_x = x + self._ca_audio_block(x, audio_memory, audio_memory_mask, audio_memory_padding_mask)

        # bridge
        x = torch.cat([vid_x, aud_x], dim=-1)    # (batch_size, num_queries, 2*d_model)
        x = self.layer_norm_3(x)
        x = self.linear_layer(x)    # (batch_size, num_queries, d_model)
        x = self.dropout(x)
        x = self.activation(x)

        x = x + self.mlp(self.layer_norm_4(x))

        return x


    def forward_post(self, target, video_memory, audio_memory, 
                    tgt_mask=None, video_memory_mask=None, audio_memory_mask=None, 
                    tgt_padding_mask=None, video_memory_padding_mask=None, audio_memory_padding_mask=None):

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

        x = target

        x = self.layer_norm_1(x + self._sa_block(x, tgt_mask, tgt_padding_mask))

        vid_x = self.layer_norm_2(x + self._ca_video_block(x, video_memory, video_memory_mask, video_memory_padding_mask))
        aud_x = self.layer_norm_2(x + self._ca_audio_block(x, audio_memory, audio_memory_mask, audio_memory_padding_mask))

        # bridge
        x = torch.cat([vid_x, aud_x], dim=-1)    # (batch_size, num_queries, 2*d_model)
        x = self.linear_layer(x)    # (batch_size, num_queries, d_model)
        x = self.dropout(x)
        x = self.layer_norm_3(x)
        x = self.activation(x)
        
        x = self.layer_norm_4(x + self.mlp(x))

        return x

    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attention(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.projection_dropout_1(x)

    def _ca_video_block(self, x, mem, attn_mask, key_padding_mask):
        x = self.video_cross_attention(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.projection_dropout_2(x)
    
    def _ca_audio_block(self, x, mem, attn_mask, key_padding_mask):
        x = self.cross_attention(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.audio_projection_dropout_3(x)



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
        # self.relu = nn.ReLU()

    def forward(self, x):

        """
        Performs a forward pass on the Feed Forward Network.
        Parameters:
            x (tensor): Tensor of dimension (batch_size, num_tokens, d_model)
        
        Returns:
            x (tensor): Tensor of dimension (batch_size, num_tokens, d_model)
        """

        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class ContextMaskModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        """
        Predict mask for context features (0 if num_token not useful, 1 otherwise)
        Parameters:
            `in_dim` (int): Input dimension (num_queries*2 + num_queries*d_model)
            `out_dim` (int): Output dimension (num_queries*num_tokens)

        """

        super(ContextMaskModel, self).__init__()

        self.layer_1 = nn.Linear(in_dim, in_dim // 2)
        # self.batch_norm_1 = nn.BatchNorm1d(in_dim // 2)
        self.layer_2 = nn.Linear(in_dim // 2, in_dim // 2)
        # self.batch_norm_2 = nn.BatchNorm1d(in_dim // 2)
        self.layer_3 = nn.Linear(in_dim // 2, out_dim)
        # self.batch_norm_3 = nn.BatchNorm1d(out_dim)

        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Performs a forward pass on the Context Mask Model.
        Parameters:
            x (tensor): Tensor of dimension (batch_size, in_dim) = (batch_size, num_queries*2 + num_queries*d_model)
        
        Returns:
            x (tensor): Tensor of dimension (batch_size, out_dim) = (batch_size, num_queries*num_tokens)
        """

        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.layer_3(x)

        return x



# TODO - forward pre-post
class FactorisedSelfAttentionEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio=4., qkv_bias=False, 
                attention_dropout=0., projection_dropout=0., mlp_dropout_1=0., mlp_dropout_2=0.):
        
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
                       dropout_1=mlp_dropout_1, dropout_2=mlp_dropout_2)

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
                attention_dropout=0., projection_dropout=0., mlp_dropout_1=0., mlp_dropout_2=0.):
        
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
                       dropout_1=mlp_dropout_1, dropout_2=mlp_dropout_2)

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