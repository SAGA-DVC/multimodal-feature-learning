""" Modules for ViViT in PyTorch """

import torch
import torch.nn as nn


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

        self.num_patches = (img_size // spatial_patch_size) ** 2

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


class PositionalEmbedding(nn.Module):
        
    def __init__(self, num_frames, num_patches, d_model, positional_embedding_dropout=0.):

        """
        Positional embeddings are initialzed and added to the input tensors. 
  
        Parameters:
            `num_frames` (int): Number of frames in the input video
            `num_patches` (int): Number of patches per frame in the input video
            `d_model` (int): Dimension of the tensors used to compute attention
            `positional_embedding_dropout` (float): dropout probability for the positional embeddings (default 0.0)
          
        """

        super(PositionalEmbedding, self).__init__()

        self.positional_embedding = nn.Parameter(torch.zeros(1, 1, num_patches, d_model).repeat(1, num_frames, 1, 1)) 
        self.positional_embedding_dropout = nn.Dropout(p=positional_embedding_dropout)
    
    def forward(self, x):

        """
        Adds positional embeddings to the input tensors. 
  
        Parameters:
           x (tensor): Tensor of dimension (batch_size, num_frames, num_patches, d_model)

        Returns:
            x (tensor): Tensor of dimension (batch_size, num_frames, num_patches, d_model)

        """

        x = self.positional_embedding_dropout(x + self.positional_embedding) 
        return x
        

class Attention(nn.Module):

    def __init__(self, d_model, num_heads=12, qkv_bias=False, attention_dropout=0., projection_dropout=0., init=''):

        """
        Initialises all the attributes of the for the multi-headed attention block. 
  
        Parameters:
            `d_model` (int): Dimension of the tensors used to compute attention
            `num_heads` (int): Number of attention heads.
            `qkv_bias` (boolean): Determines whether to use bias as part of the query/key/value linear layers in the attention block (default True)
            `attention_dropout` (float): Dropout probability for the layer after the multi-head attention mechanism (default 0.0)
            `projection_dropout` (float): Dropout probability for the layer after the projection layer (default 0.0)
            `init` (string): Intialisation method for the parameters in the attenion block (default )
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

    # masks not yet added
    def forward(self, x):

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
        self_attention = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        self_attention = self_attention.softmax(dim=-1)
        self_attention = self.attention_dropout(self_attention)

        weighted_attention = torch.matmul(self_attention, value) # (batch_size, num_heads, num_tokens, head_dim)
        weighted_attention = weighted_attention.transpose(1, 2).flatten(2) # (batch_size, num_tokens, d_model)
        
        x = self.projection_layer(weighted_attention) # (batch_size, num_tokens, d_model)
        x = self.projection_dropout(x)

        return x


class DotProductAttention(nn.Module):
    def __init__(self, d_model, num_heads=12, qkv_bias=False, attention_dropout=0., projection_dropout=0.):

        """
        Initialises all the attributes for the Dot Product Attention architecture. 
  
        Parameters:
            `d_model` (int): Dimension of the tensors used to compute attention
            `num_heads` (int): Number of attention heads.
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


class FactorisedEncoder(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio=4., qkv_bias=False, dropout_1=0., dropout_2=0., 
                attention_dropout=0., projection_dropout=0.):

        """
        Encoder consisting of the basic attention architecture.
  
        Parameters:
            `d_model` (int): Dimension of the tensors used to compute attention
            `num_heads` (int): Number of attention heads.
            `mlp_ratio` (int): Used to determine the hidden layer dimension of the MLP. (default 4)
            `qkv_bias` (boolean): Determines whether to use bias as part of the query/key/value linear layers in the attention block (default True)
            `dropout_1` (float): Dropout probability for the MLP block (default 0.0)
            `dropout_2` (float): Dropout probability for the MLP block (default 0.0)
            `attention_dropout` (float): Dropout probability for the layer after the multi-head attention mechanism (default 0.0)
            `projection_dropout` (float): Dropout probability for the layer after the projection layer (default 0.0)
    
        """

        super(FactorisedEncoder, self).__init__()

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

        x = x + self.attention(self.layer_norm_1(x)) # (batch_size, num_tokens, d_model)
        x = x + self.mlp(self.layer_norm_2(x)) # (batch_size, num_tokens, d_model)

        return x



class FactorisedSelfAttentionEncoder(nn.Module):
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

        super(FactorisedSelfAttentionEncoder, self).__init__()

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



class FactorisedDotProductAttentionEncoder(nn.Module):
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

        super(FactorisedDotProductAttentionEncoder, self).__init__()
        
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


class Encoder(nn.Module):
    def __init__(self, model_name, num_frames, num_patches, d_model, 
                depth, num_heads, mlp_ratio=4., qkv_bias=False, positional_embedding_dropout=0.,
                attention_dropout=0., projection_dropout=0., dropout_1=0., dropout_2=0.):
        
        """
        Encoder block for factorised attention, factorised self attention and factorised dot product attention.
  
        Parameters:
            `model_name` (string): One of 'factorised encoder', 'factorised self attention' or 'factorised dot product attention'
            `num_frames` (int): Number of frames in the input video
            `num_patches` (int): Number of patches per frame in the input video
            `d_model` (int): Dimension of the tensors used to compute attention
            `depth` (int): number of encoder blocks. 
            `num_heads` (int): Number of attention heads.
            `mlp_ratio` (int): Used to determine the hidden layer dimension of the MLP. (default 4)
            `qkv_bias` (boolean): Determines whether to use bias as part of the query/key/value linear layers in the attention block (default True)
            `positional_embedding_dropout` (float): Dropout probability for the positional embeddings (default 0.0)
            `attention_dropout` (float): Dropout probability for the layer after the multi-head attention mechanism (default 0.0)
            `projection_dropout` (float): Dropout probability for the layer after the projection layer (default 0.0)
            `dropout_1` (float): Dropout probability for the MLP block (default 0.0)
            `dropout_2` (float): Dropout probability for the MLP block (default 0.0)
    
        """

        super(Encoder, self).__init__()

        self.model_name = model_name

        if self.model_name == 'factorised encoder':

            self.spacial_token = nn.Parameter(torch.zeros(1, 1, d_model)) # [class] token
            self.temporal_token = nn.Parameter(torch.zeros(1, 1, d_model)) # [class] token

            self.add_positional_embedding_spatial = PositionalEmbedding(num_frames, num_patches + 1, d_model, positional_embedding_dropout)
            self.add_positional_embedding_temporal = PositionalEmbedding(num_frames + 1, 1, d_model, positional_embedding_dropout)
            
            self.spatialEncoder = nn.Sequential(
            *[
                FactorisedEncoder(
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

            self.temporalEncoder = nn.Sequential(
                *[
                    FactorisedEncoder(
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

        elif self.model_name == 'factorised self attention':
            self.add_positional_embedding = PositionalEmbedding(num_frames, num_patches, d_model, positional_embedding_dropout)
            self.encoder = nn.Sequential(
                * [
                FactorisedSelfAttentionEncoder(
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
            self.add_positional_embedding = PositionalEmbedding(num_frames, num_patches, d_model, positional_embedding_dropout)
            self.encoder = nn.Sequential(
                * [
                FactorisedDotProductAttentionEncoder(
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


    def forward(self, x):

        """
        Performs a forward pass over the specified attention architecture for all layers of the encoder.
  
        Parameters:
            x (tensor): Tensor of dimension (batch_size, num_frames, num_patches, d_model)
        
        Returns:
            x (tensor): if model_name is 'factorised encoder', Tensor of dimension (batch_size, num_frames + 1, d_model) 
                        if model _name is 'factorised self attention' or 'factorised dot product attention', 
                        Tensor of dimension (batch_size, num_frames, num_patches, d_model)  

        """
        
        batch_size, num_frames, num_patches, d_model = x.shape

        if self.model_name == 'factorised encoder':
            x = x.reshape(-1, num_patches, d_model) # (batch_size * num_frames, num_patches, d_model)

            cls_token_spatial = self.spacial_token.expand(batch_size * num_frames, 1, -1) # (1, 1, d_model) -> (batch_size * num_frames, 1, d_model)
            x = torch.cat((cls_token_spatial, x), dim=1) # (batch_size * num_frames, num_patches + 1, d_model)
            
            x = self.add_positional_embedding_spatial(x.reshape(batch_size, num_frames, num_patches + 1, d_model))
            x = x.reshape(-1, num_patches + 1, d_model) # (batch_size * num_frames, num_patches + 1, d_model)

            x = self.spatialEncoder(x) # (batch_size * num_frames, num_patches + 1, d_model)

            x = x.reshape(batch_size, num_frames, num_patches + 1, d_model)
            x = x[:, :, 0] # (batch_size, num_frames, d_model)

            cls_token_temporal = self.temporal_token.expand(batch_size, -1, -1) # (1, 1, d_model) -> (batch_size, 1, d_model)
            x = torch.cat((cls_token_temporal, x), dim=1) # (batch_size, num_frames + 1, d_model)

            x = self.add_positional_embedding_temporal(x.reshape(batch_size, num_frames + 1, 1, d_model))

            x = x.reshape(-1, num_frames + 1, d_model) # (batch_size, num_frames + 1, d_model)

            x = self.temporalEncoder(x) # (batch_size, num_frames + 1, d_model)

        elif self.model_name == 'factorised self attention' or self.model_name == 'factorised dot product attention': 
            x = self.add_positional_embedding(x) # (batch_size, num_frames, num_patches, d_model)
            x = self.encoder(x) # (batch_size, num_frames, num_patches, d_model)

        return x
