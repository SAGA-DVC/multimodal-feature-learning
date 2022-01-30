""" Vision Transformer (ViT) in PyTorch

Code used from the following repositories:
1. https://github.com/rwightman/pytorch-image-models
2. https://github.com/jankrepl/mildlyoverfitted 

"""

import torch
import torch.nn as nn
from functools import partial


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, d_model=768, layer_norm=None):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.project_to_patch_embeddings = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.layer_norm = layer_norm(d_model) if layer_norm else nn.Identity()

    def forward(self, x):
        # x = (batch_size, in_channels, img_size, img_size)
        x = self.project_to_patch_embeddings(x) # (batch_size, d_model, num_patches ** 0.5, num_patches ** 0.5)
        x = x.flatten(2)  # (batch_size, d_model, num_patches)
        x = x.transpose(1, 2)  # (batch_size, num_patches, d_model)
        x = self.layer_norm(x)
        
        return x


class Attention(nn.Module):
    def __init__(self, d_model, num_heads=12, qkv_bias=False, attention_dropout=0., projection_dropout=0.):
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
    def forward(self, x, mask=None):
        batch_size, num_patches, dim = x.shape
        assert self.d_model == dim, "The dimension of the patch embeddings is not equal to that of d_model."
        
        qkv = self.qkv(x) # (batch_size, num_patches, dim * 3)
        qkv = qkv.reshape(batch_size, num_patches, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, batch_size, num_heads, num_patches, head_dim)

        query, key, value = qkv.unbind(0) # (batch_size, num_heads, num_patches, head_dim)
        
        # (batch_size, num_heads, num_patches, head_dim) * (batch_size, num_heads, head_dim, num_patches) 
        # -> (batch_size, num_heads, num_patches, num_patches)
        self_attention = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        self_attention = self_attention.softmax(dim=-1)
        self_attention = self.attention_dropout(self_attention)

        weighted_attention = torch.matmul(self_attention, value) # (batch_size, num_heads, num_patches, head_dim)
        weighted_attention = weighted_attention.transpose(1, 2).flatten(2) # (batch_size, num_patches, d_model)
        
        x = self.projection_layer(weighted_attention) # (batch_size, num_patches, d_model)
        x = self.projection_dropout(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, activation_layer=nn.GELU, 
                dropout_1=0., dropout_2=0.):
        super(MLP, self).__init__()
        self.fully_connected_1 = nn.Linear(in_dim, hidden_dim)
        self.activation_layer = activation_layer()
        self.dropout_1 = nn.Dropout(dropout_1)
        self.fully_connected_2 = nn.Linear(hidden_dim, out_dim)
        self.dropout_2 = nn.Dropout(dropout_2)

    def forward(self, x):
        x = self.fully_connected_1(x) # (batch_size, num_patches, hidden_dim)
        x = self.activation_layer(x)
        x = self.dropout_1(x) 
        x = self.fully_connected_2(x)  # (batch_size, num_patches, out_dim)
        x = self.dropout_2(x) 

        return x


class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio=4., qkv_bias=False, dropout_1=0., dropout_2=0., 
                attention_dropout=0., projection_dropout=0., activation_layer=nn.GELU):
        super(EncoderBlock, self).__init__()
        #eps for compatibility with ViT pretrained weights??
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6) 

        self.attention = Attention(d_model, num_heads=num_heads, qkv_bias=qkv_bias, 
                                   attention_dropout=attention_dropout, projection_dropout=projection_dropout)

        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)

        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mlp = MLP(in_dim=d_model, hidden_dim=mlp_hidden_dim, out_dim=d_model, 
                       activation_layer=activation_layer, dropout_1=dropout_1, dropout_2=dropout_2)

    def forward(self, x):
        x = x + self.attention(self.layer_norm_1(x)) # (batch_size, num_patches, d_model)
        x = x + self.mlp(self.layer_norm_2(x)) # (batch_size, num_patches, d_model)

        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000, d_model=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True,
                 positional_embedding_dropout=0., attention_dropout=0., projection_dropout=0., 
                 mlp_dropout_1=0., mlp_dropout_2=0., layer_norm=None, activation_layer=None, weight_init=''):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.d_model = d_model
        norm_layer = layer_norm or partial(nn.LayerNorm, eps=1e-6)
        self.activation_layer = activation_layer or nn.GELU

        self.patch_embeddings_layer = PatchEmbedding(img_size=img_size, patch_size=patch_size, 
                                                    in_channels=in_channels, d_model=self.d_model, 
                                                    layer_norm=None)
        self.num_patches = self.patch_embeddings_layer.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) # [class] token
        self.positional_embedding = nn.Parameter(torch.zeros(1, 1 + self.num_patches, d_model))
        self.positional_embedding_dropout = nn.Dropout(p=positional_embedding_dropout)

        self.encoderBlocks = nn.Sequential(
            *[
                EncoderBlock(
                    d_model=self.d_model,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    dropout_1=mlp_dropout_1,
                    dropout_2=mlp_dropout_2,
                    attention_dropout=attention_dropout,
                    projection_dropout=projection_dropout,
                    activation_layer=self.activation_layer
                )
                for _ in range(depth)
            ]
        )

        self.layer_norm = norm_layer(d_model)
        self.head = nn.Linear(self.d_model, self.num_classes) if self.num_classes > 0 else nn.Identity()


    def forward(self, x):
        # (batch_size, in_channels, img_size, img_size)
        x = self.patch_embeddings_layer(x) # (batch_size, num_patches, d_model)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1) # (1, 1, d_model) -> (batch_size, 1, d_model)
        x = torch.cat((cls_token, x), dim=1) # (batch_size, num_patches + 1, d_model)
        x = self.positional_embedding_dropout(x + self.positional_embedding)
        x = self.encoderBlocks(x) # (batch_size, num_patches + 1, d_model)
        x = self.layer_norm(x)

        cls_token_final = x[:, 0] # (batch_size, 1, num_classes)
        x = self.head(cls_token_final) # (batch_size, num_classes)

        return x


