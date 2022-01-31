""" Video Vision Transformer (ViViT) models in PyTorch

Code used from the following repositories:
1. https://github.com/google-research/scenic
2. https://github.com/rwightman/pytorch-image-models
3. https://github.com/jankrepl/mildlyoverfitted 

"""


import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_, zeros_, ones_

from vivit import VideoVisionTransformer
from decoder import Decoder
from modules import TokenEmbedding, PositionalEmbedding, VivitEncoder
from load_weights import init_encoder_block_weights, load_token_embeddings, load_positional_embeddings, load_cls_tokens, load_vivit_encoder_weights, load_classification_weights


class Transformer(nn.Module):
    def __init__(self, model_name, num_frames, num_patches, img_size=224, spatial_patch_size=16, temporal_patch_size=1,
                tokenization_method='central frame', in_channels=3, d_model=768, depth=12, temporal_depth=4,num_heads=12, 
                mlp_ratio=4., qkv_bias=True, positional_embedding_dropout=0., attention_dropout=0., 
                projection_dropout=0., dropout_1=0., dropout_2=0., pre_norm=True, classification_head=False, num_classes=None,
                return_preclassifier=True, return_prelogits=False, weight_init=False, weight_load=False, model_official=None):
        
        """
        The Video Vision Transformer (ViViT) which consists of 3 attention architectures, namely, 
        'factorised encoder', 'factorised self attention' and 'factorised dot product attention'.
  
        Parameters:
            `model_name` (string): One of 'spatio temporal attention', 'factorised encoder', 'factorised self attention' or 'factorised dot product attention'
            `num_frames` (int): Number of frames in the input video
            `num_patches` (int): Number of patches per frame in the input video
            `img_size` (int): dimension of one frame of the video (should be a square i.e. height=width) (default 224)
            `spatial_patch_size` (int): dimension of the patch that will be used to convolve over a frame (default 16)
            `temporal_patch_size` (int): dimension of the patch that will be used to convolve over multiple frames (default 1)
            `tokenization_method` (string): One of 'filter inflation' or 'central frame'. Used for loading pre-trained weights for tubelet embedding (default 'central frame')
            `in_channels` (int): number of channels of the each frame in the video. e.g. RGB. (default 3)
            `d_model` (int): Dimension of the tensors used to compute attention
            `depth` (int): number of encoder blocks. 
            `temporal_depth` (int): number of temporal encoder blocks (for factorised encoder model only)
            `num_heads` (int): Number of attention heads.
            `mlp_ratio` (int): Used to determine the hidden layer dimension of the MLP. (default 4)
            `qkv_bias` (boolean): Determines whether to use bias as part of the query/key/value linear layers in the attention block (default True)
            `positional_embedding_dropout` (float): dropout probability for the positional embeddings (default 0.0)
            `attention_dropout` (float): Dropout probability for the layer after the multi-head attention mechanism (default 0.0)
            `projection_dropout` (float): Dropout probability for the layer after the projection layer (default 0.0)
            `dropout_1` (float): dropout probability for the MLP block (default 0.0)
            `dropout_2` (float): dropout probability for the MLP block (default 0.0)
            `pre_norm` (boolean): If True, the normalisation layer would be placed before the attention and mlp blocks. Else, after them. (default True)
            `classification_head` (boolean): If True, a classification head (fully connected layer) is added on top of the model (default False)
            `num_classes` (int): number of classes for the prediction task (default None)
            `return_preclassifier` (boolean): If True, return the representation after the transformer encoder. Useful if using this as the backbone stem as part of a bigger architecture (default False)
            `return_prelogits` (boolean): (Currently not being used) If True, return the final representation of the network before the classification head. Useful when using features for a downstream task (default False) 
            `weight_init` (boolean): If True, initialises the weights of the model (default True)
            `weight_load` (boolean): If True, loads the weights of the specified pre-trained model after initialisation (default False)
            `model_official`: This model's weights are used by ViViT
        """

        super(Transformer, self).__init__()
        
        if model_name not in ['spatio temporal attention', 'factorised encoder', 'factorised self attention', 'factorised dot product attention'] :
            raise ValueError(f'Unrecognized model: {model_name}. Choose between "spatio temporal attention",\
                            "factorised encoder", "factorised self attention" or "factorised dot product attention"')


        self.positional_embedding_layer = None
        self.spatial_positional_embedding_layer = None

        if model_name == 'spatio temporal attention':
            self.positional_embedding_layer = PositionalEmbedding((1, num_frames * num_patches, d_model), positional_embedding_dropout) 
            
        elif model_name == 'factorised encoder':
            self.spatial_positional_embedding_layer = PositionalEmbedding((1, num_patches + 1, d_model), positional_embedding_dropout)
            self.positional_embedding_layer = PositionalEmbedding((1, num_frames, d_model), positional_embedding_dropout)

        else:
            self.positional_embedding_layer = PositionalEmbedding((1, num_frames, num_patches, d_model), positional_embedding_dropout)

        
        self.vivit = VideoVisionTransformer(model_name=model_name, 
                        num_frames=num_frames, 
                        num_patches=num_patches, 
                        img_size=img_size, 
                        spatial_patch_size=spatial_patch_size, 
                        temporal_patch_size=temporal_patch_size,
                        tokenization_method=tokenization_method, 
                        in_channels=in_channels, 
                        d_model=d_model, 
                        depth=depth, 
                        temporal_depth=temporal_depth,
                        num_heads=num_heads, 
                        mlp_ratio=mlp_ratio, 
                        qkv_bias=qkv_bias, 
                        attention_dropout=attention_dropout, 
                        projection_dropout=projection_dropout, 
                        dropout_1=dropout_1, 
                        dropout_2=dropout_2, 
                        classification_head=classification_head, 
                        num_classes=num_classes,
                        return_preclassifier=return_preclassifier, 
                        return_prelogits=return_prelogits, 
                        weight_init=weight_init, 
                        weight_load=weight_load, 
                        model_official=model_official
                    )
        
        self.decoder = Decoder(d_model=d_model, 
                        depth=depth, 
                        num_heads=num_heads, 
                        mlp_ratio=mlp_ratio, 
                        qkv_bias=qkv_bias,  
                        attention_dropout=attention_dropout, 
                        projection_dropout=projection_dropout, 
                        dropout_1=dropout_1, 
                        dropout_2=dropout_2, 
                        pre_norm=pre_norm,
                        weight_init=weight_init, 
                        weight_load=weight_load, 
                        model_official=model_official
                    )
        
        if weight_load and model_official is not None:
            self.load_weights(model_official)

        elif weight_init:
            self.init_weights()
        

    def forward(self, x, target, query_embedding):

        """
        Performs a forward pass on the Transformer model
  
        Parameters:
            x (tensor): Tensor of dimension (batch_size, in_channels, num_frames, img_size, img_size)
            target (tensor): the sequence to the decoder layer, Tensor of dimension (batch_size, num_tokens, d_model)
            query_embedding (tensor): event queries, Tensor of dimension (batch_size, num_tokens, d_model)
        
        Returns:
            x (tensor): 
        """

        # (batch_size, num_frames * num_patches, d_model) OR
        # (batch_size, num_frames, d_model) OR 
        # (batch_size, num_frames, num_patches, d_model) 
        x = self.vivit(x, self.positional_embedding_layer, self.spatial_positional_embedding_layer)

        # check grad later
        if self.vivit.model_name == 'factorised self attention' or self.vivit.model_name == 'factorised dot product attention':
            x = x.reshape(x.shape[0], -1, x.shape[-1])
        
        # (batch_size, num_tokens, d_model)
        x = self.decoder(target=target, memory=x, 
                        positional_embedding_layer=self.positional_embedding_layer, query_embedding=query_embedding)

        return x
    

    def init_weights(self):

        """
        Initialises the weights and biases of the Transformer model.
        These parameters include positional embeddings.
        """

        trunc_normal_(self.positional_embedding_layer.positional_embedding, std=.02)
        if self.vivit.model_name == 'factorised encoder':
            trunc_normal_(self.spatial_positional_embedding_layer.positional_embedding, std=.02)
            

    def load_weights(self, model_official):

        """
        Loads the weights and biases from the pre-trained model to the current model
        These weights include positional embeddings.

        Parameters:
            `model_custom`: The current ViViT model
            `model_official`: The model which would be used to load the pre-trained weights
        """

        load_positional_embeddings(self, model_official)