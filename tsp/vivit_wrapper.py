""" 
A wrapper over ViViT for TSP
"""


import torch.nn as nn
from torch.nn.init import trunc_normal_

from models.vivit import VideoVisionTransformer
from models.modules import PositionalEmbedding
from models.load_weights import load_positional_embeddings

# TODO - Fix positional embeddings for model 3 and 4 of ViViT (DO NOT USE)
class VivitWrapper(nn.Module):
    def __init__(self, model_name, num_frames_in, img_size=224, spatial_patch_size=16, temporal_patch_size=1,
                tokenization_method='central frame', in_channels=3, d_model=768, depth=12, temporal_depth=4,num_heads=12, 
                mlp_ratio=4., qkv_bias=True, positional_embedding_dropout=0., attention_dropout=0., 
                projection_dropout=0., dropout_1=0., dropout_2=0., pre_norm=True, classification_head=False, num_classes=None,
                return_preclassifier=True, return_prelogits=False, weight_init=False, weight_load=False, model_official=None):
        
        """
        Wrapper class for ViViT for TSP
        """

        super(VivitWrapper, self).__init__()
        
        num_frames = num_frames_in // temporal_patch_size
        num_patches = (img_size // spatial_patch_size) ** 2
        
        self.positional_embedding_layer = None
        self.spatial_positional_embedding_layer = None

        # Attention
        if model_name == 'spatio temporal attention':
            self.positional_embedding_layer = PositionalEmbedding((1, num_frames * num_patches + 1, d_model), positional_embedding_dropout) 
            
        elif model_name == 'factorised encoder':
            self.spatial_positional_embedding_layer = PositionalEmbedding((1, num_patches + 1, d_model), positional_embedding_dropout)
            self.positional_embedding_layer = PositionalEmbedding((1, num_frames + 1, d_model), positional_embedding_dropout)

        else:
            self.positional_embedding_layer = PositionalEmbedding((1, num_frames, num_patches, d_model), positional_embedding_dropout)

        '''Feature dimensions'''
        self.d_model = d_model

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
                        pre_norm=pre_norm,
                        classification_head=classification_head, 
                        num_classes=num_classes,
                        return_preclassifier=return_preclassifier, 
                        return_prelogits=return_prelogits, 
                        weight_init=weight_init, 
                        weight_load=weight_load, 
                        model_official=model_official
                    )
        
        if weight_load and model_official is not None:
            self.load_weights(model_official)

        elif weight_init:
            self.init_weights()
        

    def forward(self, x):

        """
        Performs a forward pass on the ViViT model
  
        Parameters:
            x (tensor): Tensor of dimension (batch_size, in_channels, num_frames, img_size, img_size)
        
        Returns:
            x (tensor): Tensor of dimension (batch_size, d_model)
        """

        # (batch_size, num_frames * num_patches + 1, d_model) OR
        # (batch_size, num_frames + 1, d_model) OR 
        # (batch_size, num_frames, num_patches, d_model) 
        x = self.vivit(x, self.positional_embedding_layer, self.spatial_positional_embedding_layer)

        # TODO check grad later
        if self.vivit.model_name == 'factorised self attention' or self.vivit.model_name == 'factorised dot product attention':
            x = x.reshape(x.shape[0], -1, x.shape[-1])

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