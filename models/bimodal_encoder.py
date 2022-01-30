""" Bimodal encoder for fusion of audio and video features """

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_, zeros_, ones_

from modules import BiModalEncoderLayer
from load_weights import init_encoder_block_weights, load_bimodal_encoder_weights, load_classification_weights


class BiModalEncoder(nn.Module):
    def __init__(self, d_model=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, pre_norm=True,
                attention_dropout=0., projection_dropout=0., dropout_1=0., dropout_2=0., 
                classification_head=False, num_classes=None, return_preclassifier=False, return_prelogits=False, 
                weight_init=False, weight_load=False, model_official=None):
        
        """
        The Bi-modal Encoder which consists of cross attention modules between video and audio features.
  
        Parameters:
            `d_model` (int): Dimension of the tensors used to compute attention
            `depth` (int): number of encoder blocks. 
            `num_heads` (int): Number of attention heads.
            `mlp_ratio` (int): Used to determine the hidden layer dimension of the MLP. (default 4)
            `qkv_bias` (boolean): Determines whether to use bias as part of the query/key/value linear layers in the attention block (default True)
            `positional_embedding_dropout` (float): (Currently not being used) dropout probability for the positional embeddings (default 0.0)
            `attention_dropout` (float): Dropout probability for the layer after the multi-head attention mechanism (default 0.0)
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

        super(BiModalEncoder, self).__init__()

        self.d_model = d_model
        self.num_classes = num_classes
        self.depth = depth

        self.classification_head = classification_head
        self.return_prelogits = return_prelogits
        self.return_preclassifier = return_preclassifier
        
        self.bi_modal_encoder = nn.ModuleList(
                [
                    BiModalEncoderLayer(d_model=d_model,
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

        if classification_head:
            self.layer_norm = nn.LayerNorm(d_model, eps=1e-6) 
            self.head = nn.Linear(d_model, num_classes)

        if weight_load and model_official is not None:
            self.load_weights(model_official)

        elif weight_init:
            self.init_weights()


    def forward(self, vid, aud):
        
        """
        Performs a forward pass on the Bi-modal encoder.
  
        Parameters:
            vid (tensor): Tensor of dimension (batch_size, num_frames, d_model) representing video features
            aud (tensor): Tensor of dimension (batch_size, num_tokens, d_model) representing audio features
        
        Returns:
            x (tensor): if return_preclassifier is True, 2 Tensors of dimension 
                            (batch_size, num_frames, d_model) for video features AND
                            (batch_size, num_tokens, d_model) for audio features 
                            
                        if return_prelogits is True, Tensor of dimension (batch_size, 1, d_model) representing a
                            fusion of video and audio features 

                        else Tensor of dimension (batch_size, num_classes)
        """

        for layer in self.bi_modal_encoder:
            vid, aud = layer(vid, aud) # (batch_size, num_frames, d_model), (batch_size, num_tokens, d_model)

        if self.return_preclassifier :
            return vid, aud

        # TODO-some processing/combination of video and audio features
        x = vid 
        
        if self.classification_head:
            x = self.layer_norm(x)
            x = self.head(x) # (batch_size, num_classes)
            return x
        
        else:
            return x # (batch_size, 1, d_model)


    def init_weights(self):

        """
        Initialises the weights and biases of the all four ViViT models namely 
        spatio temporal attention, factorised encoder, factorised self attention, factorised dot product attention.

        These parameters include token embeddings, positional embeddings, [class] tokens (if any) and 
        encoder blocks (consisting of linear layers and normalisation layers).
        """

        self.bi_modal_encoder.apply(init_encoder_block_weights)        

        if self.classification_head:
            ones_(self.layer_norm.weight)
            zeros_(self.layer_norm.bias)

            trunc_normal_(self.head.weight, std=.02)
            trunc_normal_(self.head.bias, std=.02)



    def load_weights(self, model_official):

        """
        Loads the weights and biases from the pre-trained model to the current model.

        Parameters:
            `model_custom`: The current ViViT model
            `model_official`: The model which would be used to load the pre-trained weights
        """

        load_bimodal_encoder_weights(self, model_official)

        if self.classification_head:
            load_classification_weights(self, model_official)