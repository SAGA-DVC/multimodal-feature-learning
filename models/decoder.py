""" Bimodal encoder for fusion of audio and video features """

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_, zeros_, ones_

from modules import DecoderLayer

class Decoder(nn.Module):
    
    def __init__(self, d_model, depth, num_heads, mlp_ratio=4., qkv_bias=False, 
                attention_dropout=0., projection_dropout=0., dropout_1=0., dropout_2=0., pre_norm=True,
                weight_init=False, weight_load=False, model_official=None):

        """
        Decoder is Stack of N decoder layers
  
        Parameters:
            `d_model` (int): Dimension of the tensors used to compute attention
            `depth` (int): number of encoder blocks. 
            `num_heads` (int): Number of attention heads.
            `mlp_ratio` (int): Used to determine the hidden layer dimension of the MLP. (default 4)
            `qkv_bias` (boolean): Determines whether to use bias as part of the query/key/value linear layers in the attention block (default True)
            `positional_embedding_dropout` (float): dropout probability for the positional embeddings (default 0.0)
            `attention_dropout` (float): Dropout probability for the layer after the multi-head attention mechanism (default 0.0)
            `dropout_1` (float): dropout probability for the MLP block (default 0.0)
            `dropout_2` (float): dropout probability for the MLP block (default 0.0)
            `pre_norm` (boolean): If True, the normalisation layer would be placed before the attention and mlp blocks. Else, after them. (default True)
            `weight_init` (boolean): If True, initialises the weights of the model (default True)
            `weight_load` (boolean): If True, loads the weights of the specified pre-trained model after initialisation (default False)
            `model_official`: This model's weights are used by ViViT
    
        """

        super(Decoder, self).__init__()

        self.d_model = d_model
        self.depth = depth

        self.decoder = nn.ModuleList(
                [
                    DecoderLayer(
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

        # if weight_load and model_official is not None:
        #     self.load_weights(model_official)

        # else:
        #     self.init_weights()

    
    def forward(self, target, memory, position_embedding_layer, query_embedding):

        """
        Pass the inputs (and mask) through the decoder layer in turn.
        Parameters:
            target (tensor): the sequence to the decoder layer, Tensor of dimension (batch_size, num_tokens, d_model)
            memory (tensor): the sequence from the last layer of the encoder
            position_embedding_layer: position embedding layer for encoder inputs
            query_embedding (tensor): event queries
        
        Returns:
            output (tensor): Tensor of dimension (batch_size, num_tokens, d_model)
        """

        output = target

        for layer in self.decoder:
            output = layer(output, memory, position_embedding_layer, query_embedding)

        # if self.norm is not None:
        #     output = self.norm(output)

        # return output.unsqueeze(0)

        return output



    