""" Encoder """

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_, zeros_, ones_

from ..modules.layers import EncoderLayer
from ..load_weights import init_encoder_block_weights

class Encoder(nn.Module):
    
    def __init__(self, d_model, depth, num_heads, mlp_ratio=4., qkv_bias=False, 
                attention_dropout=0., projection_dropout=0., mlp_dropout_1=0., mlp_dropout_2=0., pre_norm=True,
                weight_init=False, weight_load=False, model_official=None, return_intermediate=False):

        """
        Encoder is Stack of N encoder layers
  
        Parameters:
            `d_model` (int): Dimension of the tensors used to compute attention
            `depth` (int): number of encoder blocks. 
            `num_heads` (int): Number of attention heads.
            `mlp_ratio` (int): Used to determine the hidden layer dimension of the MLP. (default 4)
            `qkv_bias` (boolean): Determines whether to use bias as part of the query/key/value linear layers in the attention block (default True)
            `attention_dropout` (float): Dropout probability for the layer after the multi-head attention mechanism (default 0.0)
            `projection_dropout` (float): Dropout probability for the layer after the projection layer (default 0.0)
            `dropout_1` (float): dropout probability for the MLP block (default 0.0)
            `dropout_2` (float): dropout probability for the MLP block (default 0.0)
            `pre_norm` (boolean): If True, the normalisation layer would be placed before the attention and mlp blocks. Else, after them. (default True)
            `weight_init` (boolean): If True, initialises the weights of the model (default True)
            `weight_load` (boolean): If True, loads the weights of the specified pre-trained model after initialisation (default False)
            `model_official`: This model's weights are used by ViViT
            `return_intermediate` (boolean) : If True, output from intermediate layers of the encoder are also returned along with the output from the final layer. (default False)
    
        """

        super(Encoder, self).__init__()

        self.d_model = d_model
        self.depth = depth

        self.return_intermediate = return_intermediate

        self.encoder = nn.ModuleList(
                [
                    EncoderLayer(
                        d_model=d_model,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        attention_dropout=attention_dropout,
                        projection_dropout=projection_dropout,
                        mlp_dropout_1=mlp_dropout_1,
                        mlp_dropout_2=mlp_dropout_2,
                        pre_norm=pre_norm
                    )
                    for _ in range(depth)
                ]
            )
        
        # self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.init_weights()

    
    def forward(self, src, src_positional_embedding, src_mask=None, src_padding_mask=None):

        """
        Pass the inputs (and mask) through the encoder layer in turn.
        Parameters:
            tgt (tensor): the sequence to the encoder layer, Tensor of dimension (batch_size, num_queries, d_model)
            memory (tensor): the sequence from the last layer of the encoder
            position_embedding_layer: position embedding layer for encoder inputs
            query_embedding (tensor): event queries
        
        Returns:
            output (tensor): if return_intermediate is True:
                                Tensor of dimension (1, batch_size, num_queries, d_model)
                             else:
                                 Tensor of dimension (depth, batch_size, num_queries, d_model)
        """

        intermediate = []
        
        for layer in self.encoder:
            src = layer(src, src_positional_embedding, src_mask, src_padding_mask)

            if self.return_intermediate:
                intermediate.append(src)
        
        if self.return_intermediate:
            src = torch.stack(intermediate)    # (depth, batch_size, seq_len, embed_dim)
        else:
            src = src.unsqueeze(0)    # (1, batch_size, seq_len, embed_dim)

        return src


    def init_weights(self):

        """
        Initialises the weights and biases of all the Encoder layers.
        """

        self.encoder.apply(init_encoder_block_weights)


def build_encoder(args):
    # return Encoder(**args)
    return Encoder(d_model=args.d_model, 
                depth=args.depth, 
                num_heads=args.num_heads, 
                mlp_ratio=args.mlp_ratio, 
                qkv_bias=args.qkv_bias,  
                attention_dropout=args.attention_dropout, 
                projection_dropout=args.projection_dropout, 
                mlp_dropout_1=args.mlp_dropout_1, 
                mlp_dropout_2=args.mlp_dropout_2, 
                pre_norm=args.pre_norm,
                weight_init=args.weight_init, 
                weight_load=args.weight_load, 
                model_official=args.model_official,
                return_intermediate=args.return_intermediate
            )