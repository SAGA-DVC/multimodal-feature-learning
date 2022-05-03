""" Decoder """

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_, zeros_, ones_

from ..modules.layers import DecoderLayer
from ..load_weights import init_encoder_block_weights

class Decoder(nn.Module):
    
    def __init__(self, d_model, depth, num_heads, mlp_ratio=4., qkv_bias=False, 
                attention_dropout=0., projection_dropout=0., dropout_1=0., dropout_2=0., pre_norm=True,
                weight_init=False, weight_load=False, model_official=None, return_intermediate=False):

        """
        Decoder is Stack of N decoder layers
  
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
            `return_intermediate` (boolean) : If True, output from intermediate layers of the decoder are also returned along with the output from the final layer. (default False)
    
        """

        super(Decoder, self).__init__()

        self.d_model = d_model
        self.depth = depth

        self.return_intermediate = return_intermediate

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
        
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        # if weight_load and model_official is not None:
        #     self.load_weights(model_official)

        # else:
        #     self.init_weights()

        self.init_weights()

    
    def forward(self, target, memory, positional_embedding_layer, query_embedding, mask=None):

        """
        Pass the inputs (and mask) through the decoder layer in turn.
        Parameters:
            target (tensor): the sequence to the decoder layer, Tensor of dimension (batch_size, num_queries, d_model)
            memory (tensor): the sequence from the last layer of the encoder
            position_embedding_layer: position embedding layer for encoder inputs
            query_embedding (tensor): event queries
        
        Returns:
            output (tensor): if return_intermediate is True:
                                Tensor of dimension (1, batch_size, num_queries, d_model)
                             else:
                                 Tensor of dimension (depth, batch_size, num_queries, d_model)
        """

        output = target

        intermediate = []
        
        for layer in self.decoder:
            output = layer(output, memory, positional_embedding_layer, query_embedding, mask)

            if self.return_intermediate:
                intermediate.append(self.layer_norm(output))

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


    def init_weights(self):

        """
        Initialises the weights and biases of all the Decoder layers.
        """

        self.decoder.apply(init_encoder_block_weights)


def build_decoder(args):
    # return Decoder(**args)
    return Decoder(d_model=args.d_model, 
                depth=args.depth, 
                num_heads=args.num_heads, 
                mlp_ratio=args.mlp_ratio, 
                qkv_bias=args.qkv_bias,  
                attention_dropout=args.attention_dropout, 
                projection_dropout=args.projection_dropout, 
                dropout_1=args.dropout_1, 
                dropout_2=args.dropout_2, 
                pre_norm=args.pre_norm,
                weight_init=args.weight_init, 
                weight_load=args.weight_load, 
                model_official=args.model_official,
                return_intermediate=args.return_intermediate
            )