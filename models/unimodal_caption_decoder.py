""" 
Caption Decoder
"""


import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_, zeros_, ones_

from .modules.embedding_layers import PositionalEncoding, PositionEmbeddingCaptionSine, VocabularyEmbedder
from .modules.layers import UnimodalCaptionDecoderLayer
from .modules.misc_modules import NestedTensor

from .load_weights import init_encoder_block_weights, load_token_embeddings, load_positional_embeddings, load_cls_tokens, load_vivit_encoder_weights, load_classification_weights


# TODO - add pos ebmbed for video features used in cross attention
# TODO - context features (for vid feats and captions(captions influence each other))
class UnimodalCaptionDecoder(nn.Module):
    def __init__(self, vocab_size, seq_len=20, d_model=768, embedding_matrix=None, emb_weights_req_grad=False, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, 
                positional_embedding_dropout=0., attention_dropout=0., projection_dropout=0., 
                bridge_dropout=0., mlp_dropout_1=0., mlp_dropout_2=0., pre_norm=True,
                weight_init=False, weight_load=False, model_official=None, return_intermediate=False):
        
        """
        Unimodal Caption Decoder
        """

        super(UnimodalCaptionDecoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.target_embedding = VocabularyEmbedder(vocab_size, d_model)
        # self.word_positional_embedding_layer = PositionEmbeddingCaptionSine(d_model, normalize=True)
        self.positional_encoding = PositionalEncoding(d_model, dropout=positional_embedding_dropout)
        
        self.d_model = d_model
        self.depth = depth
        self.return_intermediate = return_intermediate

        self.decoder = nn.ModuleList(
                [
                    UnimodalCaptionDecoderLayer(
                        d_model=d_model,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        attention_dropout=attention_dropout,
                        projection_dropout=projection_dropout,
                        bridge_dropout=bridge_dropout,
                        mlp_dropout_1=mlp_dropout_1,
                        mlp_dropout_2=mlp_dropout_2,
                        pre_norm=pre_norm
                    )
                    for _ in range(depth)
                ]
            )
        
        # self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
        self.head = nn.Linear(d_model, vocab_size)

        self.init_weights(embedding_matrix, emb_weights_req_grad)
        


    # TODO - check if pos embed should be given at every decoder layer to word and video
    # TODO - use log softmax?
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_padding_mask=None, memory_padding_mask=None):

        """
        Performs a forward pass on the Unimodal Caption Decoder
  
        Parameters:
            captions (Tensor): Tensor of dimension (batch_size, seq_len)
            memory (Tensor): Tensor of dimension (batch_size, num_tokens, d_model)
            **memory_positional_embedding_layer (nn.Module): position embedding layer for encoder inputs
            tgt_mask (Tensor): Tensor of dimension (batch_size, 1, seq_len, seq_len). Combination of the lookahead mask and padding mask for the target/captions
            padding_mask (Tensor): Tensor of dimension (batch_size, seq_len). Used for position embeddings
            memory_mask (Tensor): Tensor of dimension (batch_size, 1, 1, num_tokens). Memory padding mask to be used in the cross attention block of the decoder.
            durations (Tensor): Tensor of dimension (batch_size) representing the duration of each video in the batch in seconds
        Returns:
            x (tensor): Tensor of dimension (1, batch_size, seq_len, vocab_size) OR (depth, batch_size, seq_len, vocab_size)
        """

        tgt = self.target_embedding(tgt)    # (batch_size, seq_len, embed_dim)
        tgt = self.positional_encoding(tgt)

        # tgt = tgt + self.word_positional_embedding_layer(NestedTensor(tgt, tgt_padding_mask)).transpose(1, 2)
        # memory = memory + memory_positional_embedding_layer(NestedTensor(memory, torch.squeeze(memory_mask), durations)).transpose(1,2)

        intermediate = []
        
        for layer in self.decoder:
            tgt = layer(tgt, memory, tgt_mask, memory_mask, tgt_padding_mask, memory_padding_mask)    # (batch_size, seq_len, embed_dim)

            if self.return_intermediate:
                intermediate.append(tgt)

        if self.return_intermediate:
            tgt = torch.stack(intermediate)    # (depth, batch_size, seq_len, embed_dim)
        else:
            tgt = tgt.unsqueeze(0)    # (1, batch_size, seq_len, embed_dim)
        
        # (1, batch_size, seq_len, vocab_size) OR (depth, batch_size, seq_len, vocab_size)
        tgt = self.head(tgt).softmax(dim=-1)

        return tgt
    

    def init_weights(self, embedding_matrix, emb_weights_req_grad):

        """
        Initialises the weights and biases of the Caption Decoder model.
        These parameters include positional embeddings.
        """

        self.target_embedding.init_word_embeddings(embedding_matrix, emb_weights_req_grad)
        # trunc_normal_(self.word_positional_embedding_layer.positional_embedding, std=.02)
        self.decoder.apply(init_encoder_block_weights)
    


def build_unimodal_caption_decoder(args, vocab_size, seq_len, embedding_matrix):
    # return UnimodalCaptionDecoder(vocab_size=vocab_size, seq_len=seq_len, embedding_matrix=embedding_matrix, **args)
    return UnimodalCaptionDecoder(vocab_size=vocab_size, 
                        seq_len=seq_len, 
                        d_model=args.d_model, 
                        embedding_matrix=embedding_matrix, 
                        emb_weights_req_grad=args.emb_weights_req_grad, 
                        depth=args.depth, 
                        num_heads=args.num_heads, 
                        mlp_ratio=args.mlp_ratio, 
                        qkv_bias=args.qkv_bias, 
                        positional_embedding_dropout=args.positional_embedding_dropout,
                        attention_dropout=args.attention_dropout, 
                        projection_dropout=args.projection_dropout, 
                        bridge_dropout=args.bridge_dropout,
                        mlp_dropout_1=args.mlp_dropout_1, 
                        mlp_dropout_2=args.mlp_dropout_2, 
                        pre_norm=args.pre_norm,
                        weight_init=args.weight_init, 
                        weight_load=args.weight_load, 
                        model_official=args.model_official, 
                        return_intermediate=args.return_intermediate)