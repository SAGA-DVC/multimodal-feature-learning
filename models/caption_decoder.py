""" 
Caption Decoder
"""


import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_, zeros_, ones_

from .modules.embedding_layers import PositionEmbeddingCaptionSine, VocabularyEmbedder
from .modules.layers import CaptionDecoderLayer
from .modules.misc_modules import NestedTensor

from .load_weights import init_encoder_block_weights, load_token_embeddings, load_positional_embeddings, load_cls_tokens, load_vivit_encoder_weights, load_classification_weights


# TODO - add pos ebmbed for video features used in cross attention
# TODO - context features (for vid feats and captions(captions influence each other))
class CaptionDecoder(nn.Module):
    def __init__(self, vocab_size, seq_len=20, d_model=768, embedding_matrix=None, emb_weights_req_grad=False, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, 
                positional_embedding_dropout=0., attention_dropout=0., 
                projection_dropout=0., dropout_1=0., dropout_2=0., pre_norm=True,
                weight_init=False, weight_load=False, model_official=None, return_intermediate=False):
        
        """
        Caption Decoder
        """

        super(CaptionDecoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.target_embedding = VocabularyEmbedder(vocab_size, d_model)
        self.word_positional_embedding_layer = PositionEmbeddingCaptionSine(d_model, normalize=True)
        
        self.d_model = d_model
        self.depth = depth
        self.return_intermediate = return_intermediate

        self.decoder = nn.ModuleList(
                [
                    CaptionDecoderLayer(
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
        
        self.head = nn.Linear(d_model, vocab_size)
        
        if weight_load and model_official is not None:
            self.load_weights(model_official)

        elif weight_init:
            self.init_weights(embedding_matrix, emb_weights_req_grad)
        
    # TODO - add <start> and <end> token
    # TODO - change ordering of pos embed and query embed parameters 
    # TODO - check if pos embed should be given at every decoder layer to word and video
    # TODO - use log softmax?
    def forward(self, captions, memory, tgt_mask, padding_mask, memory_mask):

        """
        Performs a forward pass on the Transformer model
  
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

        target = self.target_embedding(captions)    # (batch_size, seq_len, embed_dim)

        target = target + self.word_positional_embedding_layer(NestedTensor(target, padding_mask)).transpose(1, 2)
        # memory = memory + memory_positional_embedding_layer(NestedTensor(memory, memory_mask.squeeze(1).squeeze(1), durations)).transpose(1,2)

        intermediate = []
        
        for layer in self.decoder:
            target = layer(target, memory, tgt_mask, memory_mask)    # (batch_size, seq_len, embed_dim)

            if self.return_intermediate:
                intermediate.append(self.layer_norm(target))

        if self.return_intermediate:
            target = torch.stack(intermediate)    # (depth, batch_size, seq_len, embed_dim)
        else:
            target = target.unsqueeze(0)    # (1, batch_size, seq_len, embed_dim)
        
        # (1, batch_size, seq_len, vocab_size) OR (depth, batch_size, seq_len, vocab_size)
        target = self.head(target).softmax(dim=-1)

        return target
    

    def init_weights(self, embedding_matrix, emb_weights_req_grad):

        """
        Initialises the weights and biases of the Caption Decoder model.
        These parameters include positional embeddings.
        """

        self.target_embedding.init_word_embeddings(embedding_matrix, emb_weights_req_grad)
        # trunc_normal_(self.word_positional_embedding_layer.positional_embedding, std=.02)
        self.decoder.apply(init_encoder_block_weights)
            

    def load_weights(self, model_official):

        """
        Loads the weights and biases from the pre-trained model to the current model
        These weights include positional embeddings.

        Parameters:
            `model_custom`: The current ViViT model
            `model_official`: The model which would be used to load the pre-trained weights
        """

        load_positional_embeddings(self, model_official)
    


def build_caption_decoder(args, vocab_size, seq_len, embedding_matrix):
    # return CaptionDecoder(vocab_size=vocab_size, seq_len=seq_len, embedding_matrix=embedding_matrix, **args)
    return CaptionDecoder(vocab_size=vocab_size, 
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
                        dropout_1=args.dropout_1, 
                        dropout_2=args.dropout_2, 
                        pre_norm=args.pre_norm,
                        weight_init=args.weight_init, 
                        weight_load=args.weight_load, 
                        model_official=args.model_official, 
                        return_intermediate=args.return_intermediate)