""" 
Caption Decoder
"""


import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_, zeros_, ones_

from .modules import PositionalEmbedding, VocabularyEmbedder, CaptionDecoderLayer
from .load_weights import init_encoder_block_weights, load_token_embeddings, load_positional_embeddings, load_cls_tokens, load_vivit_encoder_weights, load_classification_weights

# TODO - add sin/cos pos embed
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
        self.word_positional_embedding_layer = PositionalEmbedding((1, seq_len-1, d_model), positional_embedding_dropout) 
        
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
    # TODO - check if pos embed should be given at every decoder layer to word
    # TODO - use log softmax?
    def forward(self, captions, memory, positional_embedding_layer, tgt_mask, memory_mask):

        """
        Performs a forward pass on the Transformer model
  
        Parameters:
            captions (Tensor): Tensor of dimension (batch_size, seq_len)
            memory (Tensor): Tensor of dimension (batch_size, num_tokens, d_model)
            positional_embedding_layer (nn.Module): position embedding layer for encoder inputs
            tgt_mask (Tensor): Tensor of dimension (batch_size, 1, seq_len, seq_len). Combination of the lookahead mask and padding mask for the target/captions
            memory_masl (Tensor): Tensor of dimension (batch_size, 1, 1, num_tokens). Memory padding mask to be used in the cross attention block of the decoder.
        
        Returns:
            x (tensor): Tensor of dimension (1, batch_size, seq_len, vocab_size) OR (depth, batch_size, seq_len, vocab_size)
        """

        target = self.target_embedding(captions)    # (batch_size, seq_len, embed_dim)

        intermediate = []
        
        for layer in self.decoder:
            target = layer(target, memory, self.word_positional_embedding_layer, positional_embedding_layer, tgt_mask, memory_mask)    # (batch_size, seq_len, embed_dim)

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
        trunc_normal_(self.word_positional_embedding_layer.positional_embedding, std=.02)
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
    

    