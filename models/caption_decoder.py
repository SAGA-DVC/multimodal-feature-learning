""" 
Caption Decoder
"""


import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_, zeros_, ones_

from .decoder import Decoder
from .modules import PositionalEmbedding, VocabularyEmbedder
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
        self.target_embedding = VocabularyEmbedder(vocab_size, self.d_model)
        self.word_positional_embedding_layer = PositionalEmbedding((1, seq_len, d_model), positional_embedding_dropout) 

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
                        model_official=model_official,
                        return_intermediate=False
                    )
        
        self.head = nn.Linear(d_model, vocab_size)
        
        if weight_load and model_official is not None:
            self.load_weights(model_official)

        elif weight_init:
            self.init_weights(embedding_matrix, emb_weights_req_grad)
        
    # TODO - add <start> and <end> token
    # TODO - change ordering of pos embed and query embed parameters 
    # TODO - check if pos embed should be given at every decoder layer to word
    # TODO - use log softmax?
    def forward(self, captions, memory, positional_embedding_layer):

        """
        Performs a forward pass on the Transformer model
  
        Parameters:
            captions (tensor): Tensor of dimension (batch_size, seq_len)
            memory (tensor): Tensor of dimension (batch_size, num_tokens, d_model)
            positional_embedding_layer (nn.Moodule): Tensor of dimension (batch_size, num_tokens, d_model)
        
        Returns:
            x (tensor): Tensor of dimension (1, batch_size, seq_len, vocab_size) OR (depth, batch_size, seq_len, vocab_size)
        """

        tgt_mask = self.make_tgt_mask(captions)     # (batch_size, 1, seq_len, seq_len) 

        target = self.target_embedding(captions)    # (batch_size, seq_len, embed_dim)
        
        # (1, batch_size, seq_len, embed_dim) OR (depth, batch_size, seq_len, embed_dim)
        x = self.decoder(target=target, memory=memory, positional_embedding_layer=positional_embedding_layer, 
                        query_embedding=self.word_positional_embedding_layer, mask=tgt_mask)
        
        # (1, batch_size, seq_len, vocab_size) OR (depth, batch_size, seq_len, vocab_size)
        x = self.head(x).softmax(dim=-1)

        return x
    

    def init_weights(self, embedding_matrix, emb_weights_req_grad):

        """
        Initialises the weights and biases of the Caption Decoder model.
        These parameters include positional embeddings.
        """

        self.target_embedding.init_word_embeddings(embedding_matrix, emb_weights_req_grad)
        trunc_normal_(self.positional_embedding_layer.positional_embedding, std=.02)
            

    def load_weights(self, model_official):

        """
        Loads the weights and biases from the pre-trained model to the current model
        These weights include positional embeddings.

        Parameters:
            `model_custom`: The current ViViT model
            `model_official`: The model which would be used to load the pre-trained weights
        """

        load_positional_embeddings(self, model_official)
    
    
    def make_tgt_mask(self, target):
        batch_size, seq_len = target.shape
        tgt_mask = torch.tril(torch.ones((seq_len, seq_len))).expand(batch_size, 1, seq_len, seq_len)
        return tgt_mask