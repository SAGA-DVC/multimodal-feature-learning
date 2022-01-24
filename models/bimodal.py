""" Bimodal encoder which gets audio and video features as input """

import torch
import torch.nn as nn

from vivit.modules import BiModalEncoderBlock


class BiModalEncoder(nn.Module):
    def __init__(self, d_model=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, 
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
            `positional_embedding_dropout` (float): dropout probability for the positional embeddings (default 0.0)
            `attention_dropout` (float): Dropout probability for the layer after the multi-head attention mechanism (default 0.0)
            `dropout_1` (float): dropout probability for the MLP block (default 0.0)
            `dropout_2` (float): dropout probability for the MLP block (default 0.0)
            `classification_head` (boolean): If True, a classification head (fully connected layer) is added on top of the model (default False)
            `num_classes` (int): number of classes for the prediction task (default None)
            `return_preclassifier` (boolean): If True, return the representation after the transformer encoder. Useful if using this as the backbone stem as part of a bigger architecture (default False)
            `return_prelogits` (boolean): If True, return the final representation of the network before the classification head. Useful when using features for a downstream task (default False)
            `weight_init` (boolean): If True, initialises the weights of the model (default True)
            `weight_load` (boolean): If True, loads the weights of the specified pre-trained model after initialisation (default False)
            `model_official`: This model's weights are used by ViViT
        """

        super(BiModalEncoder, self).__init__()

        self.num_classes = num_classes
        self.depth = depth

        self.classification_head = classification_head
        self.return_prelogits = return_prelogits
        self.return_preclassifier = return_preclassifier
        
        self.bi_modal_encoder = nn.Sequential(
            *[
                BiModalEncoderBlock(d_model=d_model,
                            num_heads=num_heads,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            dropout_1=dropout_1,
                            dropout_2=dropout_2,
                            attention_dropout=attention_dropout,
                            projection_dropout=projection_dropout
                        )
                for _ in range(depth)
                ]
            )

        self.layer_norm_vid = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_aud = nn.LayerNorm(d_model, eps=1e-6) 

        self.head = nn.Linear(d_model, num_classes) if classification_head else nn.Identity() 

        # if weight_load and model_official is not None:
        #     self.load_weights(model_official)

        # else:
        #     self.init_weights()


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

        vid, aud = self.bi_modal_encoder(vid, aud)

        if self.return_preclassifier :
            return vid, aud

        vid = self.layer_norm_vid(vid)
        aud = self.layer_norm_aud(aud)

        # some processing
        x = vid 

        if self.return_prelogits:
            return x # (batch_size, 1, d_model)
        
        x = self.head(x) # (batch_size, num_classes)
        
        return x
