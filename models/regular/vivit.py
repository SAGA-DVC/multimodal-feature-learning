""" Video Vision Transformer (ViViT) models in PyTorch

Code used from the following repositories:
1. https://github.com/google-research/scenic
2. https://github.com/rwightman/pytorch-image-models
3. https://github.com/jankrepl/mildlyoverfitted 

"""


import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_, zeros_, ones_

from ..modules.embedding_layers import TokenEmbedding, PositionalEmbedding
from ..modules.encoders import VivitEncoder

from ..load_weights import init_encoder_block_weights, load_token_embeddings, load_positional_embeddings, load_cls_tokens, load_vivit_encoder_weights, load_classification_weights


class VideoVisionTransformer(nn.Module):
    def __init__(self, model_name, num_frames, num_patches, img_size=224, spatial_patch_size=16, temporal_patch_size=1,
                tokenization_method='central frame', in_channels=3, d_model=768, depth=12, temporal_depth=4,num_heads=12, 
                mlp_ratio=4., qkv_bias=True, attention_dropout=0., projection_dropout=0., dropout_1=0., dropout_2=0., 
                pre_norm=True, classification_head=False, num_classes=None, return_preclassifier=False, return_prelogits=False, 
                weight_init=False, weight_load=False, model_official=None):
    
        
        """
        The Video Vision Transformer (ViViT) which consists of 3 attention architectures, namely, 
        'factorised encoder', 'factorised self attention' and 'factorised dot product attention'.
  
        Parameters:
            `model_name` (string): One of 'spatio temporal attention', 'factorised encoder', 'factorised self attention' or 'factorised dot product attention'
            `num_frames` (int): Number of frames of the input video used by the transformer after the token_embedding_layer
            `img_size` (int): dimension of one frame of the video (should be a square i.e. height=width) (default 224)
            `spatial_patch_size` (int): dimension of the patch that will be used to convolve over a frame (default 16)
            `temporal_patch_size` (int): dimension of the patch that will be used to convolve over multiple frames (default 1)
            `tokenization_method` (string): One of 'filter inflation' or 'central frame'. Used for loading pre-trained weights for tubelet embedding (default 'central frame')
            `in_channels` (int): number of channels of the each frame in the video. e.g. RGB. (default 3)
            `d_model` (int): Dimension of the tensors used to compute attention
            `depth` (int): number of encoder blocks. 
            `temporal_depth` (int): number of temporal encoder blocks (for factorised encoder model only)
            `num_heads` (int): Number of attention heads.
            `mlp_ratio` (int): Used to determine the hidden layer dimension of the MLP. (default 4)
            `qkv_bias` (boolean): Determines whether to use bias as part of the query/key/value linear layers in the attention block (default True)
            `attention_dropout` (float): Dropout probability for the layer after the multi-head attention mechanism (default 0.0)
            `projection_dropout` (float): Dropout probability for the layer after the projection layer (default 0.0)
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

        super(VideoVisionTransformer, self).__init__()
        
        if model_name not in ['spatio temporal attention', 'factorised encoder', 'factorised self attention', 'factorised dot product attention'] :
            raise ValueError(f'Unrecognized model: {model_name}. Choose between "spatio temporal attention",\
                            "factorised encoder", "factorised self attention" or "factorised dot product attention"')

        self.model_name = model_name
        self.d_model = d_model

        self.img_size = img_size
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        
        self.num_frames = num_frames
        self.num_patches = num_patches

        self.tokenization_method = tokenization_method

        self.num_classes = num_classes
        self.depth = depth

        self.classification_head = classification_head
        self.return_prelogits = return_prelogits
        self.return_preclassifier = return_preclassifier

        assert classification_head or return_prelogits or return_preclassifier, f"You have classification_head={classification_head}, return_prelogits={return_prelogits} or return_preclassifier={return_preclassifier}. One of them must be true."

        self.token_embeddings_layer = TokenEmbedding(img_size=img_size, spatial_patch_size=spatial_patch_size, 
                                                    temporal_patch_size=temporal_patch_size, in_channels=in_channels, 
                                                    d_model=d_model, layer_norm=None)
        
        self.vivitEncoder = VivitEncoder(model_name=model_name,
                            num_frames=num_frames,
                            num_patches=num_patches,
                            d_model=d_model,
                            num_heads=num_heads,
                            depth=depth,
                            temporal_depth=temporal_depth,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            attention_dropout=attention_dropout,
                            projection_dropout=projection_dropout,
                            dropout_1=dropout_1,
                            dropout_2=dropout_2,
                            pre_norm=pre_norm
                        )
        
        if classification_head:
            self.layer_norm = nn.LayerNorm(d_model, eps=1e-6) 
            self.head = nn.Linear(d_model, num_classes)

        if weight_load and model_official is not None:
            self.load_weights(model_official)

        elif weight_init:
            self.init_weights()

    def forward(self, x, positional_embedding_layer=None, spatial_positional_embedding_layer=None):

        """
        Performs a forward pass on the ViViT model, based on the given attention architecture.
  
        Parameters:
            x (tensor): Tensor of dimension (batch_size, in_channels, num_frames, img_size, img_size)
            positional_embedding_layer (nn.Module): Position embeddings for the inputs
            spatial_positional_embedding_layer (nn.Module): Spatial position embeddings for the 'factorised encoder' model  
            temporal_positional_embedding_layer (nn.Module): Temporal position embeddings for the 'factorised encoder' model  

        Returns:
            x (tensor): if return_preclassifier is True, Tensor of dimension 
                            (batch_size, num_frames * num_patches + 1, d_model) for spatio temporal attention OR
                            (batch_size, num_frames + 1, d_model) for factorised encoder OR 
                            (batch_size, num_frames, num_patches, d_model) for factorised self attention and factorised dot product attention
                        if return_prelogits is True, Tensor of dimension (batch_size, d_model)
                        else Tensor of dimension (batch_size, num_classes)
        """

        batch_size, in_channels, num_frames_in, height, width = x.shape

        assert height == width, f"height and width should be the same i.e. {self.img_size}. You have height={height} and width={width}."
        img_size = height

        assert self.num_frames == num_frames_in // self.temporal_patch_size, f"number of frames in the input video should be equal to {self.num_frames * self.temporal_patch_size}. You have number of frames={num_frames_in}."
        assert self.num_patches == (img_size // self.spatial_patch_size) ** 2, f"image size should be equal to {self.img_size}. You have image size={img_size}."

        x = self.token_embeddings_layer(x) # (batch_size, num_frames, num_patches, d_model)              

        # (batch_size, num_frames * num_patches, d_model) OR
        # (batch_size, num_frames, d_model) OR 
        # (batch_size, num_frames, num_patches, d_model) 
        x = self.vivitEncoder(x, positional_embedding_layer, spatial_positional_embedding_layer) 
        
        if self.return_preclassifier :
            return x 

        # (batch_size, num_frames * num_patches, d_model) -> (batch_size, d_model)
        if self.model_name == 'spatio temporal attention': 
            x = x[:, 0]
        
        # (batch_size, num_frames + 1, d_model) -> (batch_size, d_model)
        elif self.model_name == 'factorised encoder':
            x = x[:, 0]

        # (batch_size, num_frames, num_patches, d_model) -> (batch_size, d_model)
        elif self.model_name == 'factorised self attention' or self.model_name == 'factorised dot product attention':
            x = x.reshape(batch_size, -1, self.d_model) # (batch_size, num_tokens, d_model)
            x = x.mean(dim=1)
       
        if self.return_prelogits:
            return x # (batch_size, d_model)

        elif self.classification_head:
            x = self.layer_norm(x) # check placement before after return_prelogits
            x = self.head(x) # (batch_size, num_classes)
            return x
            


    # TODO - add token embedding layer's weight init
    def init_weights(self):

        """
        Initialises the weights and biases of the all four ViViT models namely 
        spatio temporal attention, factorised encoder, factorised self attention, factorised dot product attention.

        These parameters include token embeddings, positional embeddings, [class] tokens (if any) and 
        encoder blocks (consisting of linear layers and normalisation layers).
        """

        if self.model_name == 'spatio temporal attention':
            trunc_normal_(self.vivitEncoder.cls, std=.02)
            self.vivitEncoder.encoder.apply(init_encoder_block_weights)

        elif self.model_name == 'factorised encoder':
            trunc_normal_(self.vivitEncoder.spacial_token, std=.02)
            trunc_normal_(self.vivitEncoder.temporal_token, std=.02)
            
            self.vivitEncoder.spatialEncoder.apply(init_encoder_block_weights)
            self.vivitEncoder.temporalEncoder.apply(init_encoder_block_weights)
        
        else:
            self.vivitEncoder.encoder.apply(init_encoder_block_weights)

        if self.classification_head:
            ones_(self.layer_norm.weight)
            zeros_(self.layer_norm.bias)
            
            trunc_normal_(self.head.weight, std=.02)
            trunc_normal_(self.head.bias, std=.02)


    def load_weights(self, model_official):

        """
        Loads the weights and biases from the pre-trained model to the current model, given a specific model_name/attention architecture.
        These weights include token embeddings, positional embeddings, [class] tokens (if any) and encoder blocks.

        Parameters:
            `model_custom`: The current ViViT model
            `model_official`: The model which would be used to load the pre-trained weights
        """

        load_token_embeddings(self, model_official)
        
        if self.model_name == 'spatio temporal attention' or self.model_name == 'factorised encoder':
            load_cls_tokens(self, model_official)

        load_vivit_encoder_weights(self, model_official)

        if self.classification_head:
            load_classification_weights(self, model_official)




def build_vivit(args):
    # return VideoVisionTransformer(**args)
    return VideoVisionTransformer(model_name=args.model_name, 
                                num_frames=args.num_frames, 
                                num_patches=args.num_patches, 
                                img_size=args.img_size, 
                                spatial_patch_size=args.spatial_patch_size, 
                                temporal_patch_size=args.temporal_patch_size,
                                tokenization_method=args.tokenization_method, 
                                in_channels=args.in_channels, 
                                d_model=args.d_model, 
                                depth=args.depth, 
                                temporal_depth=args.temporal_depth,
                                num_heads=args.num_heads, 
                                mlp_ratio=args.mlp_ratio, 
                                qkv_bias=args.qkv_bias, 
                                attention_dropout=args.attention_dropout, 
                                projection_dropout=args.projection_dropout, 
                                dropout_1=args.dropout_1, 
                                dropout_2=args.dropout_2,
                                pre_norm=args.pre_norm,
                                classification_head=args.classification_head, 
                                num_classes=args.num_classes,
                                return_preclassifier=args.return_preclassifier, 
                                return_prelogits=args.return_prelogits, 
                                weight_init=args.weight_init, 
                                weight_load=args.weight_load, 
                                model_official=args.model_official
                            )