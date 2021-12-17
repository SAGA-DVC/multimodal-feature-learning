""" Video Vision Transformer (ViViT) models in PyTorch

Code used from the following repositories:
1. https://github.com/google-research/scenic
2. https://github.com/rwightman/pytorch-image-models
3. https://github.com/jankrepl/mildlyoverfitted 

"""


import torch
import torch.nn as nn

from modules import TokenEmbedding, Encoder


class VideoVisionTransformer(nn.Module):
    def __init__(self, model_name, num_frames, num_patches, img_size=224, spatial_patch_size=16, temporal_patch_size=1,
                in_channels=3, num_classes=1000, d_model=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                positional_embedding_dropout=0., attention_dropout=0., projection_dropout=0., 
                dropout_1=0., dropout_2=0.,
                return_preclassifier=False, return_prelogits=False):
        
        """
        The Video Vision Transformer (ViViT) which consists of 3 attention architectures, namely, 
        'factorised encoder', 'factorised self attention' and 'factorised dot product attention'.
  
        Parameters:
            model_name (string): One of 'factorised encoder', 'factorised self attention' or 'factorised dot product attention'
            num_frames (int): Number of frames in the input video
            num_patches (int): Number of patches per frame in the input video
            img_size (int): dimension of one frame of the video (should be a square i.e. height=width) (default 224)
            spatial_patch_size (int): dimension of the patch that will be used to convolve over a frame (default 16)
            temporal_patch_size (int): dimension of the patch that will be used to convolve over multiple frames (default 1)
            in_channels (int): number of channels of the each frame in the video. e.g. RGB. (default 3)
            num_classes (int): number of classes for the prediction task (default 1000)
            d_model (int): Dimension of the tensors used to compute attention
            depth (int): number of encoder blocks. 
            num_heads (int): Number of attention heads.
            mlp_ratio (int): Used to determine the hidden layer dimension of the MLP. (default 4)
            qkv_bias (boolean): Determines whether to use bias as part of the query/key/value linear layers in the attention block (default True)
            positional_embedding_dropout (float): dropout probability for the positional embeddings (default 0.0)
            attention_dropout (float): Dropout probability for the layer after the multi-head attention mechanism (default 0.0)
            projection_dropout (float): Dropout probability for the layer after the projection layer (default 0.0)
            dropout_1 (float): dropout probability for the MLP block (default 0.0)
            dropout_2 (float): dropout probability for the MLP block (default 0.0)
            return_preclassifier: If true, return the representation after the transformer encoder. Useful if using this as the backbone stem as part of a bigger architecture.
            return_prelogits (boolean): If true, return the final representation of the network before the classification head. Useful when using features for a downstream task.

        """

        super(VideoVisionTransformer, self).__init__()
        
        self.model_name = model_name
        self.num_frames = num_frames
        self.num_patches = num_patches
        self.num_classes = num_classes
        self.depth = depth

        self.return_prelogits = return_prelogits
        self.return_preclassifier = return_preclassifier

        self.token_embeddings_layer = TokenEmbedding(img_size=img_size, spatial_patch_size=spatial_patch_size, 
                                                    temporal_patch_size=temporal_patch_size, in_channels=in_channels, 
                                                    d_model=d_model, layer_norm=None)
        
        self.encoder = Encoder(model_name=model_name,
                            num_frames=num_frames,
                            num_patches=num_patches,
                            d_model=d_model,
                            num_heads=num_heads,
                            depth=depth,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            positional_embedding_dropout=positional_embedding_dropout,
                            dropout_1=dropout_1,
                            dropout_2=dropout_2,
                            attention_dropout=attention_dropout,
                            projection_dropout=projection_dropout
                        )

        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):

        """
        Performs a forward pass on the ViViT model, based on the given attention architecture.
  
        Parameters:
            x (tensor): Tensor of dimension (batch_size, in_channels, num_frames, img_size, img_size)
        
        Returns:
            x (tensor): if return_preclassifier is True, Tensor of dimension 
                            (batch_size, num_frames + 1, d_model) for factorised encoder OR 
                            (batch_size, num_frames, num_patches, d_model) for factorised self attention and factorised dot product attention
                        if return_prelogits is True, Tensor of dimension (batch_size, 1, d_model)
                        else Tensor of dimension (batch_size, num_classes)

        """

        x = self.token_embeddings_layer(x) # (batch_size, num_frames, num_patches, d_model)

        batch_size, num_frames, num_patches, d_model = x.shape
        
        assert self.num_frames == num_frames, f"number of frames should be equal to {self.num_frames}. You \
                                                have num_frames={num_frames}. Adjust the video dimensions or \
                                                patch sizes accordingly."

        assert self.num_patches == num_patches, f"number of patches should be equal to {self.num_patches}. You \
                                                have num_patches={num_patches}. Adjust the video dimensions or \
                                                patch sizes accordingly."
                                                
        # (batch_size, num_frames + 1, d_model)  OR (batch_size, num_frames, num_patches, d_model) 
        x = self.encoder(x) 
        
        if self.return_preclassifier :
            return x # (batch_size, num_frames + 1, d_model)  OR (batch_size, num_frames, num_patches, d_model) 

        if self.model_name == 'factorised encoder':
            x = x[:, 0] # (batch_size, 1, d_model)

        elif self.model_name == 'factorised self attention':
            x = x.reshape(batch_size, -1, d_model) # (batch_size, num_tokens, d_model)
            x = x.mean(dim=1) # (batch_size, 1, d_model)

        elif self.model_name == 'factorised dot product attention':
            x = x.reshape(batch_size, -1, d_model) # (batch_size, num_tokens, d_model)
            x = x.mean(dim=1) # (batch_size, 1, d_model)

        else:
            raise ValueError(f'Unrecognized model: {self.model_name}. Choose between "factorised encoder", \
                            "factorised self attention" or "factorised dot product attention"')
        
        if self.return_prelogits :
            return x # (batch_size, 1, d_model)

        x = self.head(x) # (batch_size, num_classes)

        return x 





