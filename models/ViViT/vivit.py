""" Video Vision Transformer (ViViT) models in PyTorch

Code used from the following repositories:
1. https://github.com/google-research/scenic
2. https://github.com/rwightman/pytorch-image-models
3. https://github.com/jankrepl/mildlyoverfitted 

"""


import torch
import torch.nn as nn
import numpy as np
from modules import TokenEmbedding, Encoder


class VideoVisionTransformer(nn.Module):
    def __init__(self, model_name, num_frames, num_patches, img_size=224, spatial_patch_size=16, temporal_patch_size=1,
                tokenization_method='central frame', in_channels=3, d_model=768, depth=12, temporal_depth=4,num_heads=12, 
                mlp_ratio=4., qkv_bias=True, positional_embedding_dropout=0., attention_dropout=0., 
                projection_dropout=0., dropout_1=0., dropout_2=0., classification_head=False, num_classes=None,
                return_preclassifier=False, return_prelogits=False, weight_init=False, model_official=None):
        
        """
        The Video Vision Transformer (ViViT) which consists of 3 attention architectures, namely, 
        'factorised encoder', 'factorised self attention' and 'factorised dot product attention'.
  
        Parameters:
            `model_name` (string): One of 'spatio temporal attention', 'factorised encoder', 'factorised self attention' or 'factorised dot product attention'
            `num_frames` (int): Number of frames in the input video
            `num_patches` (int): Number of patches per frame in the input video
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
            `positional_embedding_dropout` (float): dropout probability for the positional embeddings (default 0.0)
            `attention_dropout` (float): Dropout probability for the layer after the multi-head attention mechanism (default 0.0)
            `projection_dropout` (float): Dropout probability for the layer after the projection layer (default 0.0)
            `dropout_1` (float): dropout probability for the MLP block (default 0.0)
            `dropout_2` (float): dropout probability for the MLP block (default 0.0)
            `classification_head` (boolean): If True, a classification head (fully connected layer) is added on top of the model (default False)
            `num_classes` (int): number of classes for the prediction task (default None)
            `return_preclassifier` (boolean): If True, return the representation after the transformer encoder. Useful if using this as the backbone stem as part of a bigger architecture (default False)
            `return_prelogits` (boolean): If True, return the final representation of the network before the classification head. Useful when using features for a downstream task (default False)
            `weight_init` (boolean): If True, loads the weights of the specified pre-trained model after initialisation (default False)
            `model_official`: This model's weights are used by ViViT
        """

        super(VideoVisionTransformer, self).__init__()
        
        if model_name not in ['spatio temporal attention', 'factorised encoder', 'factorised self attention', 'factorised dot product attention'] :
            raise ValueError(f'Unrecognized model: {model_name}. Choose between "spatio temporal attention",\
                            "factorised encoder", "factorised self attention" or "factorised dot product attention"')

        self.model_name = model_name
        self.num_frames = num_frames
        self.num_patches = num_patches # remove num_patches as parameter later and replace with img_size//spatial_patch_size
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
                            temporal_depth=temporal_depth,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            positional_embedding_dropout=positional_embedding_dropout,
                            dropout_1=dropout_1,
                            dropout_2=dropout_2,
                            attention_dropout=attention_dropout,
                            projection_dropout=projection_dropout
                        )
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6) 

        self.head = nn.Linear(d_model, num_classes) if classification_head else nn.Identity() 

        if weight_init and model_official is not None:
            self.load_weights(self, model_official, model_name, tokenization_method)

    @staticmethod
    def load_weights(model_custom, model_official, model_name, tokenization_method):

        """
        Loads the weights and biases from the pre-trained model to the current model, given a specific model_name/attention architecture.
        These weights include token embeddings, positional embeddings, [class] tokens (if any) and encoder blocks.

        Parameters:
            `model_custom`: The current ViViT model
            `model_official`: The model which would be used to load the pre-trained weights
            `model_name` (string): One of 'spatio temporal attention', 'factorised encoder', 'factorised self attention' or 'factorised dot product attention'
        """

        VideoVisionTransformer.load_token_embeddings(model_custom, model_official, tokenization_method)

        VideoVisionTransformer.load_positional_embeddings(model_custom, model_official, model_name)

        if model_name == 'spatio temporal attention' or model_name == 'factorised encoder':
            VideoVisionTransformer.load_cls_tokens(model_custom, model_official, model_name)

        VideoVisionTransformer.load_encoder_weights(model_custom, model_official, model_name)

        # VideoVisionTransformer.load_classification_weights(model_custom, model_official)

    @staticmethod
    def assert_tensors_equal(t1, t2):
        a1, a2 = t1.detach().numpy(), t2.detach().numpy()
        np.testing.assert_allclose(a1, a2)


    @staticmethod
    def load_token_embeddings(model_custom, model_official, tokenization_method):

        """
        Loads the weights and biases for the token embeddings from the pre-trained model to the current model.

        Parameters:
            `model_custom`: The current ViViT model
            `model_official`: The model which would be used to load the pre-trained weights
            `tokenization_method` (string): One of 'filter inflation' or 'central frame'. Used for loading pre-trained weights for tubelet embedding
        """
        temporal_patch_size = model_custom.token_embeddings_layer.temporal_patch_size

        # (batch_size, num_channels, num_frames, width, height) -> model_custom token embeddings layer
        # (batch_size, num_channels, width, height) -> model_official token embeddings layer

        if temporal_patch_size > 1 and tokenization_method == 'filter inflation':
            model_custom.token_embeddings_layer.project_to_patch_embeddings \
                        .weight.data[:] = torch.unsqueeze(model_official.patch_embed.proj.weight.data, 2) \
                                        .repeat(1, 1, temporal_patch_size, 1, 1) / temporal_patch_size
        
        elif temporal_patch_size > 1 and tokenization_method == 'central frame':
            model_custom.token_embeddings_layer.project_to_patch_embeddings \
                        .weight.data[:] = 0
            model_custom.token_embeddings_layer.project_to_patch_embeddings \
                        .weight.data[:, :, temporal_patch_size//2] = model_official.patch_embed.proj.weight.data
        
        else :
            model_custom.token_embeddings_layer.project_to_patch_embeddings \
                    .weight.data[:] = torch.unsqueeze(model_official.patch_embed.proj.weight.data, 2)
        

        model_custom.token_embeddings_layer.project_to_patch_embeddings \
                    .bias.data[:] = model_official.patch_embed.proj.bias.data


    @staticmethod
    def load_positional_embeddings(model_custom, model_official, model_name):

        """
        Loads the positional embeddings from the pre-trained model to the current model, given a specific model_name/attention architecture.

        Parameters:
            `model_custom`: The current ViViT model
            `model_official`: The model which would be used to load the pre-trained weights
            `model_name` (string): One of 'spatio temporal attention', 'factorised encoder', 'factorised self attention' or 'factorised dot product attention'
        """

        if model_name == 'spatio temporal attention':
            num_frames = model_custom.encoder.add_positional_embedding.positional_embedding.shape[1]

            model_custom.encoder.add_positional_embedding_to_cls.data[:] = torch.unsqueeze(model_official.pos_embed.data[:, 0], 1)

            model_custom.encoder.add_positional_embedding \
                        .positional_embedding.data[:] = torch.unsqueeze(model_official.pos_embed.data[:, 1:], 1).repeat(1, num_frames, 1, 1)

        # only spatial for now
        elif model_name == 'factorised encoder':
            num_frames = model_custom.encoder.add_positional_embedding_spatial.positional_embedding.shape[1]

            model_custom.encoder.add_positional_embedding_spatial \
                    .positional_embedding.data[:] = torch.unsqueeze(model_official.pos_embed.data, 1).repeat(1, num_frames, 1, 1)

        # no cls token
        elif model_name == 'factorised self attention' or model_name == 'factorised dot product attention':
            num_frames = model_custom.encoder.add_positional_embedding.positional_embedding.shape[1]
            
            model_custom.encoder.add_positional_embedding \
                        .positional_embedding.data[:] = torch.unsqueeze(model_official.pos_embed.data[:, 1:], 1).repeat(1, num_frames, 1, 1)


    # only spatial (for factorised encoder) for now
    @staticmethod
    def load_cls_tokens(model_custom, model_official, model_name):
        
        """
        Loads the [class] token from the pre-trained model to the current model

        Parameters:
            `model_custom`: The current ViViT model
            `model_official`: The model which would be used to load the pre-trained weights
            `model_name` (string): One of 'spatio temporal attention' or 'factorised encoder' 
        """
        if model_name == 'spatio temporal attention' :
            model_custom.encoder.cls.data[:] = model_official.cls_token.data
        elif model_name == 'factorised encoder':
            model_custom.encoder.spacial_token.data[:] = model_official.cls_token.data


    @staticmethod
    def load_encoder_weights(model_custom, model_official, model_name):

        """
        Loads the encoder blocks' weights and biases from the pre-trained model to the current model, given a specific model_name/attention architecture.
        These weights include normalisation layers, qkv layers, and mlp layers.

        Parameters:
            `model_custom`: The current ViViT model
            `model_official`: The model which would be used to load the pre-trained weights
            `model_name` (string): One of 'spatio temporal attention', 'factorised encoder', 'factorised self attention' or 'factorised dot product attention'
        """

        if model_name == 'spatio temporal attention':

            depth = len(model_custom.encoder.basicEncoder)

            for (name_custom, parameter_custom), (name_official, parameter_official) in zip(
                list(model_custom.named_parameters())[5 : (12 * depth) + 5], 
                list(model_official.named_parameters())[4 : 144 + 4]):

                # print(f"{name_official} | {name_custom}")

                parameter_custom.data[:] = parameter_official.data

        elif model_name == 'factorised encoder':

            spatial_depth = len(model_custom.encoder.spatialEncoder)
            temporal_depth = len(model_custom.encoder.temporalEncoder)

            #spatial
            for (name_custom, parameter_custom), (name_official, parameter_official) in zip(
                list(model_custom.named_parameters())[6 : (12 * spatial_depth) + 6], 
                list(model_official.named_parameters())[4 : 144 + 4]):

                # print(f"{name_official} | {name_custom}")

                parameter_custom.data[:] = parameter_official.data

            #temporal
            for (name_custom, parameter_custom), (name_official, parameter_official) in zip(
                list(model_custom.named_parameters())[144 + 6 : 144 + (12 * temporal_depth) + 6], 
                list(model_official.named_parameters())[4 : 144 + 4]):

                # print(f"{name_official} | {name_custom}")

                parameter_custom.data[:] = parameter_official.data
        
        elif model_name == 'factorised self attention':

            for i in range(12):
                model_custom.encoder.encoder[i].layer_norm_1.weight.data[:] = model_official.blocks[i].norm1.weight.data
                model_custom.encoder.encoder[i].layer_norm_1.bias.data[:] = model_official.blocks[i].norm1.bias.data

                model_custom.encoder.encoder[i].spatial_attention.qkv.weight.data[:] = model_official.blocks[i].attn.qkv.weight.data
                model_custom.encoder.encoder[i].spatial_attention.qkv.bias.data[:] = model_official.blocks[i].attn.qkv.bias.data

                model_custom.encoder.encoder[i].spatial_attention.projection_layer.weight.data[:] = model_official.blocks[i].attn.proj.weight.data
                model_custom.encoder.encoder[i].spatial_attention.projection_layer.bias.data[:] = model_official.blocks[i].attn.proj.bias.data

                model_custom.encoder.encoder[i].layer_norm_2.weight.data[:] = model_official.blocks[i].norm2.weight.data
                model_custom.encoder.encoder[i].layer_norm_2.bias.data[:] = model_official.blocks[i].norm2.bias.data

                model_custom.encoder.encoder[i].mlp.fully_connected_1.weight.data[:] = model_official.blocks[i].mlp.fc1.weight.data
                model_custom.encoder.encoder[i].mlp.fully_connected_1.bias.data[:] = model_official.blocks[i].mlp.fc1.bias.data

                model_custom.encoder.encoder[i].mlp.fully_connected_2.weight.data[:] = model_official.blocks[i].mlp.fc2.weight.data
                model_custom.encoder.encoder[i].mlp.fully_connected_2.bias.data[:] = model_official.blocks[i].mlp.fc2.bias.data

        elif model_name == 'factorised dot product attention':

            for i in range(12):
                model_custom.encoder.encoder[i].layer_norm_1.weight.data[:] = model_official.blocks[i].norm1.weight.data
                model_custom.encoder.encoder[i].layer_norm_1.bias.data[:] = model_official.blocks[i].norm1.bias.data

                model_custom.encoder.encoder[i].attention.qkv.weight.data[:] = model_official.blocks[i].attn.qkv.weight.data
                model_custom.encoder.encoder[i].attention.qkv.bias.data[:] = model_official.blocks[i].attn.qkv.bias.data

                model_custom.encoder.encoder[i].attention.projection_layer.weight.data[:] = model_official.blocks[i].attn.proj.weight.data
                model_custom.encoder.encoder[i].attention.projection_layer.bias.data[:] = model_official.blocks[i].attn.proj.bias.data

                model_custom.encoder.encoder[i].layer_norm_2.weight.data[:] = model_official.blocks[i].norm2.weight.data
                model_custom.encoder.encoder[i].layer_norm_2.bias.data[:] = model_official.blocks[i].norm2.bias.data

                model_custom.encoder.encoder[i].mlp.fully_connected_1.weight.data[:] = model_official.blocks[i].mlp.fc1.weight.data
                model_custom.encoder.encoder[i].mlp.fully_connected_1.bias.data[:] = model_official.blocks[i].mlp.fc1.bias.data

                model_custom.encoder.encoder[i].mlp.fully_connected_2.weight.data[:] = model_official.blocks[i].mlp.fc2.weight.data
                model_custom.encoder.encoder[i].mlp.fully_connected_2.bias.data[:] = model_official.blocks[i].mlp.fc2.bias.data

    @staticmethod
    def load_classification_weights(model_custom, model_official):

        """
        Loads the classifier's weights and biases from the pre-trained model to the current model.

        Parameters:
            `model_custom`: The current ViViT model
            `model_official`: The model which would be used to load the pre-trained weights
        """

        for (name_custom, parameter_custom), (name_official, parameter_official) in zip(
            list(model_custom.named_parameters())[-4:], list(model_official.named_parameters())[-4:]):

            # print(f"{name_official} | {name_custom}")

            parameter_custom.data[:] = parameter_official.data


    def forward(self, x):

        """
        Performs a forward pass on the ViViT model, based on the given attention architecture.
  
        Parameters:
            x (tensor): Tensor of dimension (batch_size, in_channels, num_frames, img_size, img_size)
        
        Returns:
            x (tensor): if return_preclassifier is True, Tensor of dimension 
                            (batch_size, num_frames * num_patches + 1, d_model) for spatio temporal attention OR
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
                                                
        # (batch_size, num_frames * num_patches + 1, d_model) OR (batch_size, num_frames + 1, d_model) OR 
        # (batch_size, num_frames, num_patches, d_model) 
        x = self.encoder(x) 
        
        if self.return_preclassifier :
            return x 

        # (batch_size, num_frames * num_patches + 1, d_model) -> (batch_size, 1, d_model) OR
        # (batch_size, num_frames + 1, d_model) -> (batch_size, 1, d_model)
        if self.model_name == 'spatio temporal attention' or self.model_name == 'factorised encoder':
            x = x[:, 0] 
        
        # (batch_size, num_frames, num_patches, d_model) -> (batch_size, 1, d_model)
        elif self.model_name == 'factorised self attention' or self.model_name == 'factorised dot product attention':
            x = x.reshape(batch_size, -1, d_model) # (batch_size, num_tokens, d_model)
            x = x.mean(dim=1) # (batch_size, 1, d_model)
        
        x = self.layer_norm(x)

        if self.return_prelogits :
            return x # (batch_size, 1, d_model)

        x = self.head(x) # (batch_size, num_classes)

        return x 