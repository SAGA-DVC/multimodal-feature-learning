import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_, zeros_, ones_
import numpy as np

def assert_tensors_equal(t1, t2):
    a1, a2 = t1.detach().numpy(), t2.detach().numpy()
    np.testing.assert_allclose(a1, a2)


def init_encoder_block_weights(module):
        """ 
        Initialises the weights and biases of the Linear layers and Normalisation layers of the given module.

        Parameters:
            `module`: An entire model or specific part of it consisting of nn.Linear and nn.LayerNorm layers     
        """

        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            ones_(module.weight)
            zeros_(module.bias)


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



def load_positional_embeddings(model_custom, model_official, model_name):

    """
    Loads the positional embeddings from the pre-trained model to the current model for a specific model_name and whether the model uses a distillation token.

    Parameters:
        `model_custom`: The current ViViT model
        `model_official`: The model which would be used to load the pre-trained weights
        `model_name` (string): One of 'spatio temporal attention', 'factorised encoder', 'factorised self attention' or 'factorised dot product attention'
    """

    num_frames = model_custom.num_frames

    # (1, num_frames, num_tokens, d_model) -> model_custom positional embeddings
    # (1, num_patches, d_model) -> model_official positional embeddings

    if model_name == 'spatio temporal attention':
        model_custom.encoder.add_positional_embedding_to_cls.data[:] = torch.unsqueeze(model_official.pos_embed.data[:, 0], 1)
        
        model_custom.encoder.add_positional_embedding \
                    .positional_embedding.data[:] = torch.unsqueeze(model_official.pos_embed.data[:, 1:], 1).repeat(1, num_frames, 1, 1)


    elif model_name == 'factorised encoder':
        if model_custom.distilled:
            # cls
            model_custom.encoder.add_positional_embedding_spatial \
                    .positional_embedding.data[:, :, 0] = torch.unsqueeze(model_official.pos_embed.data[:, 0], 1).repeat(1, num_frames, 1)
            
            # other tokens
            model_custom.encoder.add_positional_embedding_spatial \
                    .positional_embedding.data[:, :, 2:] = torch.unsqueeze(model_official.pos_embed.data[:, 1:], 1).repeat(1, num_frames, 1, 1)

            # temporal
            trunc_normal_(model_custom.encoder.add_positional_embedding_temporal.positional_embedding, std=.02)

        else:
            model_custom.encoder.add_positional_embedding_spatial \
                    .positional_embedding.data[:] = torch.unsqueeze(model_official.pos_embed.data, 1).repeat(1, num_frames, 1, 1)

            trunc_normal_(model_custom.encoder.add_positional_embedding_temporal.positional_embedding, std=.02)


    # no cls token
    elif model_name == 'factorised self attention' or model_name == 'factorised dot product attention':
        model_custom.encoder.add_positional_embedding \
                    .positional_embedding.data[:] = torch.unsqueeze(model_official.pos_embed.data[:, 1:], 1).repeat(1, num_frames, 1, 1)



def load_cls_tokens(model_custom, model_official, model_name):
    
    """
    Loads the [class] token from the pre-trained model to the current model

    Parameters:
        `model_custom`: The current ViViT model
        `model_official`: The model which would be used to load the pre-trained weights
        `model_name` (string): One of 'spatio temporal attention' or 'factorised encoder' 
    """

    # (1, 1, d_model) -> model_custom cls token
    # (1, 1, d_model) -> model_official cls token

    if model_name == 'spatio temporal attention' :
        model_custom.encoder.cls.data[:] = model_official.cls_token.data
    elif model_name == 'factorised encoder':
        model_custom.encoder.spacial_token.data[:] = model_official.cls_token.data



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

        if model_custom.distilled:
            for (name_custom, parameter_custom), (name_official, parameter_official) in zip(
                list(model_custom.named_parameters())[6 : (12 * depth) + 6], 
                list(model_official.named_parameters())[4 : 144 + 4]):

                # print(f"{name_official} | {name_custom}")

                parameter_custom.data[:] = parameter_official.data

        else:
            for (name_custom, parameter_custom), (name_official, parameter_official) in zip(
                list(model_custom.named_parameters())[5 : (12 * depth) + 5], 
                list(model_official.named_parameters())[4 : 144 + 4]):

                # print(f"{name_official} | {name_custom}")

                parameter_custom.data[:] = parameter_official.data

    elif model_name == 'factorised encoder':

        spatial_depth = len(model_custom.encoder.spatialEncoder)
        temporal_depth = len(model_custom.encoder.temporalEncoder)
        
        if model_custom.distilled:
            #spatial
            for (name_custom, parameter_custom), (name_official, parameter_official) in zip(
                list(model_custom.named_parameters())[8 : (12 * spatial_depth) + 8], 
                list(model_official.named_parameters())[4 : 144 + 4]):

                # print(f"{name_official} | {name_custom}")

                parameter_custom.data[:] = parameter_official.data

            #temporal
            for (name_custom, parameter_custom), (name_official, parameter_official) in zip(
                list(model_custom.named_parameters())[144 + 8 : 144 + (12 * temporal_depth) + 8], 
                list(model_official.named_parameters())[4 : 144 + 4]):

                # print(f"{name_official} | {name_custom}")

                parameter_custom.data[:] = parameter_official.data

        else:
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