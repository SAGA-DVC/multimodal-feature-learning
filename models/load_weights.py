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


def load_token_embeddings(model_custom, model_official):

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

    if temporal_patch_size > 1 and model_custom.tokenization_method == 'filter inflation':
        model_custom.token_embeddings_layer.project_to_patch_embeddings \
                    .weight.data[:] = torch.unsqueeze(model_official.patch_embed.proj.weight.data, 2) \
                                    .repeat(1, 1, temporal_patch_size, 1, 1) / temporal_patch_size
    
    elif temporal_patch_size > 1 and model_custom.tokenization_method == 'central frame':
        model_custom.token_embeddings_layer.project_to_patch_embeddings \
                    .weight.data[:] = 0
        model_custom.token_embeddings_layer.project_to_patch_embeddings \
                    .weight.data[:, :, temporal_patch_size//2] = model_official.patch_embed.proj.weight.data
    
    else :
        model_custom.token_embeddings_layer.project_to_patch_embeddings \
                .weight.data[:] = torch.unsqueeze(model_official.patch_embed.proj.weight.data, 2)
    

    model_custom.token_embeddings_layer.project_to_patch_embeddings \
                .bias.data[:] = model_official.patch_embed.proj.bias.data



# TODO - check weights for temporal positional embeddings in model 2 
def load_positional_embeddings(model_custom, model_official):

    """
    Loads the positional embeddings from the pre-trained model to the current model for a specific model_name and whether the model uses a distillation token.

    Parameters:
        `model_custom`: The current ViViT model
        `model_official`: The model which would be used to load the pre-trained weights
    """

    num_frames = model_custom.num_frames

    if model_custom.model_name == 'spatio temporal attention':
        
        # (1, num_frames * num_tokens + 1, d_model) -> model_custom positional embeddings
        # (1, num_patches + 1, d_model) -> model_official positional embeddings
        # First, the positional embedding for the cls token is initialised followed by those for 'num_frames * num_patches' tokens

        model_custom.positional_embedding_layer.positional_embedding.data[:, 0] = model_official.pos_embed.data[:, 0]
        
        model_custom.positional_embedding_layer.positional_embedding.data[:, 1:] = model_official.pos_embed.data[:, 1:].repeat(1, num_frames, 1)


    elif model_custom.model_name == 'factorised encoder':

        # (1, num_patches + 1, d_model) -> model_custom spatial positional embeddings
        # (1, num_frames + 1, d_model) -> model_custom temporal positional embeddings
        # (1, num_patches + 1, d_model) -> model_official positional embeddings
        # First, the positional embedding for the cls token is initialised followed by those for num_patches tokens

        model_custom.spatial_positional_embedding_layer.positional_embedding.data[:] = model_official.pos_embed.data

        trunc_normal_(model_custom.temporal_positional_embedding_layer.positional_embedding, std=.02)


    # no cls token
    elif model_custom.model_name == 'factorised self attention' or model_custom.model_name == 'factorised dot product attention':
        model_custom.positional_embedding_layer.positional_embedding.data[:] = torch.unsqueeze(model_official.pos_embed.data[:, 1:], 1).repeat(1, num_frames, 1, 1)



# TODO - check if temporal cls token weights should be same as cls token of vit
def load_cls_tokens(model_custom, model_official):
    
    """
    Loads the [class] token from the pre-trained model to the current model

    Parameters:
        `model_custom`: The current ViViT model
        `model_official`: The model which would be used to load the pre-trained weights
    """

    # (1, 1, d_model) -> model_custom cls token
    # (1, 1, d_model) -> model_official cls token

    if model_custom.model_name == 'spatio temporal attention' :
        model_custom.vivitEncoder.cls.data[:] = model_official.cls_token.data

    elif model_custom.model_name == 'factorised encoder':
        model_custom.vivitEncoder.spacial_token.data[:] = model_official.cls_token.data
        trunc_normal_(model_custom.vivitEncoder.temporal_token, std=.02)



def load_vivit_encoder_weights(model_custom, model_official):

    """
    Loads the encoder blocks' weights and biases from the pre-trained model to the current model, given a specific model_name/attention architecture.
    These weights include normalisation layers, qkv layers, and mlp layers.

    Parameters:
        `model_custom`: The current ViViT model
        `model_official`: The model which would be used to load the pre-trained weights
        `model_name` (string): One of 'spatio temporal attention', 'factorised encoder', 'factorised self attention' or 'factorised dot product attention'
    """

    if model_custom.model_name == 'spatio temporal attention':

        depth = len(model_custom.vivitEncoder.encoder)
        
        for (name_custom, parameter_custom), (name_official, parameter_official) in zip(
            list(model_custom.named_parameters())[3 : (12 * depth) + 3], 
            list(model_official.named_parameters())[4 : 144 + 4]):

            # print(f"{name_official} | {name_custom}")

            parameter_custom.data[:] = parameter_official.data

    elif model_custom.model_name == 'factorised encoder':

        spatial_depth = len(model_custom.vivitEncoder.spatialEncoder)
        temporal_depth = len(model_custom.vivitEncoder.temporalEncoder)
        
        #spatial
        for (name_custom, parameter_custom), (name_official, parameter_official) in zip(
            list(model_custom.named_parameters())[4 : (12 * spatial_depth) + 4], 
            list(model_official.named_parameters())[4 : 144 + 4]):

            # print(f"{name_official} | {name_custom}")

            parameter_custom.data[:] = parameter_official.data

        #temporal
        for (name_custom, parameter_custom), (name_official, parameter_official) in zip(
            list(model_custom.named_parameters())[144 + 4 : 144 + (12 * temporal_depth) + 4], 
            list(model_official.named_parameters())[4 : 144 + 4]):

            # print(f"{name_official} | {name_custom}")

            parameter_custom.data[:] = parameter_official.data
    
    elif model_custom.model_name == 'factorised self attention':

        depth = len(model_custom.vivitEncoder.encoder)

        for i in range(depth):
            model_custom.vivitEncoder.encoder[i].layer_norm_1.weight.data[:] = model_official.blocks[i].norm1.weight.data
            model_custom.vivitEncoder.encoder[i].layer_norm_1.bias.data[:] = model_official.blocks[i].norm1.bias.data

            model_custom.vivitEncoder.encoder[i].spatial_attention.qkv.weight.data[:] = model_official.blocks[i].attn.qkv.weight.data
            model_custom.vivitEncoder.encoder[i].spatial_attention.qkv.bias.data[:] = model_official.blocks[i].attn.qkv.bias.data

            model_custom.vivitEncoder.encoder[i].spatial_attention.projection_layer.weight.data[:] = model_official.blocks[i].attn.proj.weight.data
            model_custom.vivitEncoder.encoder[i].spatial_attention.projection_layer.bias.data[:] = model_official.blocks[i].attn.proj.bias.data

            model_custom.vivitEncoder.encoder[i].layer_norm_2.weight.data[:] = model_official.blocks[i].norm2.weight.data
            model_custom.vivitEncoder.encoder[i].layer_norm_2.bias.data[:] = model_official.blocks[i].norm2.bias.data

            model_custom.vivitEncoder.encoder[i].mlp.fully_connected_1.weight.data[:] = model_official.blocks[i].mlp.fc1.weight.data
            model_custom.vivitEncoder.encoder[i].mlp.fully_connected_1.bias.data[:] = model_official.blocks[i].mlp.fc1.bias.data

            model_custom.vivitEncoder.encoder[i].mlp.fully_connected_2.weight.data[:] = model_official.blocks[i].mlp.fc2.weight.data
            model_custom.vivitEncoder.encoder[i].mlp.fully_connected_2.bias.data[:] = model_official.blocks[i].mlp.fc2.bias.data

    elif model_custom.model_name == 'factorised dot product attention':

        depth = len(model_custom.vivitEncoder.encoder)

        for i in range(depth):
            model_custom.vivitEncoder.encoder[i].layer_norm_1.weight.data[:] = model_official.blocks[i].norm1.weight.data
            model_custom.vivitEncoder.encoder[i].layer_norm_1.bias.data[:] = model_official.blocks[i].norm1.bias.data

            model_custom.vivitEncoder.encoder[i].attention.qkv.weight.data[:] = model_official.blocks[i].attn.qkv.weight.data
            model_custom.vivitEncoder.encoder[i].attention.qkv.bias.data[:] = model_official.blocks[i].attn.qkv.bias.data

            model_custom.vivitEncoder.encoder[i].attention.projection_layer.weight.data[:] = model_official.blocks[i].attn.proj.weight.data
            model_custom.vivitEncoder.encoder[i].attention.projection_layer.bias.data[:] = model_official.blocks[i].attn.proj.bias.data

            model_custom.vivitEncoder.encoder[i].layer_norm_2.weight.data[:] = model_official.blocks[i].norm2.weight.data
            model_custom.vivitEncoder.encoder[i].layer_norm_2.bias.data[:] = model_official.blocks[i].norm2.bias.data

            model_custom.vivitEncoder.encoder[i].mlp.fully_connected_1.weight.data[:] = model_official.blocks[i].mlp.fc1.weight.data
            model_custom.vivitEncoder.encoder[i].mlp.fully_connected_1.bias.data[:] = model_official.blocks[i].mlp.fc1.bias.data

            model_custom.vivitEncoder.encoder[i].mlp.fully_connected_2.weight.data[:] = model_official.blocks[i].mlp.fc2.weight.data
            model_custom.vivitEncoder.encoder[i].mlp.fully_connected_2.bias.data[:] = model_official.blocks[i].mlp.fc2.bias.data


def load_classification_weights(model_custom, model_official):

    """
    Loads the classifier's weights and biases from the pre-trained model to the current model.

    Parameters:
        `model_custom`: The current model
        `model_official`: The model which would be used to load the pre-trained weights
    """

    for (name_custom, parameter_custom), (name_official, parameter_official) in zip(
        list(model_custom.named_parameters())[-4:], list(model_official.named_parameters())[-4:]):

        # print(f"{name_official} | {name_custom}")

        parameter_custom.data[:] = parameter_official.data


def load_bimodal_encoder_weights(model_custom, model_official):

    """
    Loads the encoder blocks' weights and biases from the pre-trained model to the current model.
    These weights include normalisation layers, qkv layers, and mlp layers.

    Parameters:
        `model_custom`: The current bi-modal model
        `model_official`: The model which would be used to load the pre-trained weights
    """

    depth = len(model_custom.bi_modal_encoder)
    d_model = model_custom.d_model

    for i in range(depth):
        # Layer Norm
        model_custom.bi_modal_encoder[i].layer_norm_av_1.weight.data[:] = model_official.blocks[i].norm1.weight.data
        model_custom.bi_modal_encoder[i].layer_norm_av_1.bias.data[:] = model_official.blocks[i].norm1.bias.data

        model_custom.bi_modal_encoder[i].layer_norm_va_1.weight.data[:] = model_official.blocks[i].norm1.weight.data
        model_custom.bi_modal_encoder[i].layer_norm_va_1.bias.data[:] = model_official.blocks[i].norm1.bias.data

        model_custom.bi_modal_encoder[i].layer_norm_av_2.weight.data[:] = model_official.blocks[i].norm2.weight.data
        model_custom.bi_modal_encoder[i].layer_norm_av_2.bias.data[:] = model_official.blocks[i].norm2.bias.data

        model_custom.bi_modal_encoder[i].layer_norm_va_2.weight.data[:] = model_official.blocks[i].norm2.weight.data
        model_custom.bi_modal_encoder[i].layer_norm_va_2.bias.data[:] = model_official.blocks[i].norm2.bias.data


        # Cross Attention
        model_custom.bi_modal_encoder[i].attention_av.q_linear.weight.data[:] = model_official.blocks[i].attn.qkv.weight.data[:d_model]
        model_custom.bi_modal_encoder[i].attention_av.q_linear.bias.data[:] = model_official.blocks[i].attn.qkv.bias.data[:d_model]

        model_custom.bi_modal_encoder[i].attention_av.k_linear.weight.data[:] = model_official.blocks[i].attn.qkv.weight.data[:d_model]
        model_custom.bi_modal_encoder[i].attention_av.k_linear.bias.data[:] = model_official.blocks[i].attn.qkv.bias.data[:d_model]

        model_custom.bi_modal_encoder[i].attention_av.v_linear.weight.data[:] = model_official.blocks[i].attn.qkv.weight.data[:d_model]
        model_custom.bi_modal_encoder[i].attention_av.v_linear.bias.data[:] = model_official.blocks[i].attn.qkv.bias.data[:d_model]

        model_custom.bi_modal_encoder[i].attention_va.q_linear.weight.data[:] = model_official.blocks[i].attn.qkv.weight.data[:d_model]
        model_custom.bi_modal_encoder[i].attention_va.q_linear.bias.data[:] = model_official.blocks[i].attn.qkv.bias.data[:d_model]

        model_custom.bi_modal_encoder[i].attention_va.k_linear.weight.data[:] = model_official.blocks[i].attn.qkv.weight.data[:d_model]
        model_custom.bi_modal_encoder[i].attention_va.k_linear.bias.data[:] = model_official.blocks[i].attn.qkv.bias.data[:d_model]

        model_custom.bi_modal_encoder[i].attention_va.v_linear.weight.data[:] = model_official.blocks[i].attn.qkv.weight.data[:d_model]
        model_custom.bi_modal_encoder[i].attention_va.v_linear.bias.data[:] = model_official.blocks[i].attn.qkv.bias.data[:d_model]


        # Projection layer
        model_custom.bi_modal_encoder[i].attention_av.projection_layer.weight.data[:] = model_official.blocks[i].attn.proj.weight.data
        model_custom.bi_modal_encoder[i].attention_av.projection_layer.bias.data[:] = model_official.blocks[i].attn.proj.bias.data

        model_custom.bi_modal_encoder[i].attention_va.projection_layer.weight.data[:] = model_official.blocks[i].attn.proj.weight.data
        model_custom.bi_modal_encoder[i].attention_va.projection_layer.bias.data[:] = model_official.blocks[i].attn.proj.bias.data

        # MLP
        model_custom.bi_modal_encoder[i].mlp_av.fully_connected_1.weight.data[:] = model_official.blocks[i].mlp.fc1.weight.data
        model_custom.bi_modal_encoder[i].mlp_av.fully_connected_1.bias.data[:] = model_official.blocks[i].mlp.fc1.bias.data

        model_custom.bi_modal_encoder[i].mlp_av.fully_connected_2.weight.data[:] = model_official.blocks[i].mlp.fc2.weight.data
        model_custom.bi_modal_encoder[i].mlp_av.fully_connected_2.bias.data[:] = model_official.blocks[i].mlp.fc2.bias.data

        model_custom.bi_modal_encoder[i].mlp_va.fully_connected_1.weight.data[:] = model_official.blocks[i].mlp.fc1.weight.data
        model_custom.bi_modal_encoder[i].mlp_va.fully_connected_1.bias.data[:] = model_official.blocks[i].mlp.fc1.bias.data

        model_custom.bi_modal_encoder[i].mlp_va.fully_connected_2.weight.data[:] = model_official.blocks[i].mlp.fc2.weight.data
        model_custom.bi_modal_encoder[i].mlp_va.fully_connected_2.bias.data[:] = model_official.blocks[i].mlp.fc2.bias.data