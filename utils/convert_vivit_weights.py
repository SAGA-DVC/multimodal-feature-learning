"""Convert Flax checkpoints from original paper to PyTorch"""
from argparse import ArgumentParser
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from flax.training.checkpoints import restore_checkpoint


def transform_state_encoder_block(state_dict, i):
    state = state_dict['optimizer']['target']['Transformer'][f'encoderblock_{i}']

    new_state = OrderedDict()

    prefix = f'encoder.basicEncoder.{i}.'
    new_state = {
        prefix+'mlp.fully_connected_1.bias': state['MlpBlock_0']['Dense_0']['bias'],
        prefix+'mlp.fully_connected_1.weight': state['MlpBlock_0']['Dense_0']['kernel'].transpose(),
        prefix+'mlp.fully_connected_2.bias': state['MlpBlock_0']['Dense_1']['bias'],
        prefix+'mlp.fully_connected_2.weight': state['MlpBlock_0']['Dense_1']['kernel'].transpose(),

        prefix+'layer_norm_1.bias': state['LayerNorm_0']['bias'],
        prefix+'layer_norm_1.weight': state['LayerNorm_0']['scale'],
        prefix+'layer_norm_2.bias': state['LayerNorm_1']['bias'],
        prefix+'layer_norm_2.weight': state['LayerNorm_1']['scale'],
    }
    qbias = state['MultiHeadDotProductAttention_0']['query']['bias']
    qweight = state['MultiHeadDotProductAttention_0']['query']['kernel'].reshape(768, 768).transpose()

    kbias = state['MultiHeadDotProductAttention_0']['key']['bias']
    kweight = state['MultiHeadDotProductAttention_0']['key']['kernel'].reshape(768, 768).transpose()

    vbias = state['MultiHeadDotProductAttention_0']['value']['bias']
    vweight = state['MultiHeadDotProductAttention_0']['value']['kernel'].reshape(768, 768).transpose()

    qkv_bias = np.concatenate((qbias, kbias, vbias), axis=0)
    qkv_weight = np.concatenate((qweight, kweight, vweight), axis=0)

    new_state[prefix + "attention.qkv.bias"] = qkv_bias
    new_state[prefix + "attention.qkv.weight"] = qkv_weight

    new_state[prefix+'attention.projection_layer.bias'] = state['MultiHeadDotProductAttention_0']['out']['bias']
    new_state[prefix+'attention.projection_layer.weight'] = state['MultiHeadDotProductAttention_0']['out']['kernel'].reshape(768, 768).transpose()

    return new_state


def transform_state(state_dict, transformer_layers=12):
    new_state = OrderedDict()

    new_state['layer_norm.bias'] = state_dict['optimizer']['target']['Transformer']['encoder_norm']['bias']
    new_state['layer_norm.weight'] = state_dict['optimizer']['target']['Transformer']['encoder_norm']['scale']

    # [768, 3, 2, 16, 16] <-- (2, 16, 16, 3, 768)
    new_state['token_embeddings_layer.project_to_patch_embeddings.weight'] = state_dict['optimizer']['target']['embedding']['kernel'].transpose((4, 3, 0, 1, 2))
    new_state['token_embeddings_layer.project_to_patch_embeddings.bias'] = state_dict['optimizer']['target']['embedding']['bias']

    new_state['encoder.cls'] = state_dict['optimizer']['target']['cls']

    # (1, 16, 196, 768) <-- (1, 3137, 768)
    new_state['encoder.add_positional_embedding.positional_embedding'] = state_dict['optimizer']['target']['Transformer']['posembed_input']['pos_embedding'][:, :-1, :].reshape(1, 16, 196, 768)
    new_state['encoder.add_positional_embedding_to_cls'] = state_dict['optimizer']['target']['Transformer']['posembed_input']['pos_embedding'][:, -1, :]

    for i in range(transformer_layers):
        new_state.update(transform_state_encoder_block(state_dict, i))

    
    return {k: torch.tensor(v) for k,v in new_state.items()}


def get_n_layers(state_dict):
    return sum([1 if 'encoderblock_' in k else 0 for k in state_dict['optimizer']['target']['Transformer'].keys()])


if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument('--flax_model', type=str, help='Path to flax model', required=True)
    parser.add_argument('--output_model_name', type=str, help='Name of the outputed file', required=True)
    
    args = parser.parse_args()
    
    state_dict = restore_checkpoint(args.flax_model, None)
    
    n_layers = get_n_layers(state_dict)
    new_state = transform_state(state_dict, n_layers)
    
    out_path = Path(args.flax_model).parent.absolute()
    
    if '.pt' in args.output_model_name:
        out_path = out_path / args.output_model_name
        
    else:
        out_path = out_path / (args.output_model_name + '.pt')
    
    torch.save(new_state, out_path)
