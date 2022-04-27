import argparse
import torch


def convert(state_dict):
    new_dict = {
        "encoder.cls_token": state_dict["module.v.cls_token"],
        "encoder.positional_embedding": state_dict["module.v.pos_embed"][:, :61, :],
        "encoder.patch_embeddings_layer.project_to_patch_embeddings.weight": state_dict["module.v.patch_embed.proj.weight"],
        "encoder.patch_embeddings_layer.project_to_patch_embeddings.bias": state_dict["module.v.patch_embed.proj.bias"],
        "encoder.layer_norm.weight": state_dict["module.v.norm.weight"],
        "encoder.layer_norm.bias": state_dict["module.v.norm.bias"]
    }

    for i in range(12):
        prefix = f"encoder.encoderBlocks.{i}."
        old_prefix = f"module.v.blocks.{i}."

        new_dict[prefix + "layer_norm_1.weight"] = state_dict[old_prefix + "norm1.weight"]
        new_dict[prefix + "layer_norm_1.bias"] = state_dict[old_prefix + "norm1.bias"]

        new_dict[prefix + "attention.qkv.weight"] = state_dict[old_prefix + "attn.qkv.weight"]
        new_dict[prefix + "attention.qkv.bias"] = state_dict[old_prefix + "attn.qkv.bias"]

        new_dict[prefix + "attention.projection_layer.weight"] = state_dict[old_prefix + "attn.proj.weight"]
        new_dict[prefix + "attention.projection_layer.bias"] = state_dict[old_prefix + "attn.proj.bias"]

        new_dict[prefix + "layer_norm_2.weight"] = state_dict[old_prefix + "norm2.weight"]
        new_dict[prefix + "layer_norm_2.bias"] = state_dict[old_prefix + "norm2.bias"]

        new_dict[prefix + "mlp.fully_connected_1.weight"] = state_dict[old_prefix + "mlp.fc1.weight"]
        new_dict[prefix + "mlp.fully_connected_1.bias"] = state_dict[old_prefix + "mlp.fc1.bias"]

        new_dict[prefix + "mlp.fully_connected_2.weight"] = state_dict[old_prefix + "mlp.fc2.weight"]
        new_dict[prefix + "mlp.fully_connected_2.bias"] = state_dict[old_prefix + "mlp.fc2.bias"]
    
    return new_dict

def main(args):
    state_dict = torch.load(args.input_pth)
    new_dict = convert(state_dict)
    torch.save(new_dict, args.output_pth)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts Audioset AST weights to compatible AST weights')

    parser.add_argument('--input-pth', required=True, type=str,
                      help='Path to pth file of the Audioset pretrained weights')
    parser.add_argument('--output-pth', required=True, type=str,
                      help='Where to save the new state dict pth')

    args = parser.parse_args()

    main(args)



