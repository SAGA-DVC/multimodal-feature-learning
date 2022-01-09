import sys
sys.path.insert(0, '/home/arnav/Documents/projects/multimodal-feature-learning/dataset')

import torch
import numpy as np
import timm
import time
from vivit import VideoVisionTransformer
import kinetics



# Helpers
def get_n_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def assert_tensors_equal(t1, t2):
        a1, a2 = t1.detach().numpy(), t2.detach().numpy()
        np.testing.assert_allclose(a1, a2)

model_official = None
model_name = "vit_base_patch16_224"
model_official = timm.create_model(model_name, pretrained=True)
model_official.eval()

models = ['spatio temporal attention', 'factorised encoder', 'factorised self attention', 'factorised dot product attention']
tokenization_method = ['filter inflation', 'central frame']

custom_config = {
        "model_name": models[1],
        "num_frames": 5,
        "num_patches": 196,
        "img_size": 224,
        "spatial_patch_size": 16,
        "temporal_patch_size": 2,
        "tokenization_method": tokenization_method[1],
        "in_channels": 3,
        "d_model": 768,
        "depth": 12,
        "temporal_depth": 4,
        "num_heads": 12,
        "mlp_ratio": 4,
        "qkv_bias": True,
        "positional_embedding_dropout": 0,
        "attention_dropout": 0,
        "projection_dropout": 0,
        "dropout_1": 0,
        "dropout_2": 0,
        "classification_head": False,
        "num_classes": 400,
        "return_preclassifier": False,
        "return_prelogits": False,
        "weight_init": True,
        "model_official": model_official,
}


start_time = time.time()

model_custom = VideoVisionTransformer(**custom_config)
model_custom.eval()


# for (name_custom, parameter_custom) in model_custom.named_parameters():
#     print(f"{name_custom}, {parameter_custom.shape}")

# print('-----------------------')

# for (name_official, parameter_official) in model_official.named_parameters():
#     print(f"{name_official} , {parameter_official.shape}")


dataset, loader = kinetics.get_kinetics(
        kinetics_root="../../data/sample",
        num_temporal_samples=10,
        frame_size=(224, 224),
        batch_size=3
    )

for i, batch in enumerate(iter(loader)):
        res = model_custom(batch['video'])
        print(res.shape)



print(f"--- {time.time() - start_time} seconds ---")