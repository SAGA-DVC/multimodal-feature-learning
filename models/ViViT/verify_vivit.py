import torch
import numpy as np
import timm
import time
from vivit import VideoVisionTransformer


# Helpers
def get_n_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def assert_tensors_equal(t1, t2):
        a1, a2 = t1.detach().numpy(), t2.detach().numpy()
        np.testing.assert_allclose(a1, a2)


model_name = "vit_base_patch16_224"
model_official = timm.create_model(model_name, pretrained=True)
model_official.eval()

models = ['factorised encoder', 'factorised self attention', 'factorised dot product attention']

custom_config = {
        "model_name": models[0],
        "num_frames": 100,
        "num_patches": 196,
        "img_size": 224,
        "spatial_patch_size": 16,
        "temporal_patch_size": 1,
        "in_channels": 3,
        "num_classes": 1000,
        "d_model": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4,
        "qkv_bias": True,
        "positional_embedding_dropout": 0,
        "attention_dropout": 0,
        "projection_dropout": 0,
        "dropout_1": 0,
        "dropout_2": 0,
        "return_preclassifier": False,
        "return_prelogits": False,
        "weight_init": True,
        "model_official": model_official,
}



start_time = time.time()

model_custom = VideoVisionTransformer(**custom_config)
# # model_custom = model_custom.to(torch.device("cuda:0"))
model_custom.eval()


print(f"--- {time.time() - start_time} seconds ---")

# for (name_custom, parameter_custom) in model_custom.named_parameters():
#     print(f"{name_custom}, {parameter_custom.shape}")

# print('-----------------------')

# for (name_official, parameter_official) in model_official.named_parameters():
#     print(f"{name_official} , {parameter_official.shape}")


# a = torch.zeros(1, 3, 100, 224, 224)
# # a = a.to(torch.device("cuda:0"))
# res = model_custom(a)
# print(res.shape)


