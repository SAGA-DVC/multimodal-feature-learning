import torch
from vivit import VideoVisionTransformer

models = ['factorised encoder', 'factorised self attention', 'factorised dot product attention']
custom_config = {
        "model_name": models[2],
        "num_frames": 100,
        "num_patches": 196,
        "img_size": 224,
        "spatial_patch_size": 16,
        "temporal_patch_size": 1,
        "in_channels": 3,
        "num_classes": 50,
        "d_model": 768,
        "depth": 6,
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
}

model_custom = VideoVisionTransformer(**custom_config)
# model_custom = model_custom.to(torch.device("cuda:0"))
model_custom.eval()

a = torch.zeros(1, 3, 100, 224, 224)
# a = a.to(torch.device("cuda:0"))
res = model_custom(a)
print(res.shape)


