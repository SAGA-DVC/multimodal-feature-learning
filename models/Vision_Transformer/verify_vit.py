import numpy as np
import timm
import torch
from models.Vision_Transformer.vit import VisionTransformer

# Helpers
def get_n_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def assert_tensors_equal(t1, t2):
    a1, a2 = t1.detach().numpy(), t2.detach().numpy()
    np.testing.assert_allclose(a1, a2)

model_name = "vit_base_patch16_384"
model_official = timm.create_model(model_name, pretrained=True)
model_official.eval()
print(type(model_official))

custom_config = {
        "img_size": 384,
        "patch_size": 16,
        "in_channels": 3,
        "d_model": 768,
        "depth": 12,
        "num_heads": 12,
        "qkv_bias": True,
        "mlp_ratio": 4,
}

model_custom = VisionTransformer(**custom_config)
model_custom.eval()


for (name_official, parameter_official), (name_custom, parameter_custom) in zip(
        model_official.named_parameters(), model_custom.named_parameters()
):
    assert parameter_official.numel() == parameter_custom.numel()
    print(f"{name_official} | {name_custom}")

    parameter_custom.data[:] = parameter_official.data

    assert_tensors_equal(parameter_custom.data, parameter_official.data)

inp = torch.rand(1, 3, 384, 384)
res_custom = model_custom(inp)
res_official = model_official(inp)

# Asserts
assert get_n_params(model_custom) == get_n_params(model_official)
assert_tensors_equal(res_custom, res_official)

# Save custom model
torch.save(model_custom, "models/Vision_Transformer/vit_model.pth")