import sys
sys.path.insert(1, '../config')

import torch
import numpy as np
import timm
import time
from models.bimodal_encoder import BiModalEncoder
from config import load_config


# Helpers
def get_n_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def assert_tensors_equal(t1, t2):
    a1, a2 = t1.detach().numpy(), t2.detach().numpy()
    np.testing.assert_allclose(a1, a2)


cfg = load_config()

model_name = cfg.pretrained_models.vit
model_official = timm.create_model(model_name, pretrained=True)
model_official.eval()


model_custom = BiModalEncoder(**cfg.bimodal, model_official=model_official)
model_custom.eval()

print(get_n_params(model_custom))


# for (name_custom, parameter_custom) in model_custom.named_parameters():
#     print(f"{name_custom}, {parameter_custom.shape}")

    
