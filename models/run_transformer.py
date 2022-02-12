import sys
sys.path.insert(0, '../dataset')
sys.path.insert(1, '../config')

import torch
import numpy as np
import timm
import time
from transformer import Transformer 
from kinetics import get_kinetics
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


model_custom = Transformer(**cfg.transformer, model_official=model_official)
model_custom.eval()

query_embedding = torch.rand(100, 768)
vid = torch.rand(1, 3, 10, 224, 224)

# for (name_custom, parameter_custom) in model_custom.named_parameters():
#     print(f"{name_custom}, {parameter_custom.shape}")

# print('-----------------------')

# for (name_official, parameter_official) in model_official.named_parameters():
#     print(f"{name_official} , {parameter_official.shape}")


dataset, loader = get_kinetics(**cfg.dataset.kinetics)


start_time = time.time()


# for i, batch in enumerate(iter(loader)):
# 	res = model_custom(batch['video'], target, query_embedding)
# 	print(res.shape)

res = model_custom(vid, query_embedding)
print(res.shape)



print(f"--- {time.time() - start_time} seconds ---")