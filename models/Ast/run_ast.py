import torch
import numpy as np
import timm
import time
from ast import AudioSpectrogramTransformer
from kinetics import get_kinetics
from config import load_config


cfg = load_config()

model_name = cfg.pretrained_models.for_ast.base384
model_official = timm.create_model(model_name, pretrained=cfg.ast.imagenet_pretrain)
model_official.eval()


model_custom = AudioSpectrogramTransformer(**cfg.ast, model_official=model_official)
model_custom.eval()


dataset, loader = get_kinetics(**cfg.dataset.kinetics)

start_time = time.time()
test_input = torch.rand([10, cfg.ast.input_tdim, 128])
test_output = model_custom(test_input)

# output should be in shape [10, 527], i.e., 10 samples, each with prediction of 527 classes.
print(test_output.shape)
print(f"--- {time.time() - start_time} seconds ---")