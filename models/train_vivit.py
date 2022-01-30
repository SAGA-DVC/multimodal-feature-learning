import sys
sys.path.insert(0, '../dataset')
sys.path.insert(1, '../config')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import timm
import time
from vivit import VideoVisionTransformer
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


model_custom = VideoVisionTransformer(**cfg.vivit, model_official=model_official)

# for (name_custom, parameter_custom) in model_custom.named_parameters():
#     print(f"{name_custom}, {parameter_custom.shape}")

# print('-----------------------')

# for (name_official, parameter_official) in model_official.named_parameters():
#     print(f"{name_official} , {parameter_official.shape}")


dataset, loader = get_kinetics(**cfg.dataset.kinetics)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_custom.parameters(), lr=0.001, momentum=0.9)
epochs = 1



for epoch in range(epochs):
    for i, batch in enumerate(iter(loader)):

        input, labels = batch['video'], batch['label']

        optimizer.zero_grad()

        res = model_custom(input)
        loss = criterion(res, labels)
        loss.backward()
        optimizer.step()

        if i % 5 == 0:    # print every 5 batches
            print(f'epoch: {epoch + 1}, batch: {i + 1}, loss: {loss.item() / 2000}')

print('Finished Training')



# PATH = './vivit.pth'
# torch.save(model_custom.state_dict(), PATH)