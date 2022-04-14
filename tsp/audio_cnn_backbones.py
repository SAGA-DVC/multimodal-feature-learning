import torch
import torch.nn as nn

def vggish(pretrained=True, progress=False, **kwargs):
    return torch.hub.load('harritaylor/torchvggish', model='vggish', 
        pretrained=pretrained, progress=progress, preprocess=False, **kwargs)


class PermuteAudioChannel(nn.Module):
 
    def forward(self, x):
        x = x.unsqueeze(1)
        return x
