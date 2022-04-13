import torch

def vggish(pretrained=True, progress=False, **kwargs):
    return torch.hub.load('harritaylor/torchvggish', model='vggish', 
        pretrained=pretrained, progress=progress, **kwargs)
