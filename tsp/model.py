'''
Code modified from https://github.com/HumamAlwassel/TSP
Alwassel, H., Giancola, S., & Ghanem, B. (2021). TSP: Temporally-Sensitive Pretraining of Video Encoders for Localization Tasks. Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops.
'''



import torch
import torch.nn as nn

class TSPModel(nn.Module):

    def __init__(self, backbones, d_feats, d_tsp_feat,  num_tsp_classes, num_tsp_heads=1, concat_gvf=False, combiner=None):
        '''
        Args:
            backbone (List[torch.nn.Module]): One or more backbone architectures
            d_feats (List[int]): The dimension of features extracted by backbones
            d_tsp_feat (int): The dimension for input to TSP layers (output dimension of combiner function)
            num_tsp_heads (int): The number of output heads
            num_tsp_classes (list of int): The number of labels per head
            concat_gvf (bool): If True and num_heads == 2, then concat global video features (GVF) to clip
                features before applying the second head FC layer.
            combiner (function or torch.nn.Module): function or network that combines features extracted by backbones.
                If not provided, the identity function will be used.
        '''
        super().__init__()
        # print(f'<TSPModel>: backbone {backbone} num_classes {num_tsp_classes} num_heads {num_tsp_heads} kwargs {kwargs}')
        assert len(num_tsp_classes) == num_tsp_heads, f'<TSPModel>: incompatible configuration. len(num_classes) must be equal to num_heads'
        assert num_tsp_heads == 1 or num_tsp_heads == 2, f'<TSPModel>: num_heads = {num_tsp_heads} must be either 1 or 2'
        assert isinstance(backbones, list), "<TSPModel>: backbones must be a list of models"
        assert len(backbones) > 0, "<TSPModel>: At least one backbone is required"

        self.backbones = backbones
        self.combiner = combiner if combiner is not None else nn.Identity()
        self.num_classes = num_tsp_classes
        self.num_heads = num_tsp_heads
        self.concat_gvf = concat_gvf

        self.d_tsp_feat = d_tsp_feat

        if self.num_heads == 1:
            # Linear layer for multiclass classification (action recognition)
            self.action_fc = TSPModel._build_fc(self.d_tsp_feat, num_tsp_classes[0])
        else:
            # Linear layer for multiclass classification (action recognition)
            self.action_fc = TSPModel._build_fc(self.d_tsp_feat, num_tsp_classes[0])

            # Linear layer for binary classification (temporal region classification: foreground / background)
            self.region_fc = TSPModel._build_fc(2 * self.d_tsp_feat if self.concat_gvf else self.d_tsp_feat, num_tsp_classes[1])


    def forward(self, x, gvf=None, return_features=False):
        # Extract features using the backbones
        features = []  # List of features extracted by individual backbones
        for backbone in self.backbones:
            features.append(backbone(x))

        features = self.combiner(torch.cat(features))

        if self.num_heads == 1:
            logits = [self.action_fc(features)]
        else:
            logits = [self.action_fc(features)]
            if self.concat_gvf:
                assert gvf is not None, "Forward pass expects a global video feature input but got None"
                logits.append(self.region_fc(torch.cat([features, gvf], dim=-1)))
            else:
                logits.append(self.region_fc(features))
        return (logits, features) if return_features else logits


    @staticmethod
    def _build_fc(in_features, out_features):
        fc = nn.Linear(in_features, out_features)
        nn.init.normal(fc.weight, 0, 0.01)
        nn.init.constant_(fc.bias, 0)
        return fc
