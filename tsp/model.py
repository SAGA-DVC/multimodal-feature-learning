'''
Code modified from https://github.com/HumamAlwassel/TSP
Alwassel, H., Giancola, S., & Ghanem, B. (2021). TSP: Temporally-Sensitive Pretraining of Video Encoders for Localization Tasks. Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops.
'''



import torch
import torch.nn as nn

class TSPModel(nn.Module):

    def __init__(self, backbone, d_feat,  num_tsp_classes, num_tsp_heads=1, concat_gvf=False, **kwargs):
        '''
        Args:
            backbone (torch.nn.module): The backbone architecture
            d_feat (int): The dimension of features extracted by model
            num_tsp_heads (int): The number of output heads
            num_tsp_classes (list of int): The number of labels per head
            concat_gvf (bool): If True and num_heads == 2, then concat global video features (GVF) to clip
                features before applying the second head FC layer.
            **kwargs: keyword arguments to pass to backbone architecture constructor
        '''
        super().__init__()
        # print(f'<TSPModel>: backbone {backbone} num_classes {num_tsp_classes} num_heads {num_tsp_heads} kwargs {kwargs}')
        assert len(num_tsp_classes) == num_tsp_heads, f'<TSPModel>: incompatible configuration. len(num_classes) must be equal to num_heads'
        assert num_tsp_heads == 1 or num_tsp_heads == 2, f'<TSPModel>: num_heads = {num_tsp_heads} must be either 1 or 2'

        self.backbone = backbone
        self.num_classes = num_tsp_classes
        self.num_heads = num_tsp_heads
        self.concat_gvf = concat_gvf

        self.feature_extractor = backbone
        self.feature_size = d_feat
        

        if self.num_heads == 1:
            # Linear layer for multiclass classification (action recognition)
            self.action_fc = TSPModel._build_fc(self.feature_size, num_tsp_classes[0])
        else:
            # Linear layer for multiclass classification (action recognition)
            self.action_fc = TSPModel._build_fc(self.feature_size, num_tsp_classes[0])

            # Linear layer for binary classification (temporal region classification: foreground / background)
            self.region_fc = TSPModel._build_fc(2 * self.feature_size if self.concat_gvf else self.feature_size, num_tsp_classes[1])


    def forward(self, x, gvf=None, return_features=False):
        # Extract features using the backbone
        features = self.feature_extractor(x)
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
