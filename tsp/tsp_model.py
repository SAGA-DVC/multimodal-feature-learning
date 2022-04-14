'''
Code adapted from https://github.com/HumamAlwassel/TSP
Alwassel, H., Giancola, S., & Ghanem, B. (2021). TSP: Temporally-Sensitive Pretraining of Video Encoders for Localization Tasks. Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops.
'''

from functools import reduce

import torch
import torch.nn as nn

class TSPModel(nn.Module):
    '''
    Model for Temporally Sensitive Pretraining of backbone architectures. Backbone models are passed in as a list.
    If `num_tsp_heads` is `None` or 0, then combined features from backbone(s) are returned, irrespective of `return_features` parameter of the forward method
    If `num_tsp_heads` is 1, then only the action classification head is created.
    If `num_tsp_heads` is 2, then the action classification head as well as the temporal region classification head is created.
    For 1 <= `num_tsp_heads` <= 2, the parameter `return_features` of the `forward` method determines whether features are returned along with logits or not
    '''

    def __init__(self, backbones, input_modalities, d_feats, d_tsp_feat,  num_tsp_classes, num_tsp_heads=1, concat_gvf=False, combiner=None):
        '''
        Args:
            backbone (List[torch.nn.Module]): One or more backbone architectures
            d_feats (List[int]): The dimension of features extracted by backbones
            d_tsp_feat (int): The dimension for input to TSP layers (output dimension of combiner function)
            num_tsp_heads (int or None): The number of output heads. If `None` or 0, then combined features from backbones are returned. Must be <= 2
            num_tsp_classes (list of int): The number of labels per head
            concat_gvf (bool): If True and num_heads == 2, then concat global video features (GVF) to clip
                features before applying the second head FC layer.
            combiner (function or torch.nn.Module): function or network that combines features extracted by backbones.
                If not provided, the addition function will be used. The function should take as inputs the features of
                individual backbones and give as output features of dimension `d_tsp_feat`.
            input_modalities (List[str]): list of keys for accessing features of modalities for each backbone
                Example ['video', 'audio']. x['video'] will be given to the first backbone, x['audio'] will be given to 
                the second backbone
        '''
        super().__init__()

        assert len(num_tsp_classes) == num_tsp_heads, f'<TSPModel>: incompatible configuration. len(num_classes) must be equal to num_heads'
        assert (num_tsp_heads is None) or (num_tsp_heads >= 0 and num_tsp_heads <= 2), f'<TSPModel>: num_tsp_heads = {num_tsp_heads} must be either None or 0 <= num_heads <= 2'
        assert isinstance(backbones, list), "<TSPModel>: backbones must be a list of models"
        assert len(backbones) > 0, "<TSPModel>: At least one backbone is required"

        self.backbones = backbones
        self.input_modalities = input_modalities
        self.d_tsp_feat = d_tsp_feat

        # Combiner function for the backbones' representations
        # Default combiner is addition
        self.combiner = combiner if combiner is not None else add_combiner

        if num_tsp_heads:
            self.num_classes = num_tsp_classes
            self.num_heads = num_tsp_heads
            self.concat_gvf = concat_gvf

            if self.num_heads == 1:
                # Linear layer for multiclass classification (action recognition)
                self.action_fc = TSPModel._build_fc(self.d_tsp_feat, num_tsp_classes[0])
            elif self.num_heads == 2:
                # Linear layer for multiclass classification (action recognition)
                self.action_fc = TSPModel._build_fc(self.d_tsp_feat, num_tsp_classes[0])

                # Linear layer for binary classification (temporal region classification: foreground / background)
                self.region_fc = TSPModel._build_fc(2 * self.d_tsp_feat if self.concat_gvf else self.d_tsp_feat, num_tsp_classes[1])
            else:
                raise NotImplementedError

        else:
            self.num_heads = None


    def forward(self, x, gvf=None, return_features=False):
        features = []  # List of features extracted by individual backbones
        for (backbone, modality) in zip(self.backbones, self.input_modalities):
            features.append(backbone(x[modality]))

        # Combine features from all backbones
        features = self.combiner(*features)

        if self.num_heads is None:
            return features

        if self.num_heads == 1:
            logits = [self.action_fc(features)]
            return (logits, features) if return_features else logits

        elif self.num_heads == 2:
            logits = [self.action_fc(features)]
            if self.concat_gvf:
                assert gvf is not None, "Forward pass expects a global video feature input but got None"
                logits.append(self.region_fc(torch.cat([features, gvf], dim=-1)))
            else:
                logits.append(self.region_fc(features))

            return (logits, features) if return_features else logits
        
        else:
            raise NotImplementedError


    @staticmethod
    def _build_fc(in_features, out_features):
        fc = nn.Linear(in_features, out_features)
        nn.init.normal_(fc.weight, 0, 0.01)
        nn.init.constant_(fc.bias, 0)
        return fc


def add_combiner(*features):
    return reduce(lambda t1, t2: t1 + t2, features)


def concat_combiner(*features):
    return torch.cat(features, dim=1)