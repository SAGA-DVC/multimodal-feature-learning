# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""

import torch
from torch import nn
from scipy.optimize import linear_sum_assignment

from utils.box_ops import segment_cl_to_xy, generalized_box_iou 

# TODO
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class=1., cost_segment=1., cost_giou=1., cost_alpha=0.25, cost_gamma=2.0):
        """
        Creates the bipartite matcher using the Hungarian Algorithm
        Parameters:
            `cost_class` (float) : This is the relative weight of the classification error in the matching cost (default=1)
            `cost_segment` (float) : This is the relative weight of the L1 error of the bounding box coordinates in the matching cost (default=1)
            `cost_giou` (float) : This is the relative weight of the giou loss of the bounding box in the matching cost (default=1)
            `cost_alpha` (float) :
            `cost_gamma` (float) :
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_segment = cost_segment
        self.cost_giou = cost_giou
        self.cost_alpha = cost_alpha
        self.cost_gamma = cost_gamma

        assert cost_class != 0 or cost_segment != 0 or cost_giou != 0, "Costs cant be 0."

    # TODO - check CPU in forward
    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim (batch_size, num_queries, num_classes) with the classification logits
                 "pred_segments": Tensor of dim (batch_size, num_queries, 2) with the predicted segments

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim (num_target_segments) (where num_target_segments is the number of ground-truth
                           objects in the target) containing the class labels
                 "segments": Tensor of dim (num_target_segments, 2) containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of tensors (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_segments)
        """

        batch_size, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # (batch_size * num_queries, num_classes)
        out_segments = outputs["pred_segments"].flatten(0, 1)  # (batch_size * num_queries, 2)

        # Also concat the target labels and segments
        tgt_ids = torch.cat([v["labels"] for v in targets]) # (nb_target_segments)
        tgt_segments = torch.cat([v["segments"] for v in targets]) # (nb_target_segments, 2)

        # Compute the classification cost.
        alpha = self.cost_alpha
        gamma = self.cost_gamma

        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids] # (batch_size * num_queries, nb_target_segments)

        # Compute the L1 cost between segments
        cost_segment = torch.cdist(out_segments, tgt_segments, p=1) # (batch_size * num_queries, nb_target_segments)

        # Compute the giou cost betwen segments
        cost_giou = -generalized_box_iou(segment_cl_to_xy(out_segments), segment_cl_to_xy(tgt_segments)) # (batch_size * num_queries, nb_target_segments)

        # Final cost matrix
        cost_matrix = self.cost_segment * cost_segment + self.cost_class * cost_class + self.cost_giou * cost_giou # (batch_size * num_queries, nb_target_segments)

        # Why CPU???
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu() # (batch_size, num_queries, nb_target_segments)

        sizes = [len(v["segments"]) for v in targets] # (batch_size)
        
        # c[i]: (batch_size, num_queries, nb_target_segments) -> (batch_size, num_queries, num_target_segments) per ground truth event
        # (batch_size, num_queries, gt_target_segments) -> list (len=batch_size) of tuple of lists (shape=(2, gt_target_segments))
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.cost_class, 
                            cost_segment=args.cost_segment, 
                            cost_giou=args.cost_giou,
                            cost_alpha = args.cost_alpha,
                            cost_gamma = args.cost_gamma)