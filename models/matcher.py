# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from torch import nn
from scipy.optimize import linear_sum_assignment

def segment_cl_to_xy(x):
    c, l = x.unbind(-1)
    s = [c - 0.5 * l, c + 0.5 * l]
    return torch.stack(s, dim=-1) # (N, 2)


def segment_xy_to_cl(x):
    x0, x1 = x.unbind(-1)
    s = [(x0 + x1) / 2, (x1 - x0)]
    return torch.stack(s, dim=-1) # (N, 2)


# modified from torchvision to also return the union
def box_iou(segment1, segment2):
    area1 = segment1[:, 1] - segment1[:, 0]
    area2 = segment2[:, 1] - segment2[:, 0]

    lt = torch.max(segment1[:, None, 0], segment2[:, 0])  # (N,M)
    rb = torch.min(segment1[:, None, 1], segment2[:, 1])  # (N,M)

    inter = (rb - lt).clamp(min=0)  # (N,M)

    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-5)

    return iou, union # (N,M)


def generalized_box_iou(segment1, segment2):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(segment1)
    and M = len(segment2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (segment1[:, 1] >= segment1[:, 0]).all(), "Segment start > Segment end (from output)"
    assert (segment2[:, 1] >= segment2[:, 0]).all(), "Segment start > Segment end (from target)"

    iou, union = box_iou(segment1, segment2)

    lt = torch.min(segment1[:, None, 0], segment2[:, 0]) # (N,M)
    rb = torch.max(segment1[:, None, 1], segment2[:, 1]) # (N,M)

    area = (rb - lt).clamp(min=0)  # (N,M)

    giou = iou - (area - union) / (area + 1e-5)

    return giou # (N,M)


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class=1., cost_segment=1., cost_giou=1., cost_alpha=0.25, cost_gamma=2):
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

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim (batch_size, num_queries, num_classes) with the classification logits
                 "pred_event_segments": Tensor of dim (batch_size, num_queries, 4) with the predicted event segments

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim (num_target_segments) (where num_target_segments is the number of ground-truth
                           objects in the target) containing the class labels
                 "segments": Tensor of dim (num_target_segments, 4) containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_segments)
        """

        batch_size, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # (batch_size * num_queries, num_classes)
        out_segments = outputs["pred_event_segments"].flatten(0, 1)  # (batch_size * num_queries, 2)

        # Also concat the target labels and segments
        tgt_ids = torch.cat([v["labels"] for v in targets]) # (total_num_segments_in_target_batch)
        tgt_segments = torch.cat([v["segments"] for v in targets]) # (total_num_segments_in_target_batch, 2)

        # Compute the classification cost.
        alpha = self.cost_alpha
        gamma = self.cost_gamma
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids] # (batch_size * num_queries, total_num_segments_in_target_batch)

        # Compute the L1 cost between segments
        cost_segment = torch.cdist(out_segments, tgt_segments, p=1) # (batch_size * num_queries, total_num_segments_in_target_batch)

        # Compute the giou cost betwen segments
        cost_giou = -generalized_box_iou(segment_cl_to_xy(out_segments), segment_cl_to_xy(tgt_segments)) # (batch_size * num_queries, total_num_segments_in_target_batch)

        # Final cost matrix
        cost_matrix = self.cost_segment * cost_segment + self.cost_class * cost_class + self.cost_giou * cost_giou

        # Why CPU???
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        sizes = [len(v["segments"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, 
                            cost_segment=args.set_cost_segment, 
                            cost_giou=args.set_cost_giou,
                            cost_alpha = args.cost_alpha,
                            cost_gamma = args.cost_gamma)