import torch

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