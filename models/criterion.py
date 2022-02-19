import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.box_ops import segment_cl_to_xy, segment_xy_to_cl, generalized_box_iou, box_iou 
from utils.misc import accuracy, is_dist_avail_and_initialized, get_world_size

from dvc import DVC
from matcher import build_matcher


class SetCriterion(nn.Module):

    """ 
    This class computes the loss for the entire DVC model.The process happens in two steps:
        1) we compute hungarian assignment between ground truth segments and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and segments)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, focal_alpha=0.25, focal_gamma=2, opt={}):

        """ 
        Create the criterion.
        Parameters:
            `num_classes` : number of object categories, omitting the special no-object category
            `matcher` : module able to compute a matching between targets and proposals
            `weight_dict` : dict containing as key the names of the losses and as values their relative weight.
            `eos_coef` : relative classification weight applied to the no-object category
            `losses` : list of all the losses to be applied. See get_loss for list of available losses.
            `focal_alpha` : alpha in Focal Loss (default 0.25)
            `focal_gamma` : alpha in Focal Loss (default 2)
            `opt` : 
        """

        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses

        self.eos_coef = eos_coef
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma


    def loss_labels(self, outputs, targets, indices, num_segments, log=True):

        """
        Classification loss (Negative Log Likelihood)
            targets dicts must contain the key "labels" containing a tensor of dim [nb_target_segments]
        """
        
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'] # (batch_size, num_queries, num_classes {+ 1??})

        # (nb_target_segments) contains batch numbers AND (nb_target_segments) contains target indices of matcher
        # eg. [0, 0, 0,   1, 1] AND [2, 0, 1,   1, 0] 
        idx = self._get_src_permutation_idx(indices) 

        # (nb_target_segments) contains class labels
        # eg. [6, 9, 25,   4, 7] (based on above eg - each index represents a class in its batch)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        # (batch_size, num_queries)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        
        target_classes[idx] = target_classes_o

        # used in detr
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        # # used in pdvc
        # # (batch_size, num_queries, num_classes {+ 1??})
        # target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
        #                                     dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        
        # # 1 for positive class, 0 for negative class
        # target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        # target_classes_onehot = target_classes_onehot[:,:,:-1]

        # loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_segments, alpha=self.focal_alpha, gamma=self.focal_gamma) * src_logits.shape[1]
        
        # losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses


    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_segments):

        """ 
        Compute the cardinality error, ie the absolute error in the number of predicted non-empty segments
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """

        pred_logits = outputs['pred_logits']
        device = pred_logits.device

        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device) # (batch_size)

        # Count the number of predictions that are NOT "no-object" (which is the last class)
        # (batch_size, num_queries) --(sum)->  (batch_size)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1) 

        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())

        losses = {'cardinality_error': card_err}
        return losses


    def loss_segments(self, outputs, targets, indices, num_segments):

        """
        Compute the losses related to the bounding segments, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "segments" containing a tensor of dim [nb_target_segments, 2]
           The target segments are expected in format (center_x, center_y, w, h), normalized by the image size.
        """

        assert 'pred_segments' in outputs

        # (nb_target_segments) contains batch numbers AND (nb_target_segments) contains target indices of matcher
        # eg. [0, 0, 0,   1, 1] AND [2, 0, 1,   1, 0] 
        idx = self._get_src_permutation_idx(indices)

        # (nb_target_segments, 2)
        src_segments = outputs['pred_segments'][idx]

        # (nb_target_segments, 2) contains segment coordinates
        # eg. [6, 9, 25,   4, 7] (based on above eg - each index represents a class in its batch)
        target_segments = torch.cat([t['segments'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_segments, target_segments, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_segments

        # (nb_target_segments)
        loss_giou = 1 - torch.diag(generalized_box_iou(
            segment_cl_to_xy(src_segments),
            segment_cl_to_xy(target_segments)))

        losses['loss_giou'] = loss_giou.sum() / num_segments

        # # used in pdvc
        # # (nb_target_segments, nb_target_segments)
        # self_iou = torch.triu(box_iou(
        #     segment_cl_to_xy(src_segments),
        #     segment_cl_to_xy(src_segments))[0], diagonal=1)

        # # [nb_target_segments]
        # sizes = [len(v[0]) for v in indices]
        # self_iou_split = 0

        # for i, c in enumerate(self_iou.split(sizes, -1)):
        #     cc = c.split(sizes, -2)[i] # (num_segments) --varies per batch
        #     self_iou_split += cc.sum() / (0.5 * (sizes[i]) * (sizes[i]-1))

        # losses['loss_self_iou'] = self_iou_split

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_segments, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'segments': self.loss_segments
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_segments, **kwargs)

    def forward(self, outputs, targets):

        """ 
        This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target segments accross all nodes, for normalization purposes
        num_segments = sum(len(t["labels"]) for t in targets)
        num_segments = torch.as_tensor([num_segments], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_segments)
        num_segments = torch.clamp(num_segments / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_segments))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_segments, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses



def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """

    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes