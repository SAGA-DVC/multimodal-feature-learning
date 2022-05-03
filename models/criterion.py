import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.box_ops import segment_cl_to_xy, segment_xy_to_cl, generalized_box_iou, box_iou 
from utils.misc import accuracy, is_dist_avail_and_initialized, get_world_size
from utils.dam import idx_to_flat_grid, attn_map_to_flat_grid, compute_corr

from .matcher import build_matcher


class SetCriterion(nn.Module):

    """ 
    This class computes the loss for the entire DVC model.The process happens in two steps:
        1) we compute hungarian assignment between ground truth segments and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and segments)
    """
    # TODO - check __init__() attributes
    def __init__(self, is_multimodal, num_classes, matcher, weight_dict, eos_coef, losses, pad_idx, smoothing=0.7, 
                focal_alpha=0.25, focal_gamma=2, lloss_gau_mask=1, lloss_beta=1.):

        """ 
        Create the criterion.
        Parameters:
            `num_classes` : number of object categories, omitting the special no-action category
            `matcher` : module to compute a bipartite matching between targets and proposals
            `weight_dict` : dict containing as key the names of the losses and as values their relative weight.
            `eos_coef` : relative classification weight applied to the no-action category
            `losses` : list of all the losses to be applied. See get_loss for list of available losses.
            `pad_idx` (int): index of the padding token '<pad>' in the vocabulary
            `smoothing` (float): smoothing coefficient (epsilon) for the LabelSmoother (default 0.7)
            `focal_alpha` : alpha in Focal Loss (default 0.25)
            `focal_gamma` : alpha in Focal Loss (default 2)
            
        """

        super().__init__()
        self.is_multimodal = is_multimodal
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

        self.lloss_gau_mask = lloss_gau_mask
        self.lloss_beta = lloss_beta

        self.pad_idx = pad_idx

        self.labelSmoothing = LabelSmoothing(smoothing, self.pad_idx)

        counter_class_rate = [0.00000000e+00, 0.00000000e+00, 1.93425917e-01, 4.12129084e-01,
                            1.88929963e-01, 7.81296833e-02, 5.09541413e-02, 3.12718553e-02,
                            1.84833650e-02, 8.39244680e-03, 6.59406534e-03, 4.49595364e-03,
                            2.19802178e-03, 1.79838146e-03, 5.99460486e-04, 4.99550405e-04,
                            4.99550405e-04, 1.99820162e-04, 2.99730243e-04, 3.99640324e-04,
                            2.99730243e-04, 0.00000000e+00, 1.99820162e-04, 0.00000000e+00,
                            0.00000000e+00, 0.00000000e+00, 9.99100809e-05, 9.99100809e-05]
        self.counter_class_rate = torch.tensor(counter_class_rate)



    def loss_labels(self, outputs, targets, indices, num_segments, num_tokens_without_pad, memory_mask, log=True):

        """
        Classification loss (Negative Log Likelihood)
            targets dicts must contain the key "labels" containing a tensor of dim [nb_target_segments]
        
        Parameters:
            `outputs` (dict) : Output of the model. See forward() for the format.
            `targets` (list) : Ground truth targets of the dataset. See forward() for the format.
            `indices` (list) : Bipartite matching of the output and target segments. list (len=batch_size) of tuple of tensors (shape=(2, gt_target_segments)).
            `num_segments` (int) : Average number of target segments accross all nodes, for normalization purposes.
            `num_tokens_without_pad` (int): Number of tokens in the caption excluding the '<pad>' token, for normalization purposes
            `memory_mask`(tensor: int): 0 if num_token useless, else 1 (nb_target_segments, num_tokens)
            `log` (boolean) : If True, 'class_error' is also calculated and returned.
        
        Returns: dict {loss : value} where loss can be 'labels' and/or 'class_error'.
        """
        
        assert 'pred_logits' in outputs, "Outputs does not have the key 'pred_logits'."

        src_logits = outputs['pred_logits'] # (batch_size, num_queries, num_classes + 1)

        # batch_idx - tensor (nb_target_segments) contains batch numbers AND 
        # src_idx - tensor (nb_target_segments) contains source indices of bipartite matcher
        # eg. [0, 0, 0,   1, 1] AND [2, 14, 88,   3, 91] 
        idx = self._get_src_permutation_idx(indices) 

        # tensor (nb_target_segments) contains class labels
        # eg. [6, 9, 25,   4, 7] (each index represents a class in its batch)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets['video_target'], indices)])

        # (batch_size, num_queries) where all elements have a value of self.num_classes ('no-action' has an index of self.num_classes)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        
        # (batch_size, num_queries) where class labels are assigned based on batch_idx and src_idx. Other elements have a value of self.num_classes
        target_classes[idx] = target_classes_o

        # used in detr
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        # used in pdvc
        # # (batch_size, num_queries, num_classes + 1)
        # target_classes_onehot = torch.zeros(src_logits.shape, dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        
        # # 1 for positive class, 0 for negative class
        # target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        # target_classes_onehot = target_classes_onehot[:,:,:-1]

        # loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_segments, alpha=self.focal_alpha, gamma=self.focal_gamma) * src_logits.shape[1]
        
        # losses = {'loss_ce': loss_ce}

        # used in pdvc (event counter)
        pred_count = outputs['pred_count']
        max_length = pred_count.shape[1] - 1
        counter_target = [len(target['segments']) if len(target['segments']) < max_length  else max_length for target in targets['video_target']]
        counter_target = torch.tensor(counter_target, device=src_logits.device, dtype=torch.long)
        counter_target_onehot = torch.zeros_like(pred_count)
        counter_target_onehot.scatter_(1, counter_target.unsqueeze(-1), 1)
        weight = self.counter_class_rate[:max_length + 1].to(src_logits.device)

        counter_loss = cross_entropy_with_gaussian_mask(pred_count, counter_target_onehot, weight, self.lloss_gau_mask, self.lloss_beta)
        losses['loss_counter'] = counter_loss

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            # only takes top-1 accuracy for now
            # losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]    # takes into account 'no-action' class 
            losses['class_error'] = 100 - accuracy(src_logits[idx][..., 1:], target_classes_o)[0]    # ignores 'no-action' class 
        
        return losses


    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_segments, num_tokens_without_pad, memory_mask):

        """ 
        Compute the cardinality error, ie the absolute error in the number of predicted non-empty segments
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients.

        Parameters:
            `outputs` (dict) : Output of the model. See forward() for the format.
            `targets` (list) : Ground truth targets of the dataset. See forward() for the format.
            `indices` (list) : Bipartite matching of the output and target segments. list (len=batch_size) of tuple of tensors (shape=(2, gt_target_segments)).
            `num_segments` (int) : Average number of target segments accross all nodes, for normalization purposes.
            `num_tokens_without_pad` (int): Number of tokens in the caption excluding the '<pad>' token, for normalization purposes
            `memory_mask`(tensor: int): 0 if num_token useless, else 1 (nb_target_segments, num_tokens)
        
        Returns: dict {loss : value} where loss is 'cardinality_error'.
        """

        assert 'pred_logits' in outputs, "Outputs does not have the key 'pred_logits'."

        pred_logits = outputs['pred_logits'] # (batch_size, num_queries, num_classes + 1)
        device = pred_logits.device

        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets['video_target']], device=device) # (batch_size)

        # Count the number of predictions that are NOT "no-action" (which is the last class)
        # (batch_size, num_queries) --(sum)->  (batch_size)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)

        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())

        losses = {'cardinality_error': card_err}
        return losses


    def loss_segments(self, outputs, targets, indices, num_segments, num_tokens_without_pad, memory_mask):

        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.
        targets dicts must contain the key "segments" containing a tensor of dim (nb_target_segments, 2)
        The target segments are expected in format (center_offset, length), normalized by the video duration.
        
        Parameters:
            `outputs` (dict) : Output of the model. See forward() for the format.
            `targets` (list) : Ground truth targets of the dataset. See forward() for the format.
            `indices` (list) : Bipartite matching of the output and target segments. list (len=batch_size) of tuple of tensors (shape=(2, gt_target_segments)).
            `num_segments` (int) : Average number of target segments accross all nodes, for normalization purposes.
            `num_tokens_without_pad` (int): Number of tokens in the caption excluding the '<pad>' token, for normalization purposes
            `memory_mask`(tensor: int): 0 if num_token useless, else 1 (nb_target_segments, num_tokens)
        
        Returns: dict {loss : value} where loss can be 'loss_bbox' or 'loss_giou'.
        """

        assert 'pred_segments' in outputs, "Outputs does not have the key 'pred_segments'."

        # batch_idx - tensor (nb_target_segments) contains batch numbers AND 
        # src_idx - tensor (nb_target_segments) contains source indices of bipartite matcher
        # eg. [0, 0, 0,   1, 1] AND [2, 14, 88,   3, 91] 
        idx = self._get_src_permutation_idx(indices)

        # (nb_target_segments, 2)
        src_segments = outputs['pred_segments'][idx]

        # (nb_target_segments, 2) contains segment coordinates
        target_segments = torch.cat([t['segments'][i] for t, (_, i) in zip(targets['video_target'], indices)], dim=0)

        # (nb_target_segments)
        loss_bbox = F.l1_loss(src_segments, target_segments, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_segments

        # (nb_target_segments, nb_target_segments) -> (nb_target_segments)
        loss_giou = 1 - torch.diag(generalized_box_iou(segment_cl_to_xy(src_segments), segment_cl_to_xy(target_segments)))

        losses['loss_giou'] = loss_giou.sum() / num_segments

        # used in pdvc
        # (nb_target_segments, nb_target_segments)
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

    
    def loss_mask_prediction(self, outputs, targets, indices, num_segments, num_tokens_without_pad, memory_mask):
        assert "backbone_mask_prediction" in outputs
        assert "sampling_locations_dec" in outputs
        assert "attn_weights_dec" in outputs
        assert "temporal_shapes" in outputs
        assert "level_start_index" in outputs

        mask_prediction = outputs["backbone_mask_prediction"]
        loss_key = "loss_mask_prediction"

        sampling_locations_dec = outputs["sampling_locations_dec"]
        attn_weights_dec = outputs["attn_weights_dec"]
        temporal_shapes = outputs["temporal_shapes"]
        level_start_index = outputs["level_start_index"]

        flat_grid_attn_map_dec = attn_map_to_flat_grid(
            temporal_shapes, level_start_index, sampling_locations_dec, attn_weights_dec).sum(dim=(1,2))

        losses = {}

        if 'mask_flatten' in outputs:
            flat_grid_attn_map_dec = flat_grid_attn_map_dec.masked_fill(
                outputs['mask_flatten'], flat_grid_attn_map_dec.min()-1)
                
        sparse_token_nums = outputs["sparse_token_nums"]
        num_topk = sparse_token_nums.max()

        topk_idx_tgt = torch.topk(flat_grid_attn_map_dec, num_topk)[1]
        target = torch.zeros_like(mask_prediction)
        for i in range(target.shape[0]):
            target[i].scatter_(0, topk_idx_tgt[i][:sparse_token_nums[i]], 1)

        losses.update({loss_key: F.multilabel_soft_margin_loss(mask_prediction, target)})

        return losses


    @torch.no_grad()
    def corr(self, outputs, targets, indices, num_segments, num_tokens_without_pad, memory_mask):
        if "backbone_topk_proposals" not in outputs.keys():
            return {}

        assert "backbone_topk_proposals" in outputs
        assert "sampling_locations_dec" in outputs
        assert "attn_weights_dec" in outputs
        assert "temporal_shapes" in outputs
        assert "level_start_index" in outputs

        backbone_topk_proposals = outputs["backbone_topk_proposals"]
        sampling_locations_dec = outputs["sampling_locations_dec"]
        attn_weights_dec = outputs["attn_weights_dec"]
        temporal_shapes = outputs["temporal_shapes"]
        level_start_index = outputs["level_start_index"]

        flat_grid_topk = idx_to_flat_grid(temporal_shapes, backbone_topk_proposals)
        flat_grid_attn_map_dec = attn_map_to_flat_grid(
            temporal_shapes, level_start_index, sampling_locations_dec, attn_weights_dec).sum(dim=(1,2))
        corr = compute_corr(flat_grid_topk, flat_grid_attn_map_dec, temporal_shapes)

        losses = {}
        losses["loss_corr"] = corr[0].mean()
        # for i, _corr in enumerate(corr[1:]):
        #     losses[f"loss_corr_{i}"] = _corr.mean()
        return losses


    def loss_captions(self, outputs, targets, indices, num_segments, num_tokens_without_pad, memory_mask):

        """
        Compute the losses related to the captions, KL Divergence using Label Smoothing.
        targets dicts must contain the key "cap_tensor" containing a tensor of dim (total_caption_num, max_caption_length)
        
        Parameters:
            `outputs` (dict) : Output of the model. See forward() for the format.
            `targets` (list) : Ground truth targets of the dataset. See forward() for the format.
            `indices` (list) : Bipartite matching of the output and target segments. list (len=batch_size) of tuple of tensors (shape=(2, gt_target_segments)).
            `num_segments` (int) : Average number of target segments accross all nodes, for normalization purposes.
            `num_tokens_without_pad` (int): Number of tokens in the caption excluding the '<pad>' token, for normalization purposes
            `memory_mask`(tensor: int): 0 if num_token useless, else 1 (nb_target_segments, num_tokens)
        
        Returns: dict {loss : value} where loss is 'loss_caption'.
        """

        assert 'pred_captions' in outputs, "Outputs does not have the key 'pred_captions'."

        losses = {}
        loss_caption = self.labelSmoothing(outputs['pred_captions'], targets['cap_tensor'][:, 1:])
        losses['loss_caption'] = loss_caption / num_tokens_without_pad
        return losses

    # TODO - focal loss and dice loss for masks
    def unimodal_loss_contexts(self, outputs, targets, indices, num_segments, num_tokens_without_pad, memory_mask):

        """
        Compute the losses related to the context_mask using BCE.
        targets dicts must contain the key "pred_memory_mask" containing a tensor of dim (nb_target_segements, num_tokens)
        
        Parameters:
            `outputs` (dict) : Output of the model. See forward() for the format.
            `targets` (list) : Ground truth targets of the dataset. See forward() for the format.
            `indices` (list) : Bipartite matching of the output and target segments. list (len=batch_size) of tuple of tensors (shape=(2, gt_target_segments)).
            `num_segments` (int) : Average number of target segments accross all nodes, for normalization purposes.
            `num_tokens_without_pad` (int): Number of tokens in the caption excluding the '<pad>' token, for normalization purposes
            `memory_mask`(tensor: int): 1 if num_token useless, else 0 (nb_target_segments, num_tokens)
        
        Returns: dict {loss : value} where loss is 'loss_context'.
        """

        assert 'pred_memory_mask' in outputs, "Outputs does not have the key 'pred_memory_mask'."

        losses = {}
        loss_context = F.binary_cross_entropy_with_logits(outputs['pred_memory_mask'], memory_mask)
        losses['loss_context'] = loss_context
        return losses

    # TODO - check avg or sum of video/audio contexts
    def multimodal_loss_contexts(self, outputs, targets, indices, num_segments, num_tokens_without_pad, memory_mask):

        """
        Compute the losses related to the context_mask using BCE.
        targets dicts must contain the key "pred_memory_mask" containing a tensor of dim (nb_target_segements, num_tokens)
        
        Parameters:
            `outputs` (dict) : Output of the model. See forward() for the format.
            `targets` (list) : Ground truth targets of the dataset. See forward() for the format.
            `indices` (list) : Bipartite matching of the output and target segments. list (len=batch_size) of tuple of tensors (shape=(2, gt_target_segments)).
            `num_segments` (int) : Average number of target segments accross all nodes, for normalization purposes.
            `num_tokens_without_pad` (int): Number of tokens in the caption excluding the '<pad>' token, for normalization purposes
            `memory_mask`(tensor: int): 1 if num_token useless, else 0 (nb_target_segments, num_tokens)
        
        Returns: dict {loss : value} where loss is 'loss_context'.
        """
        
        assert 'video_pred_memory_mask' in outputs, "Outputs does not have the key 'video_pred_memory_mask'."
        assert 'audio_pred_memory_mask' in outputs, "Outputs does not have the key 'audio_pred_memory_mask'."

        losses = {}
        video_memory_mask, audio_memory_mask = memory_mask

        video_loss_context = F.binary_cross_entropy_with_logits(outputs['video_pred_memory_mask'], video_memory_mask)
        audio_loss_context = F.binary_cross_entropy_with_logits(outputs['audio_pred_memory_mask'], audio_memory_mask)

        losses['loss_context'] = (video_loss_context + audio_loss_context) / 2
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


    def get_loss(self, loss, outputs, targets, indices, num_segments, num_tokens_without_pad, memory_mask, **kwargs):

        """
        Calculates a specific loss of the outputs w.r.t. the targets. 
        The losses include classification loss, cardinality loss and segment loss
        
        Parameters:
            `loss` (string): Determines the loss to be calculated. Can be one of 'labels', 'cardinality', 'segments' or 'captions.
            `outputs` (dict): Output of the model. See forward() for the format.
            `targets` (list): Ground truth targets of the dataset. See forward() for the format.
            `indices` (list): Bipartite matching of the output and target segments. list (len=batch_size) of tuple of tensors (shape=(2, gt_target_segments)).
            `num_segments` (int): Average number of target segments accross all nodes, for normalization purposes.
            `num_tokens_without_pad` (int): Number of tokens in the caption excluding the '<pad>' token, for normalization purposes
            `memory_mask`(tensor: int): 0 if num_token useless, else 1 (nb_target_segments, num_tokens)
        
        Returns: dict {loss : value} where loss is the one of 'labels', 'cardinality' or 'segments'.
        """

        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'segments': self.loss_segments,
            'captions': self.loss_captions,
            'contexts': self.multimodal_loss_contexts if self.is_multimodal else self.unimodal_loss_contexts,
            "mask_prediction": self.loss_mask_prediction,
            "corr": self.corr,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_segments, num_tokens_without_pad, memory_mask, **kwargs)

    def forward(self, outputs, targets, indices, indices_aux, memory_mask):

        """ 
        This performs the loss computation.
        Parameters:
            `outputs` (dict): dict of tensors
                    - "pred_logits": the classification logits (including no-action) for all queries
                                    shape (batch_size, num_queries, num_classes + 1)
                    - "pred_segments": The normalized segments for all queries, represented as
                                    (center_offset, length). Shape (batch_size, num_queries, 2)
                    - "pred_captions": All captions in a batch with shape (total_caption_num, seq_len, vocab_size)
                    - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                    dictionaries containing the two above keys for each decoder layer.

            `targets` (dict): check collate_fn in dataset/anet.py for description (obj)
            
            `indices` (list): matching between the outputs of the last layer and the targets
                            list (len=batch_size) of tuple of tensors (shape=(2, gt_target_segments))

            `indices_aux` (list): list of len=depth. Matching between the outputs of the each layer (except the last) 
                            and the targets list (len=batch_size) of tuple of tensors (shape=(2, gt_target_segments))

            `memory_mask`(tensor: int): 1 if num_token useless, else 0 (nb_target_segments, num_tokens)
        
        Returns:
            `losses`: dict consisting of the following items
                    - "loss_ce" (float): cross entropy loss
                    - "class_error" (float): classification error based on accuracy of ...
                    - "cardinality_error" (float): based on number of predicted non-empty segments
                    - "loss_bbox" (float): bounding box loss
                    - "loss_giou" (float): general intersection over union loss
                    - "loss_captions (float): KL Divergence loss using Label Smoothing
           
        """

        # outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        # list (len=batch_size) of tuple of tensors (shape=(2, gt_target_segments))
        # indices = self.matcher(outputs_without_aux, targets['video_target']) 

        # Average number of target segments accross all nodes, for normalization purposes
        num_segments = sum(len(t["labels"]) for t in targets['video_target'])
        num_segments = torch.as_tensor([num_segments], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_segments)
        num_segments = torch.clamp(num_segments / get_world_size(), min=1).item()

        # Number of tokens in the caption excluding the '<pad>' token, for normalization purposes
        # ignore <bos> token as it is not predicted as part of the output
        num_tokens_without_pad = (targets['cap_tensor'][:, 1:] != self.pad_idx).sum()
        num_tokens_without_pad = torch.as_tensor([num_tokens_without_pad], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_tokens_without_pad)
        num_tokens_without_pad = torch.clamp(num_tokens_without_pad / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_segments, num_tokens_without_pad, memory_mask))


        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, (aux_outputs, index_aux) in enumerate(zip(outputs['aux_outputs'], indices_aux)):
                # index_aux = self.matcher(aux_outputs, targets['video_target'])
                for loss in self.losses:
                    if loss == 'contexts' or loss == 'mask_prediction' or loss == 'corr':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer (class error)
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, index_aux, num_segments, num_tokens_without_pad, memory_mask, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        
        if 'aux_outputs_enc' in outputs:
            for i, (aux_outputs, index_aux) in enumerate(zip(outputs['aux_outputs_enc'], indices_aux)):
                # index_aux_enc = self.matcher(aux_outputs, targets['video_target'])
                for loss in self.losses:
                    if loss == 'contexts' or loss == 'mask_prediction' or loss == 'corr':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer (class error)
                        kwargs = {'log': False}
                    if loss == 'captions':
                        # there are no captions in encoder loss
                        continue
                    l_dict = self.get_loss(loss, aux_outputs, targets, index_aux, num_segments, num_tokens_without_pad, memory_mask, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)


        return losses



class LabelSmoothing(nn.Module):
    
    def __init__(self, smoothing, pad_idx):
        super(LabelSmoothing, self).__init__()
        self.smoothing = smoothing
        self.pad_idx = pad_idx
        
    def forward(self, pred, target):  # pred (B, S, V), target (B, S)
        # Note: preds are expected to be after log
        B, S, V = pred.shape
        # (B, S, V) -> (B * S, V); (B, S) -> (B * S)
        pred = pred.contiguous().view(-1, V)
        target = target.contiguous().view(-1)
        
        # prior (uniform)
        dist = self.smoothing * torch.ones_like(pred) / (V - 2)
        # add smoothed ground-truth to prior (args: dim, index, src (value))
        dist.scatter_(1, target.unsqueeze(-1).long(), 1-self.smoothing)
        # make the padding token to have zero probability
        dist[:, self.pad_idx] = 0
        # ?? mask: 1 if target == pad_idx; 0 otherwise
        mask = torch.nonzero(target == self.pad_idx)
        
        if mask.sum() > 0 and len(mask) > 0:
            # dim, index, val
            dist.index_fill_(0, mask.squeeze(), 0)
            
        return F.kl_div(pred.log(), dist, reduction='sum')





def cross_entropy_with_gaussian_mask(inputs, targets, weight, lloss_gau_mask=1, lloss_beta=1.):
    gau_mask = lloss_gau_mask
    beta = lloss_beta

    N_, max_seq_len = targets.shape
    gassian_mu = torch.arange(max_seq_len, device=inputs.device).unsqueeze(0).expand(max_seq_len,
                                                                                     max_seq_len).float()
    x = gassian_mu.transpose(0, 1)
    gassian_sigma = 2
    mask_dict = torch.exp(-(x - gassian_mu) ** 2 / (2 * gassian_sigma ** 2))
    _, ind = targets.max(dim=1)
    mask = mask_dict[ind]

    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none", weight= 1 - weight)
    if gau_mask:
        coef = targets + ((1 - mask) ** beta) * (1 - targets)
    else:
        coef = targets + (1 - targets)
    loss = loss * coef
    loss = loss.mean(1)
    return loss.mean()


    
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