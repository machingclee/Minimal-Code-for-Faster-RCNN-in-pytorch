import torch
import numpy as np
import math
import torchvision
from src.anchor_generator import AnchorGenerator
from src import config
from torch import Tensor
from src.device import device
from typing import List, Tuple
from src.utils import random_choice


def assign_targets_to_anchors_or_proposals(
    target_boxes,
    anchors,
    n_sample,
    pos_sample_ratio,
    pos_iou_thresh,
    neg_iou_thresh,
    target_cls_indexes=None
):
    target_boxes = target_boxes.to(device)
    anchors = anchors.to(device)
    """
    return:
        labels: [len(anchors), ], -1 = neg, 0 = ignore, 1 = pos
        target: [len(anchors), 4]
    """
    anchors = anchors.to(device)
    labels = torch.zeros(len(anchors), dtype=torch.int32).to(device)
    distributed_cls_indexes = None

    if target_cls_indexes is not None:
        distributed_cls_indexes = torch.zeros(len(anchors), dtype=torch.float32).to(device)

    distributed_targets = torch.zeros(len(anchors), 4, dtype=torch.float32).to(device)

    ious = box_iou(anchors, target_boxes)
    max_iou_anchor_index = torch.argmax(ious, dim=0)  # return 2 anchor indexes corresponding to 2 targets
    labels[max_iou_anchor_index] = 1
    distributed_targets[max_iou_anchor_index] = target_boxes

    if target_cls_indexes is not None:
        distributed_cls_indexes[max_iou_anchor_index] = target_cls_indexes

    max_iou_target_idx_per_anchor = torch.argmax(ious, dim=1)
    max_ious = ious[torch.arange(len(anchors)), max_iou_target_idx_per_anchor]
    pos_index = torch.where(max_ious > pos_iou_thresh)[0]
    neg_mask = max_ious < neg_iou_thresh
    distributed_targets[pos_index] = target_boxes[max_iou_target_idx_per_anchor[pos_index]]

    if target_cls_indexes is not None:
        distributed_cls_indexes[pos_index] = target_cls_indexes[max_iou_target_idx_per_anchor[pos_index]]

    labels[pos_index] = 1
    pos_mask = labels == 1
    non_pos_mask = torch.logical_not(pos_mask).to(device)
    labels[non_pos_mask * neg_mask] = -1

    pos_ratio = pos_sample_ratio
    n_desired_pos_sample = n_sample * pos_ratio

    actual_pos_anchor_index = torch.where(labels == 1)[0]
    n_actual_pos_anchor = len(actual_pos_anchor_index)

    if n_actual_pos_anchor > n_desired_pos_sample:
        surplus = n_actual_pos_anchor - n_desired_pos_sample
        discarded_index = random_choice(actual_pos_anchor_index, surplus)
        labels[discarded_index] = 0

    n_desired_neg_sample = n_sample - torch.sum(labels == 1)

    actual_neg_anchor_index = torch.where(labels == -1)[0]
    n_actual_neg_anchor = len(actual_neg_anchor_index)

    if n_actual_neg_anchor > n_desired_neg_sample:
        surplus = n_actual_neg_anchor - n_desired_neg_sample
        discarded_index = random_choice(actual_neg_anchor_index, surplus)
        labels[discarded_index] = 0

    return labels, distributed_targets, distributed_cls_indexes


def encode_boxes_to_deltas(distributed_targets, anc_or_pro):
    # type: (Tensor, Tensor) -> Tensor
    epsilon = 1e-8
    anc_or_pro = anc_or_pro.to(device)
    anchors_x1 = anc_or_pro[:, 0].unsqueeze(1)
    anchors_y1 = anc_or_pro[:, 1].unsqueeze(1)
    anchors_x2 = anc_or_pro[:, 2].unsqueeze(1)
    anchors_y2 = anc_or_pro[:, 3].unsqueeze(1)

    target_boxes_x1 = distributed_targets[:, 0].unsqueeze(1)
    target_boxes_y1 = distributed_targets[:, 1].unsqueeze(1)
    target_boxes_x2 = distributed_targets[:, 2].unsqueeze(1)
    target_boxes_y2 = distributed_targets[:, 3].unsqueeze(1)

    an_widths = anchors_x2 - anchors_x1
    an_heights = anchors_y2 - anchors_y1
    an_ctr_x = anchors_x1 + 0.5 * an_widths
    an_ctr_y = anchors_y1 + 0.5 * an_heights

    gt_widths = target_boxes_x2 - target_boxes_x1
    gt_heights = target_boxes_y2 - target_boxes_y1
    gt_ctr_x = target_boxes_x1 + 0.5 * gt_widths
    gt_ctr_y = target_boxes_y1 + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - an_ctr_x) / (an_widths)
    targets_dy = (gt_ctr_y - an_ctr_y) / (an_heights)
    targets_dw = torch.log((gt_widths + epsilon) / (an_widths))
    targets_dh = torch.log((gt_heights + epsilon) / (an_heights))

    deltas = torch.cat((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
    return deltas


def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format

    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1, boxes2):
    boxes1 = boxes1.to(device)
    boxes2 = boxes2.to(device)
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    if len(boxes1.shape) == 1:
        boxes1 = boxes1[None, ...]
    if len(boxes2.shape) == 1:
        boxes2 = boxes2[None, ...]

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    #  When the shapes do not match,
    #  the shape of the returned output tensor follows the broadcasting rules
    upper_left = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # left-top [N,M,2]
    lower_right = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # right-bottom [N,M,2]

    wh = (lower_right - upper_left).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    epsilon = 1e-8
    iou = inter / (area1[:, None] + area2 - inter + epsilon)
    return iou


def decode_deltas_to_boxes(deltas, anchors):
    deltas = deltas.to(device)
    anchors = anchors.to(device)[None, ...]
    # type: (Tensor, List[Tensor]) -> Tensor
    if not isinstance(anchors, (list, tuple)):
        anchors = [anchors]
    assert isinstance(anchors, (list, tuple))
    assert isinstance(deltas, torch.Tensor)
    n_boxes_per_image = [b.size(0) for b in anchors]
    concat_boxes = torch.cat(anchors, dim=0).squeeze(0)

    box_sum = 0
    for val in n_boxes_per_image:
        box_sum += val
    # single mean single feature scale,
    # there are many scales in fpn and each scales contain many boxes
    pred_boxes = decode_single(
        deltas, concat_boxes
    )

    if box_sum > 0:
        pred_boxes = pred_boxes.reshape(-1, box_sum, 4)

    return pred_boxes


def decode_single(deltas, anchors, weights=[1, 1, 1, 1]):
    """
    weights: in RPN we use [1,1,1,1], in fastrcnn we use [10,10,5,5]
    From a set of original boxes and encoded relative box offsets,
    get the decoded boxes.

    Arguments:
        rel_codes (Tensor): encoded boxes (bbox regression parameters)
        boxes (Tensor): reference boxes (anchors/proposals)
    """
    anchors = anchors.to(deltas.dtype)
    bbox_xform_clip = math.log(1000. / 16)
    # xmin, ymin, xmax, ymax
    widths = anchors[..., 2] - anchors[..., 0]
    heights = anchors[..., 3] - anchors[..., 1]
    ctr_x = anchors[..., 0] + 0.5 * widths
    ctr_y = anchors[..., 1] + 0.5 * heights

    wx, wy, ww, wh = weights
    dx = deltas[..., 0] / wx
    dy = deltas[..., 1] / wy
    dw = deltas[..., 2] / ww
    dh = deltas[..., 3] / wh

    # limit max value, prevent sending too large values into torch.exp()
    # bbox_xform_clip=math.log(1000. / 16)   4.135
    dw = torch.clamp(dw, max=bbox_xform_clip)
    dh = torch.clamp(dh, max=bbox_xform_clip)

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    xmins = pred_ctr_x - torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
    ymins = pred_ctr_y - torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
    xmaxs = pred_ctr_x + torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
    ymaxs = pred_ctr_y + torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h

    pred_boxes = torch.stack((xmins, ymins, xmaxs, ymaxs), dim=2).flatten(1)
    # [22500, batch_size, 4]
    return pred_boxes


def clip_boxes_to_image(boxes, size=config.image_shape):
    # type: (Tensor, Tuple[int, int]) -> Tensor
    """
    Clip boxes so that they lie inside an image of size `size`.
    Arguments:
        boxes (Tensor[N, 4]): boxes in (x1, y1, x2, y2) format
        size (Tuple[height, width]): size of the image

    Returns:
        clipped_boxes (Tensor[N, 4])
    """
    dim = boxes.dim()
    boxes_x = boxes[..., 0:: 2]  # x1, x2
    boxes_y = boxes[..., 1:: 2]  # y1, y2
    height, width = size

    boxes_x = boxes_x.clamp(min=0, max=width)
    boxes_y = boxes_y.clamp(min=0, max=height)

    clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)
    return clipped_boxes.reshape(boxes.shape)
