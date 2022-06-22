import torch
from src.anchor_generator import AnchorGenerator
from src import config
from torch import Tensor
from src.device import device


def encode_target_to_anchors(target_box):
    anchors = AnchorGenerator().get_anchors()
    fg_bg_ignores = torch.zeros(len(anchors),)
    deltas = torch.zeros(len(anchors), 4)
    ious = box_iou(anchors, target_box[None, ...])
    ious = ious[..., 0]

    max_iou = torch.max(ious)
    max_iou_indexes = ious == max_iou
    fg_bg_ignores[max_iou_indexes] = 1

    fg_anchor = ious >= config.target_pos_iou_thres
    fg_bg_ignores[fg_anchor] = 1

    bg_anchor = ious <= config.target_neg_iou_thres
    fg_bg_ignores[bg_anchor] = -1


def assign_targets_to_anchors(targets, anchors):

    labels = []
    matched_gt_boxes = []

    gt_boxes = targets
    if gt_boxes.numel() == 0:
        matched_gt_boxes_per_image = torch.zeros_like(anchors, dtype=torch.float32, device=device)
        labels_per_image = torch.zeros_like((anchors[0],), dtype=torch.float32, device=device)
    else:
        ious = box_iou(gt_boxes, anchors)

        matched_idxs = ious

        matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]

        labels_per_image = matched_idxs >= 0
        labels_per_image = labels_per_image.to(dtype=torch.float32)

        bg_indices = matched_idxs == proposal_matcher.BELOW_LOW_THRESHOLD  # -1
        labels_per_image[bg_indices] = 0.0

        inds_to_discard = matched_idxs == proposal_matcher.BETWEEN_THRESHOLDS  # -2
        labels_per_image[inds_to_discard] = -1.0

        labels.append(labels_per_image)
        matched_gt_boxes.append(matched_gt_boxes_per_image)
    return labels, matched_gt_boxes


def encode_boxes(target_boxes, anchors):
    # type: (Tensor, Tensor) -> Tensor
    anchors_x1 = anchors[:, 0].unsqueeze(1)
    anchors_y1 = anchors[:, 1].unsqueeze(1)
    anchors_x2 = anchors[:, 2].unsqueeze(1)
    anchors_y2 = anchors[:, 3].unsqueeze(1)

    target_boxes_x1 = target_boxes[:, 0].unsqueeze(1)
    target_boxes_y1 = target_boxes[:, 1].unsqueeze(1)
    target_boxes_x2 = target_boxes[:, 2].unsqueeze(1)
    target_boxes_y2 = target_boxes[:, 3].unsqueeze(1)

    an_widths = anchors_x2 - anchors_x1
    an_heights = anchors_y2 - anchors_y1
    an_ctr_x = anchors_x1 + 0.5 * an_widths
    an_ctr_y = anchors_y1 + 0.5 * an_heights

    gt_widths = target_boxes_x2 - target_boxes_x1
    gt_heights = target_boxes_y2 - target_boxes_y1
    gt_ctr_x = target_boxes_x1 + 0.5 * gt_widths
    gt_ctr_y = target_boxes_y1 + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - an_ctr_x) / an_widths
    targets_dy = (gt_ctr_y - an_ctr_y) / an_heights
    targets_dw = torch.log(gt_widths / an_widths)
    targets_dh = torch.log(gt_heights / an_heights)

    targets = torch.cat((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
    return targets


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
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    #  When the shapes do not match,
    #  the shape of the returned output tensor follows the broadcasting rules
    upper_left = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # left-top [N,M,2]
    lower_right = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # right-bottom [N,M,2]

    wh = (lower_right - upper_left).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou
