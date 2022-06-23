import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchsummary import summary
from src.feature_extractor import FeatureExtractor
from src.box_utils import assign_targets_to_anchors_or_proposals, box_iou, clip_boxes_to_image, encode_boxes_to_deltas
from src.rpn import RPN
from src import config
from typing import cast
from torch import Tensor
from torchvision.ops import nms
from src.utils import random_choice
from torchvision.ops import RoIAlign

l1_loss = nn.L1Loss()
cce_loss = nn.CrossEntropyLoss()


class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
        self.mlp_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.Linear(4096, 4096)
        )
        self.cls_score = nn.Linear(4096, config.n_classes)
        self.bbox_pred = nn.Linear(4096, config.n_classes * 4)

    def forward(self, pooling):
        x = pooling.view(config.roi_n_sample, -1)
        x = self.mlp_head(x)
        scores_logits = self.cls_score(x)
        deltas = self.bbox_pred(x)
        return scores_logits, deltas


class FasterRCNN(nn.Module):
    def __init__(self):
        super(FasterRCNN, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.rpn = RPN()
        self.roi_align = RoIAlign(output_size=7, sampling_ratio=2, spatial_scale=1.0)
        self.mlp_detector = Detector()

    def filter_small_rois(self, logits, rois):
        hs = rois[..., 2] - rois[..., 0]
        ws = rois[..., 3] - rois[..., 1]
        keep_mask = (hs >= config.min_size) * (ws >= config.min_size)
        logits = logits[keep_mask]
        rois = rois[keep_mask]
        return logits, rois

    def filter_by_nms(self, logits, rois):
        scores = cast(Tensor, logits).softmax(dim=1)[:, 1]
        order = scores.ravel().argsort(descending=True)
        order = order[:config.n_train_pre_nms]
        scores = scores[order]
        rois = rois[order, :]
        keep = nms(rois, scores, config.nms_iou_thresh)
        keep = keep[:config.n_train_post_nms]
        rois = rois[keep]
        return rois

    def get_rpn_loss(self, target_boxes, pred_deltas, pred_fg_bg_logit):
        labels, distributed_targets, _ = assign_targets_to_anchors_or_proposals(
            target_boxes,
            self.rpn.anchors,
            n_sample=config.rpn_n_sample,
            pos_sample_ratio=config.rpn_pos_ratio,
            pos_iou_thresh=config.target_pos_iou_thres,
            neg_iou_thresh=config.target_neg_iou_thres,
            target_cls_indexes=None
        )

        pos_mask = labels == 1
        keep_mask = torch.abs(labels) == 1

        target_deltas = encode_boxes_to_deltas(distributed_targets, self.rpn.anchors)
        reg_loss = l1_loss(pred_deltas[:, pos_mask], target_deltas[pos_mask][None, ...])
        cls_loss = cce_loss(pred_fg_bg_logit.squeeze(0)[keep_mask], torch.where(labels[keep_mask] > 0, 1, 0))
        return cls_loss + 10 * reg_loss

    def get_roi_loss(self, labels, distributed_targets_to_roi, rois, pred_deltas, cls_logits, distributed_cls_index):
        target_deltas = encode_boxes_to_deltas(distributed_targets_to_roi, rois)
        target_deltas = target_deltas[labels == 1]
        keep_mask = torch.abs(labels) == 1
        sub_labels = labels[keep_mask]

        distributed_cls_index = distributed_cls_index[keep_mask]
        pos_idx = torch.where(sub_labels == 1)[0]
        classes = distributed_cls_index[sub_labels == 1]

        reg_loss = l1_loss(
            target_deltas,
            torch.stack([pred_deltas[(int(pos_idx_), int(class_))] for pos_idx_, class_ in zip(pos_idx, classes)])
        )
        cls_loss = cce_loss(cls_logits[pos_idx], classes.long())
        return cls_loss + 10 * reg_loss

    def forward(
        self,
        x,
        target_boxes=None,
        target_cls_indexes=None
    ):
        if self.training:
            assert target_boxes is not None
        features = self.feature_extractor(x)
        pred_fg_bg_logits, pred_deltas, rois = self.rpn(features)

        if self.training:
            rpn_loss = self.get_rpn_loss(target_boxes, pred_deltas, pred_fg_bg_logits)

        rois = clip_boxes_to_image(rois)
        pred_fg_bg_logits, rois = self.filter_small_rois(pred_fg_bg_logits, rois)

        if self.training:
            rois = self.filter_by_nms(pred_fg_bg_logits, rois)
            # distribute to anchors

            labels, distributed_targets_to_roi, distributed_cls_indexes = \
                assign_targets_to_anchors_or_proposals(
                    target_boxes,
                    rois,
                    n_sample=config.roi_n_sample,
                    pos_sample_ratio=config.roi_pos_ratio,
                    pos_iou_thresh=config.roi_pos_iou_thresh,
                    neg_iou_thresh=config.roi_neg_iou_thresh,
                    target_cls_indexes=target_cls_indexes
                )

            pooling = self.roi_align(
                features,
                [rois[torch.abs(labels) == 1].mul_(1 / 16.0)]
            )
        else:
            pooling = self.roi_align(
                features,
                [rois.mul_(1 / 16.0)]
            )

        cls_logits, roi_pred_deltas = self.mlp_detector(pooling)
        roi_pred_deltas = roi_pred_deltas.view(config.roi_n_sample, -1, 4)

        if self.training:
            roi_loss = self.get_roi_loss(
                labels,
                distributed_targets_to_roi,
                rois,
                roi_pred_deltas,
                cls_logits,
                distributed_cls_indexes
            )

        if self.training:
            return cls_logits, roi_pred_deltas, rois, rpn_loss, roi_loss
        else:
            return cls_logits, roi_pred_deltas, rois
