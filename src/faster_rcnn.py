from random import random
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchsummary import summary
from src.feature_extractor import FeatureExtractor
from src.box_utils import assign_targets_to_anchors_or_proposals, box_iou, clip_boxes_to_image, decode_deltas_to_boxes, encode_boxes_to_deltas
from src.rpn import RPN
from src import config
from typing import cast
from torch import Tensor
from torchvision.ops import nms
from src.utils import random_choice, smooth_l1_loss
from src.device import device
from torchvision.ops import RoIAlign
from PIL import Image, ImageDraw

cce_loss = nn.CrossEntropyLoss()


class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
        self.mlp_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.Linear(4096, 1024)
        )
        self.cls_score = nn.Linear(1024, config.n_classes)
        self.bbox_pred = nn.Linear(1024, config.n_classes * 4)

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
        self.rpn = RPN().to(device)
        self.roi_align = RoIAlign(output_size=7, sampling_ratio=2, spatial_scale=1.0)
        self.mlp_detector = Detector().to(device)

    def filter_small_rois(self, logits, rois):
        rois = rois.squeeze(1).unsqueeze(0)
        hs = rois[..., 2] - rois[..., 0]
        ws = rois[..., 3] - rois[..., 1]
        keep_mask = (hs >= config.min_size) * (ws >= config.min_size)
        logits = logits[keep_mask]
        rois = rois[keep_mask]
        return logits, rois

    def filter_by_nms(self, logits, rois, n_pre_nms, n_post_nms, thresh):
        scores = logits.softmax(dim=1)[:, 1]
        order = scores.ravel().argsort(descending=True)
        order = order[:n_pre_nms]
        scores = scores[order]
        rois = rois[order, :]
        keep = nms(rois, scores, thresh)
        keep = keep[:n_post_nms]
        logits = logits[keep]
        rois = rois[keep]
        return logits, rois

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
        labels = labels.to(device)
        pos_mask = labels == 1
        neg_mask = labels == -1
        target_deltas = encode_boxes_to_deltas(distributed_targets, self.rpn.anchors)
        objectness_label = torch.where(labels > 0, 1, 0)

        rpn_reg_loss = smooth_l1_loss(pred_deltas[pos_mask], target_deltas[pos_mask][None, ...])
        rpn_cls_loss = cce_loss(pred_fg_bg_logit.squeeze(0)[pos_mask], objectness_label[pos_mask])
        rpn_cls_loss += cce_loss(pred_fg_bg_logit.squeeze(0)[neg_mask], objectness_label[neg_mask])

        return rpn_cls_loss, rpn_reg_loss

    def get_roi_loss(self, labels, distributed_targets_to_roi, rois, pred_deltas, cls_logits, distributed_cls_index):
        target_deltas = encode_boxes_to_deltas(distributed_targets_to_roi, rois)
        target_deltas = target_deltas[labels == 1]
        keep_mask = torch.abs(labels) == 1
        sub_labels = labels[keep_mask]

        distributed_cls_index = distributed_cls_index[keep_mask]
        pos_idx = torch.where(sub_labels == 1)[0]
        neg_idx = torch.where(sub_labels == -1)[0]
        classes = distributed_cls_index[sub_labels == 1]

        roi_reg_loss = smooth_l1_loss(
            target_deltas,
            torch.stack([pred_deltas[(int(class_), int(pos_idx_))] for pos_idx_, class_ in zip(pos_idx, classes)])
        )
        roi_cls_loss = cce_loss(cls_logits[pos_idx], distributed_cls_index[pos_idx].long())
        roi_cls_loss += cce_loss(cls_logits[neg_idx], distributed_cls_index[neg_idx].long())
        return roi_cls_loss, roi_reg_loss

    def forward(
        self,
        x,
        target_boxes=None,
        target_cls_indexes=None
    ):
        x = x.to(device)

        if target_boxes is not None:
            target_boxes = target_boxes.to(device)

        if target_cls_indexes is not None:
            target_cls_indexes = target_cls_indexes.to(device)

        if self.training:
            assert target_boxes is not None
            assert target_cls_indexes is not None

        features = self.feature_extractor(x)
        pred_fg_bg_logits, pred_deltas = self.rpn(features)
        pred_fg_bg_logits = pred_fg_bg_logits.squeeze(0)
        pred_deltas = pred_deltas.squeeze(0)

        if self.training:
            rpn_cls_loss, rpn_reg_loss = self.get_rpn_loss(target_boxes, pred_deltas, pred_fg_bg_logits)

        rois = decode_deltas_to_boxes(pred_deltas.detach(), self.rpn.anchors)
        rois = clip_boxes_to_image(rois)
        rois = rois.squeeze(0)

        if self.training:
            pred_fg_bg_logits, rois = self.filter_by_nms(
                pred_fg_bg_logits,
                rois,
                config.n_train_pre_nms,
                config.n_train_post_nms,
                config.nms_iou_thresh
            )
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
            if pooling.shape[0] == 1:
                print("test")

        else:
            pred_fg_bg_logits, rois = self.filter_by_nms(
                pred_fg_bg_logits.clone(),
                rois.clone(),
                config.n_test_pre_nms,
                config.n_test_post_nms,
                config.nms_iou_thresh
            )

            pred_fg_bg_logits = pred_fg_bg_logits[:config.roi_n_sample]
            rois = rois[:config.roi_n_sample]

            pooling = self.roi_align(
                features,
                [rois.clone().mul_(1 / 16.0)]
            )

        cls_logits, roi_pred_deltas = self.mlp_detector(pooling)
        roi_pred_deltas = roi_pred_deltas.view(-1, config.roi_n_sample, 4)

        if self.training:
            roi_cls_loss, roi_reg_loss = self.get_roi_loss(
                labels,
                distributed_targets_to_roi,
                rois.detach(),
                roi_pred_deltas,
                cls_logits,
                distributed_cls_indexes
            )

        if self.training:
            return rpn_cls_loss, rpn_reg_loss, roi_cls_loss, roi_reg_loss
        else:
            pred_boxes = decode_deltas_to_boxes(roi_pred_deltas, rois).squeeze(0)
            pred_boxes = pred_boxes.permute(1, 0, 2)
            return cls_logits, pred_boxes
