from random import random
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchsummary import summary
from src.feature_extractor import FeatureExtractor
from src.box_utils import (
    assign_targets_to_anchors_or_proposals, clip_boxes_to_image,
    decode_deltas_to_boxes, encode_boxes_to_deltas, remove_small_boxes,
    decode_single
)
from src.rpn import RPN
from src import config
from typing import cast
from torch import Tensor
from torchvision.ops import nms
from src.utils import random_choice, smooth_l1_loss
from src.device import device
from torchvision.ops import roi_align
from PIL import Image, ImageDraw


cce_loss = nn.CrossEntropyLoss()


class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
        self.mlp_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.cls_score = nn.Linear(4096, config.n_classes)
        self.bbox_pred = nn.Linear(4096, config.n_classes * 4)

    def forward(self, pooling):
        x = pooling.reshape(config.roi_n_sample, -1)
        x = self.mlp_head(x)
        scores_logits = self.cls_score(x)
        deltas = self.bbox_pred(x)
        return scores_logits, deltas


class FasterRCNN(nn.Module):
    def __init__(self):
        super(FasterRCNN, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.rpn = RPN().to(device)
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
        keep_mask = torch.abs(labels) == 1

        
        objectness_label = torch.zeros_like(labels, device=device, dtype=torch.long)
        objectness_label[labels == 1] = 1.0
        
        rpn_cls_loss = cce_loss(pred_fg_bg_logit.squeeze(0)[keep_mask], objectness_label[keep_mask])
        
        if torch.sum(pos_mask) > 0:
            target_deltas = encode_boxes_to_deltas(distributed_targets, self.rpn.anchors)
            rpn_reg_loss = smooth_l1_loss(pred_deltas[pos_mask], target_deltas[pos_mask])
        else:
            rpn_reg_loss = torch.sum(pred_deltas)*0

        return rpn_cls_loss, rpn_reg_loss

    def get_roi_loss(self, labels, distributed_targets_to_roi, rois, pred_deltas, cls_logits, distributed_cls_index):
        target_deltas = encode_boxes_to_deltas(
            distributed_targets_to_roi, rois, weights=config.roi_head_encode_weights
        )
        N = cls_logits.shape[0]
        pred_deltas = pred_deltas.reshape(N, -1, 4)

        target_deltas = target_deltas[labels == 1]
        keep_mask = torch.abs(labels) == 1
        sub_labels = labels[keep_mask]

        distributed_cls_index = distributed_cls_index[keep_mask]
        pos_idx = torch.where(sub_labels == 1)[0]
        neg_idx = torch.where(sub_labels == -1)[0]
        classes = distributed_cls_index[pos_idx]

        roi_reg_loss = smooth_l1_loss(
            target_deltas,
            pred_deltas[pos_idx, classes.long()]
        )
        n_pos = len(pos_idx)
        n_neg = len(neg_idx)
        
        if n_pos > 0:
            roi_cls_loss = n_pos * cce_loss(cls_logits[pos_idx], distributed_cls_index[pos_idx].long())
            roi_cls_loss += n_neg * cce_loss(cls_logits[neg_idx], distributed_cls_index[neg_idx].long())
            roi_cls_loss = roi_cls_loss / (n_pos + n_neg)
        else:
            roi_cls_loss = n_neg * cce_loss(cls_logits[neg_idx], distributed_cls_index[neg_idx].long())
            roi_reg_loss = torch.sum(pred_deltas)*0
            
        return roi_cls_loss, roi_reg_loss

    def filter_boxes_by_scores_and_size(self, cls_logits, pred_boxes):
        cls_idxes = torch.arange(config.n_classes, device=device)
        cls_idxes = cls_idxes[None, ...].expand_as(cls_logits)

        scores = cls_logits.softmax(dim=1)[:, 1:]
        boxes = pred_boxes[:, 1:]
        cls_idxes = cls_idxes[:, 1:]

        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        cls_idxes = cls_idxes.reshape(-1)

        indxes = torch.where(torch.gt(scores, config.pred_score_thresh))[0]
        boxes = boxes[indxes]
        scores = scores[indxes]
        cls_idxes = cls_idxes[indxes]

        keep = remove_small_boxes(boxes, min_size=1)
        boxes = boxes[keep]
        scores = scores[keep]
        cls_idxes = cls_idxes[keep]

        return scores, boxes, cls_idxes

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

        out_feat = self.feature_extractor(x)

        features = out_feat
        pred_fg_bg_logits, pred_deltas = self.rpn(features)
        pred_fg_bg_logits = pred_fg_bg_logits.squeeze(0)
        pred_deltas = pred_deltas.squeeze(0)

        if self.training:
            rpn_cls_loss, rpn_reg_loss = self.get_rpn_loss(
                target_boxes, pred_deltas, pred_fg_bg_logits
            )

        rois = decode_deltas_to_boxes(pred_deltas.detach().clone(), self.rpn.anchors)
        rois = clip_boxes_to_image(rois)
        rois = rois.squeeze(0)

        if self.training:
            pred_fg_bg_logits, rois = self.filter_by_nms(
                pred_fg_bg_logits.detach().clone(),
                rois,
                config.n_train_pre_nms,
                config.n_train_post_nms,
                config.nms_train_iou_thresh
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

            pooling = roi_align(
                out_feat,
                [rois[torch.abs(labels) == 1].mul_(1 / 16.0)],
                (7, 7),
            )
        else:
            pred_fg_bg_logits, rois = self.filter_by_nms(
                pred_fg_bg_logits.clone(),
                rois.clone(),
                config.n_eval_pre_nms,
                config.n_eval_post_nms,
                config.nms_eval_iou_thresh
            )

            pred_fg_bg_logits = pred_fg_bg_logits[:config.roi_n_sample]
            rois = rois[:config.roi_n_sample]

            pooling = roi_align(
                out_feat,
                [rois.clone().mul_(1 / 16.0)],
                (7, 7)
            )

        cls_logits, roi_pred_deltas = self.mlp_detector(pooling)

        if self.training:
            roi_cls_loss, roi_reg_loss = self.get_roi_loss(
                labels,
                distributed_targets_to_roi,
                rois,
                roi_pred_deltas,
                cls_logits,
                distributed_cls_indexes
            )

        if self.training:
            return rpn_cls_loss, rpn_reg_loss, roi_cls_loss, roi_reg_loss
        else:
            N = rois.shape[0]
            roi_pred_deltas = roi_pred_deltas.reshape(N, -1, 4)
            pred_boxes = decode_deltas_to_boxes(
                roi_pred_deltas, rois, weights=config.roi_head_encode_weights
            ).squeeze(0)

            scores, boxes, cls_idxes = self.filter_boxes_by_scores_and_size(cls_logits, pred_boxes)
            cls_idxes = cls_idxes - 1

            return scores, boxes, cls_idxes, rois
