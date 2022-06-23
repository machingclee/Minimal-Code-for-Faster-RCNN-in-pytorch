import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchsummary import summary
from src.device import device
from src.anchor_generator import AnchorGenerator
from src.feature_extractor import FeatureExtractor
from src.box_utils import decode_deltas_to_boxes
from src import config
from typing import cast
from torch import Tensor


class RPNHead(nn.Module):
    def __init__(self):
        super(RPNHead, self).__init__()
        n_anchors = len(config.anchor_ratios) * len(config.anchor_scales)
        self.conv1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv_logits = nn.Conv2d(512, n_anchors * 2, 1, 1)
        self.conv_deltas = nn.Conv2d(512, n_anchors * 4, 1, 1)
        self.weight_init()

    def weight_init(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_normal_(module.weight)
                module.bias.data.zero_()

    def forward(self, feature):
        feature = self.conv1(feature)
        logits = self.conv_logits(feature)
        deltas = self.conv_deltas(feature)
        return logits, deltas


class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()
        self.anchors = AnchorGenerator().get_anchors()
        self.rpn_head = RPNHead().to(device)

    def forward(self, features):
        batch_size = features.shape[0]
        logits, deltas = self.rpn_head(features)

        logits = cast(Tensor, logits).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
        deltas = cast(Tensor, deltas).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)
        pred_boxes = decode_deltas_to_boxes(deltas, self.anchors)
        return logits, deltas, pred_boxes
