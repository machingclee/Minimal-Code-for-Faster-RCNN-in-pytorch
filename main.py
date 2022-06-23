import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
from torchsummary import summary
from src.faster_rcnn import FasterRCNN
from src.rpn import RPN
from src.box_utils import box_iou, assign_targets_to_anchors_or_proposals, encode_boxes_to_deltas
from src.anchor_generator import AnchorGenerator


def main():
    # anchor_gen = AnchorGenerator()
    # anchors = anchor_gen.get_anchors()
    # anchors = torch.as_tensor(anchors)
    # # training logic:
    # target_boxes = np.array([[160, 147, 260, 234], [139, 312, 200, 348]])
    # target_boxes = torch.as_tensor(target_boxes)
    # target_cls_indexes = torch.as_tensor([1, 1], dtype=torch.int32)
    # anchors = AnchorGenerator().get_anchors()
    # dummy_img = torch.randn((1, 3, 800, 800))
    # faster_rcnn = FasterRCNN()
    # a = faster_rcnn(dummy_img, target_boxes, target_cls_indexes)
    # print(a.shape)


if __name__ == "__main__":
    main()
