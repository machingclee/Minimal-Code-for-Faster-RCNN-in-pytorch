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
from src.train import train

def main():
    train()


if __name__ == "__main__":
    main()
