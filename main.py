import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
from torchsummary import summary
from src.box_utils import box_iou, encode_target_to_anchors
from src.anchor_generator import AnchorGenerator


def main():
    anchor_gen = AnchorGenerator()
    anchors = anchor_gen.get_anchors()
    anchors = torch.as_tensor(anchors)
    bbox0 = np.array([[160, 147, 260, 234], [139, 312, 200, 348]])
    bbox0 = torch.as_tensor(bbox0)
    for bbox in bbox0:
        encode_target_to_anchors(bbox)


if __name__ == "__main__":
    main()
