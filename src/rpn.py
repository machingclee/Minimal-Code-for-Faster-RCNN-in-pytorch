import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchsummary import summary
from src.device import device
from src.anchor_generator import AnchorGenerator


class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()
        self.anchors = AnchorGenerator().get_anchors().to(device)

    def forward(self, x):
        pass
