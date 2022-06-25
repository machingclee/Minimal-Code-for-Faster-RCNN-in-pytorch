from turtle import forward
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchsummary import summary
from torch import Tensor
from typing import Dict, List, Tuple, Optional, Any
from src import config
from src.device import device


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.vgg = models.vgg16(pretrained=True).to(device)
        self.features = self.vgg.features
        self.out_channels = None
        self.feature_extraction = nn.Sequential(*self._get_layers())

    def freeze_vgg(self):
        for param in self.vgg.parameters():
            param.requires_grad = False

    def unfreeze_vgg(self):
        for param in self.vgg.parameters():
            param.requires_grad = True

    def weight_init(self):
        for layer in self.vgg.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

    def _get_layers(self):
        dummy_img = torch.randn((1, 3, config.input_height, config.input_width)).to(device)
        x = dummy_img
        desired_layers = []
        for feat in self.features:
            x = feat(x)
            if x.shape[2] < config.input_height // 16:
                # desired ouput shape is 1024//16 = 64
                break
            desired_layers.append(feat)
            self.out_channels = x.shape[1]
        return desired_layers

    def forward(self, x):
        return self.feature_extraction(x)


if __name__ == "__main__":
    fe = FeatureExtractor()
    dummy_img = torch.randn((1, 3, config.input_height, config.input_width))
    print(fe(dummy_img).shape)
    print(fe.out_channels)
