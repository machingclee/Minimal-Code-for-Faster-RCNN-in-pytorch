import os, random, torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from src.faster_rcnn import FasterRCNN
from src.box_utils import decode_deltas_to_boxes
from src.dataset import AnnotationDataset, torch_img_transform, resize_and_padding
from src import config
from glob import glob
from PIL import Image, ImageDraw, ImageFont
from typing import cast
from torch import Tensor
from copy import deepcopy
from src.device import device
from torchvision.ops import nms

def visualize(fast_rcnn: nn.Module = None, image_name=None):
    dataset = AnnotationDataset(mode="test")
    
    img, boxes, _ = next(iter(dataset))    
    
    with torch.no_grad():
        if fast_rcnn is None:
            fast_rcnn = FasterRCNN().to(device)
            
        fast_rcnn.eval()
               
        img, padding_window, original_wh = resize_and_padding(img,return_window=True)
        img_ori = deepcopy(img)
        img = torch_img_transform(img).to(device)
        
        scores, boxes, cls_idxes, rois = fast_rcnn(img[None, ...])
        
        draw = ImageDraw.Draw(img_ori, 'RGBA')
        
        for roi in rois:
            draw.rectangle(((roi[0], roi[1]), (roi[2], roi[3])), outline=(255,255,255,150), width=1)
            
        keep = nms(boxes, scores, 0.5)
        scores = scores[keep]
        boxes = boxes[keep]
        cls_idxes = cls_idxes[keep]
            
        for score, box, cls_idx in zip(scores, boxes, cls_idxes):
            # if score < 0.25:
            #     continue 
            xmin, ymin, xmax, ymax = box
            draw.rectangle(((xmin, ymin), (xmax, ymax)), outline='blue', width=1)
            draw.text(
                (xmin + 4, ymin + 1),
                "{}: {:.2f}".format(dataset.classnames[cls_idx.item()], score.item()), (255, 255, 255)
            )

        if image_name is None:
            imgname = "test.jpg"
        else:
            imgname = image_name 
            
        img_ori = img_ori.crop(padding_window)
        img_ori.resize(original_wh)
        img_ori.save("performance_check/{}".format(imgname))
        img_ori.save("performance_check/latest.jpg".format(imgname))
    
        fast_rcnn.train()
    
if __name__=="__main__":
    visualize()
        
        
        
    
    