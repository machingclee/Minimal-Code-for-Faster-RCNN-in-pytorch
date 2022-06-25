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
from src.utils import clip_box, too_small
from torchvision.ops import nms

def visualize(fast_rcnn: nn.Module = None, image_name=None):
    dataset = AnnotationDataset(mode="test")
    
    img, boxes, target_indexes = next(iter(dataset))    
    
    with torch.no_grad():
        if fast_rcnn is None:
            fast_rcnn = FasterRCNN().to(device)
            
        fast_rcnn.eval()
               
        img, padding_window, original_wh = resize_and_padding(img,return_window=True)
        img_ori = deepcopy(img)
        img = torch_img_transform(img).to(device)
        
        cls_logits, pred_boxes = fast_rcnn(img[None, ...])
        
        scores = cls_logits.softmax(dim=-1)
        # scores = pred_fg_bg_logits.softmax(dim=-1)
        
        draw = ImageDraw.Draw(img_ori)
        # for score, box in zip(scores, rois):
        bg_scores= scores[:,0]
        objness_scores = 1 - bg_scores
        nonbg_scores = scores[:, 1:3]
        clses_index = torch.argmax(nonbg_scores, dim=-1)
        clses_index = clses_index + 1
        
        target_boxes = []
        
        for i, pred_boxe_by_cls in enumerate(pred_boxes):
            pred_box = pred_boxe_by_cls[clses_index[i]]
            target_boxes.append(pred_box)
            
        target_boxes = torch.stack(target_boxes, dim=0)
        
        keep = nms(target_boxes, objness_scores, 0.5)
        objness_scores = objness_scores[keep]
        target_boxes = target_boxes[keep]
  
        for obj_score, score_by_cls, target_box, cls in zip(objness_scores, scores, target_boxes, clses_index):
            if obj_score > 0.25:
                target_score = score_by_cls[cls]
                if target_score > 0.6:
                    target_box = clip_box(target_box)
                    
                    if too_small(target_box):
                        continue 
                    
                    xmin, ymin, xmax, ymax = target_box
                    draw.rectangle(((xmin, ymin), (xmax, ymax)), outline='blue', width=1)
                    draw.text((xmin + 1, ymin + 1),
                            "{}: {:.2f}".format(dataset.classnames[cls.item()], target_score.item()), (255, 255, 255))
        
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
        
        
        
    
    