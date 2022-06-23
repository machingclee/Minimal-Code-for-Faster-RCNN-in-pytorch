import os, random, torch
import torch.nn as nn
from datetime import datetime
from src.faster_rcnn import FasterRCNN
from src.box_utils import decode_deltas_to_boxes
from src.dataset import torch_img_transform, resize_and_padding
from src import config
from glob import glob
from PIL import Image, ImageDraw
from typing import cast
from torch import Tensor
from copy import deepcopy
from src.device import device

def visualize(fast_rcnn: nn.Module = None, image_name=None):
    with torch.no_grad():
        if fast_rcnn is None:
            fast_rcnn = FasterRCNN().to(device)

        fast_rcnn.eval()
        test_imgs = glob(f"{config.test_img_dir}/*.jpg")
        random.shuffle(test_imgs)
        img = test_imgs[0]
        img = Image.open(img)
    
        img, padding_window, (ori_w, ori_h) = resize_and_padding(img,return_window=True)
        img_ori = deepcopy(img)
        img = torch_img_transform(img).to(device)
        cls_logits, roi_pred_deltas, rois = fast_rcnn(img[None, ...])
        scores = cast(Tensor, cls_logits).softmax(dim=-1)
        pred_boxes = decode_deltas_to_boxes(roi_pred_deltas, rois)
        
        draw = ImageDraw.Draw(img_ori)
        
        for box in pred_boxes:
            xmin, ymin, xmax, ymax = box.squeeze(0)
            draw.rectangle(((xmin, ymin), (xmax,ymax)), outline='Green')
        
        if image_name is None:
            imgname = str(datetime.now()).split(" ")[0] +".jpg"
        else:
            imgname = image_name 
        img_ori.save("performance_check/{}".format(imgname))
    
    fast_rcnn.train()
    
if __name__=="__main__":
    visualize()
        
        
        
    
    