import torch
import numpy as np
from tqdm import tqdm
from src.visualize import visualize
from src.utils import ConsoleLog
from src.faster_rcnn import FasterRCNN
from src.dataset import AnnotationDataset
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw

console_log = ConsoleLog(lines_up_on_end=1)

def train():
    faster_rcnn = FasterRCNN()
    faster_rcnn.train()
    lr = 1e-4
    opt = torch.optim.Adam(faster_rcnn.parameters(), lr=lr)
    dataset = AnnotationDataset()
    data_loader = DataLoader(dataset, shuffle=True, batch_size=1)
    for epoch in range(60):
        epoch = epoch+1
        
        for batch_id, (img, boxes, cls_indexes) in enumerate(tqdm(data_loader)):
            batch_id = batch_id+1
            
            rpn_cls_loss, rpn_reg_loss, roi_cls_loss, roi_reg_loss = faster_rcnn(img, boxes[0], cls_indexes[0])
            total_loss = rpn_cls_loss + 10*rpn_reg_loss + roi_cls_loss + 40*roi_reg_loss
            opt.zero_grad()
            total_loss.backward()
            opt.step()
            
            with torch.no_grad():
                console_log.print([
                    ("total_loss", total_loss.item()),
                    ("-rpn_cls_loss", rpn_cls_loss.item()),
                    ("-rpn_reg_loss", rpn_reg_loss.item()),
                    ("-roi_cls_loss", roi_cls_loss.item()),
                    ("-roi_reg_loss", roi_reg_loss.item())
                ])
            
                if batch_id % 60 == 0:
                    visualize(faster_rcnn, f"{epoch}_batch_{batch_id}.jpg")



if __name__ == "__main__":
    train()
    