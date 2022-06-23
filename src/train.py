import torch
from src.faster_rcnn import FasterRCNN
from src.dataset import AnnotationDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.visualize import visualize

def train():
    faster_rcnn = FasterRCNN()
    faster_rcnn.train()
    lr = 1e-3
    opt = torch.optim.Adam(faster_rcnn.parameters(), lr=lr)
    dataset = AnnotationDataset()
    data_loader = DataLoader(dataset, shuffle=True, batch_size=1)
    for epoch in range(20):
        epoch = epoch+1
        for batch_id, (img, boxes) in enumerate(tqdm(data_loader)):
            batch_id = batch_id+1
            classes = torch.ones((len(boxes[0]),))
            cls_logits, roi_pred_deltas, rois, rpn_loss, roi_loss = faster_rcnn(img, boxes[0], classes)
            total_loss = rpn_loss + roi_loss
            opt.zero_grad()
            total_loss.backward()
            opt.step()
            
            if batch_id % 30 == 0:
                visualize(faster_rcnn, f"{epoch}_batch_{batch_id}.jpg")


if __name__ == "__main__":
    train()
    