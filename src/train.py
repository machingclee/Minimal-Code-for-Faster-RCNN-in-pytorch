import torch
from src.faster_rcnn import FasterRCNN
from src.dataset import AnnotationDataset
from torch.utils.data import DataLoader
from tqdm import tqdm


def train():
    faster_rcnn = FasterRCNN()
    faster_rcnn.train()
    lr = 1e-3
    opt = torch.optim.Adam(faster_rcnn.parameters(), lr=lr)
    dataset = AnnotationDataset()
    data_loader = DataLoader(dataset, shuffle=True, batch_size=1)
    for epoch in range(10):
        for img, boxes in tqdm(data_loader):
            classes = torch.ones((len(boxes[0]),))
            cls_logits, roi_pred_deltas, rois, rpn_loss, roi_loss = faster_rcnn(img, boxes[0], classes)
            total_loss = rpn_loss + roi_loss
            opt.zero_grad()
            total_loss.backward()
            opt.step()


if __name__ == "__main__":
    train()
    