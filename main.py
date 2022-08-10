from src.faster_rcnn import FasterRCNN
from src.train import train, train_with_nan
from src.device import device
import torch

def train_model():
    model_path = "pths/model_epoch_1.pth"

    faster_rcnn = FasterRCNN().to(device)

    if model_path is not None:
        faster_rcnn.load_state_dict(torch.load(model_path))
       
    faster_rcnn.train() 
    train_with_nan(
        faster_rcnn,
        lr=1e-5,
        start_epoch=2,
        epoches=20,
        save_weight_interval=1
    ) 

if __name__ == "__main__":
    train_model()
