
import csv
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from src import config
from pydash import get, set_
from typing import List
from torchvision import transforms
from torchvision.transforms import ToPILImage


torch_img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

data_transform = transforms.Compose([transforms.ToTensor()])


def resize_img(img):
    """
    img:  Pillow image
    """
    h, w = img.height, img.width
    if h >= w:
        ratio = config.input_height / h
        new_h, new_w = int(h * ratio), int(w * ratio)
    else:
        ratio = config.input_height / h
        new_h, new_w = int(h * ratio), int(w * ratio)

        if new_w > config.input_width:
            ratio = config.input_width / new_w
            new_h, new_w = int(new_h * ratio), int(new_w * ratio)

    img = img.resize((new_w, new_h), Image.BILINEAR)
    return img, (w, h)


def pad_img(img):
    h = img.height
    w = img.width
    img = np.array(img)
    img = np.pad(img, pad_width=((0, config.input_height - h), (0, config.input_width - w), (0, 0)), mode="constant")
    img = Image.fromarray(img)
    assert img.height == config.input_height
    assert img.width == config.input_width
    return img


def resize_and_padding(img, return_window=False):

    img, (ori_w, ori_h) = resize_img(img)
    w = img.width
    h = img.height
    padding_window = (w, h)
    img = pad_img(img)

    if not return_window:
        return img
    else:
        return img, padding_window, (ori_w, ori_h)


class AnnotationDataset(Dataset):
    def __init__(self):
        super(AnnotationDataset, self).__init__()
        self.annotation = {}
        with open("dataset/train_solution_bounding_boxes.csv") as f:
            reader = csv.reader(f)
            next(reader)  # skip first line
            for line in reader:
                img_basename, x1, y1, x2, y2 = line
                box = [x1, y1, x2, y2]

                if self.annotation.get(img_basename, None) is None:
                    self.annotation.update({img_basename: [box]})
                else:
                    self.annotation[img_basename].append(box)

            self.data: List[str, List[List[float]]] = [[k, v] for k, v in self.annotation.items()]

    def __getitem__(self, index):
        data = self.data[index]
        img_basename, boxes = data

        boxes_ = []
        for box in boxes:
            box = [float(v) for v in box]
            boxes_.append(box)

        img_path = os.path.join(config.training_img_dir, img_basename)
        img = Image.open(img_path)
        img = resize_and_padding(img)

        img = torch_img_transform(img)
        boxes = torch.as_tensor(boxes_)
        return img, boxes

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    dataset = AnnotationDataset()
    a, b = dataset[0]
    print(a, b)
