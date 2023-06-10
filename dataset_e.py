import pandas as pd
import matplotlib.image as mpimg
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
import numpy as np
from PIL import Image


def load_image(uploaded_file):
    image = Image.open(uploaded_file).convert('RGB')
    return image

def create_dataloader(uploaded_file, is_train=True, shuffle=True, batch_size=128):
    image = load_image(uploaded_file)
    dataset = AestheticsDataset(image, is_train)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

class AestheticsDataset(Dataset):
    def __init__(self, image, is_train):
        self.image = image
        normalize_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize([299, 299]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize_transform
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize([299, 299]),
                transforms.ToTensor(),
                normalize_transform
            ])

    def __len__(self):
        return 1

    def __getitem__(self, _):
        img = self.transform(self.image)
        return {
            "image": img
        }
