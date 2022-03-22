from __future__ import annotations
import os
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')
import re


class ImageDataset(Dataset):
    def __get_labels(self, image_name):
        labels = []
        image_name = re.findall(r'\d+', image_name)[0]
        for label_name in os.listdir(self.label_dir):
            with open(os.path.join(self.label_dir, label_name), 'r') as f:
                if image_name in f.read():
                    label_name, _ = os.path.splitext(label_name)
                    labels.append(label_name)
        return list(set(labels))

    def __init__(self, label_dir, img_dir, transform=None, target_transform=None):
        self.label_dir = label_dir
        self.img_dir = img_dir
        self.n_classes = len([name for name in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir,name))])
        image_names = []
        label_names = []
        for img_name in os.listdir(self.img_dir):
            image_names.append(img_name)
            label_names.append(self.__get_labels(img_name))
        self.df = pd.DataFrame(columns=['image', 'labels'])
        self.df['image'] = image_names
        self.df['labels'] = label_names
        del image_names
        del label_names
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):
       img_name,label = self.df[index]
       img_path = os.path.join(self.data_dir, img_name+self.img_ext)
       image = read_image(img_path)
       target = torch.zeros(self.n_classes)
       target[label] = 1.
       if self.transform is not None:
           image = self.transform(image)
       return image, target