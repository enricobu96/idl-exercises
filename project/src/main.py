from __future__ import annotations
import os
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, label_dir, img_dir, transform=None, target_transform=None):
        self.label_dir = label_dir
        self.img_dir = img_dir
        df = pd.DataFrame(columns=['image','labels'])
        for filename in os.listdir(self.img_dir):
            image_name = filename


        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        pass


obj = ImageDataset(label_dir='../data/annotations', img_dir='../data/images')





        # img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # image = read_image(img_path)
        # label = self.img_labels.iloc[idx, 1]
        # if self.transform:
        #     image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        # return image, label







# def __getitem__(self, index):
#        img_name,label = self.df[index]
#        img_path = os.path.join(self.data_dir, img_name+self.img_ext)
#        image = cv2.imread(img_path)
#        label = list(map(int, label.split(' '))) #'1 40 56 813' --> [1, 40, 56, 813]
#        target = torch.zeros(nb_classes)
#        target[label] = 1.
#        if self.transform is not None:
#            image = self.transform(image)
#        return image, target