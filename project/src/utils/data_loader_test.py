from __future__ import annotations
import os
import pandas as pd
import torchvision
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')
import re
from sklearn import preprocessing

"""
WARNING: this class is needed only to correctly output the final results on the test dataset
"""
class ImageDatasetTest(Dataset):
    def __get_labels(self, image_name):
        labels = []
        image_name = re.findall(r'\d+', image_name)[0]
        for label_name in os.listdir(self.label_dir):
            with open(os.path.join(self.label_dir, label_name), 'r') as f:
                if image_name in f.read():
                    label_name, _ = os.path.splitext(label_name)
                    labels.append(label_name)
        return list(set(labels))

    def __init__(self, label_dir, img_dir, classes, transform=None, target_transform=None):
        self.label_dir = label_dir
        self.img_dir = img_dir
        self.classes = classes
        self.le = preprocessing.LabelEncoder()
        self.le.fit(self.classes)
        self.n_classes = len(classes)
        image_names = []
        label_names = []
        for img_name in os.listdir(self.img_dir):
            image_names.append(img_name)
            label_names.append(self.__get_labels(img_name))
        self.df = pd.DataFrame(columns=['image', 'labels'])
        self.df['image'] = image_names
        self.df['labels'] = label_names
        self.image_names = image_names
        del label_names
        self.transform = transform
        self.target_transform = target_transform
        self.enc_labels = list(self.le.inverse_transform([0,1,2,3,4,5,6,7,8,9,10,11,12,13]))

    def __len__(self):
        return len(self.df.index) 

    def __getitem__(self, index):
        img_name,label = self.df.values[index]
        img_path = os.path.join(self.img_dir, img_name)
        image = read_image(img_path, mode=torchvision.io.ImageReadMode.RGB)
        label = self.le.transform(label)
        target = torch.zeros(self.n_classes)
        target[label] = 1
        if self.transform is not None:
            image = self.transform(image)
        return image, target
