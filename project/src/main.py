from __future__ import annotations
import os
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')
import re
from utils.data_loader import ImageDataset


obj = ImageDataset(label_dir='../data/annotations', img_dir='../data/images')
