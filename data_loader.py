
import os
import glob
import torch
import torch.nn as nn
# from models.models_utils import init_weights
import numpy as np
import time, copy
import cv2
from PIL import Image

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class CustomDataset(Dataset):
    def __init__(self, image_list, label_list, train_mode=True, transforms=None): 
        self.transforms = transforms
        self.train_mode = train_mode
        self.image_list = image_list
        self.label_list = label_list
        

    def __getitem__(self, index): # index번째 data를 return
        # self.img_path_list = train_img
        img_path = self.image_list[index]
        
        # Get image data
        image = cv2.imread(img_path)
        image  = Image.fromarray(image)
        
        if self.transforms is not None:
            image = self.transforms(image)

        if self.train_mode:
            # self.label_list = training_label
            # EDIT
            label1 = self.label_list[index]
            return image, label1, label2
        else:
            return image
    
    def __len__(self): # 길이 return
        return len(self.image_list)