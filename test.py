# USB cam version
# helmet & head two class (separate)
# for visualize

from gettext import npgettext
from imghdr import tests
import cv2
import numpy as np
import datetime
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm.notebook import tqdm
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
import time, copy
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import torchvision.models as models
import glob

ROOT = '/home/ydm/4-2/project/dataset/test_test/'
testset = []
testset = glob.glob(ROOT + '*/*/*.JPG') # save full name
# print(testset)
# exit()
# video_capture = cv2.VideoCapture(0) # 내 컴퓨터 카메라 
classes = ['no', 'yes']
# exit()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_head = models.mobilenet_v2(pretrained=True)
model_helmet = models.mobilenet_v2(pretrained=True)

# print(model)
# exit()
model_head.classifier = nn.Linear(in_features=1280, out_features=2)
model_helmet.classifier = nn.Linear(in_features=1280, out_features=2)

test_transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((244,224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

checkpoint_head = torch.load('/home/ydm/4-2/project/sep/weight/head/best_model_mobileV2.pth') #, map_location=torch.device('cpu'))
checkpoint_helmet = torch.load('/home/ydm/4-2/project/sep/weight/helmet/best_model_mobileV2.pth') # , map_location=torch.device('cpu'))

model_head.load_state_dict(checkpoint_head)   
model_helmet.load_state_dict(checkpoint_helmet)   

count=0
for i in range(len(testset)):
    model_head.eval()
    model_helmet.eval()

    model_head.to(device)  
    model_helmet.to(device)  

    count+=1
    # grabbed, frame = video_capture.read()
    # cv2.imshow('Original Video', frame)
    image = cv2.imread(testset[i])
    # print(frame.shape)
    # image = frame
    img_origin = image.copy()
    image = test_transform(image).unsqueeze(0).to(device)
    # print(image.size())
    with torch.no_grad():
        head_outputs = model_head(image)
        helmet_outputs = model_helmet(image)
        _, head_preds = torch.max(head_outputs, 1)
        _, helmet_preds = torch.max(helmet_outputs, 1)
        # head_preds.eq(targets.view_as(pred))
        # print(head_preds)
        # imshow(image.cpu().data[0], title='예측 결과: ' + class_names[preds[0]])
        print("#"*15)
        print('Head Predict : ', classes[head_preds[0]])
        print('Helmet Predict : ', classes[helmet_preds[0]])
        cv2.imshow('Image', img_origin)
        
    key = cv2.waitKey(0)
    if key == ord('q'):
        break
#     elif key == ord('s'):
#         file = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f") + '.jpg'
#         cv2.imwrite(file, frame)
#         print(file, ' saved')

# video_capture.release()
cv2.destroyAllWindows()