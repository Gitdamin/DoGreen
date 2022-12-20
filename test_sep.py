# /home/damin/Downloads/test
# head & helmet seperate!

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
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='kind of train/model name')
# parser.add_argument("--predict", type=str, default="head", help="head/helmet")
parser.add_argument("--model_name", type=str, default="res18", help="res*/VGG*")
args = parser.parse_args()

# fix seed
def seed_everything(seed):
    torch.manual_seed(seed) #torch를 거치는 모든 난수들의 생성순서를 고정한다
    torch.cuda.manual_seed(seed) #cuda를 사용하는 메소드들의 난수시드는 따로 고정해줘야한다 
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True #딥러닝에 특화된 CuDNN의 난수시드도 고정 
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed) #numpy를 사용할 경우 고정
    random.seed(seed) #파이썬 자체 모듈 random 모듈의 시드 고정

seed_everything(42)

# /home/ydm/4-2/project/dataset/damin_2/head
ROOT = '/home/ydm/4-2/project/dataset/final/sep_final/'
# '/home/ydm/4-2/project/dataset/multi-feature/test/'
data_dir_head = ROOT + 'head' # /home/ydm/4-2/project/dataset/multi-feature/test/head'
data_dir_helmet = ROOT + 'helmet' # /home/ydm/4-2/project/dataset/multi-feature/test/helmet'

# data_dir = '' 
# print('Folders :', os.listdir(data_dir))
head_classes = os.listdir(data_dir_head)
helmet_classes = os.listdir(data_dir_helmet)

print('Classes :', head_classes)
# exit()

test_transform = transforms.Compose([transforms.Resize((244,224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


head_test_set = ImageFolder(data_dir_head, transform = test_transform)
helmet_test_set = ImageFolder(data_dir_helmet, transform = test_transform)
total = len(head_test_set + helmet_test_set)
print('Size of all dataset :', total)

head_test_loader = DataLoader(head_test_set, 1, num_workers=2, pin_memory=True)
helmet_test_loader = DataLoader(helmet_test_set, 1, num_workers=2, pin_memory=True)

### EDIT ###
if 'res' in args.model_name:
    if '18' in args.model_name:
        print("We test res18 model!")
        head_model = models.resnet18(pretrained=True)
        helmet_model = models.resnet18(pretrained=True)

    elif '34' in args.model_name:
        print("We test res34 model!")
        head_model = models.resnet34(pretrained=True)
        helmet_model = models.resnet34(pretrained=True)

    else:
        print("We test res50 model!")
        head_model = models.resnet50(pretrained=True)
        helmet_model = models.resnet50(pretrained=True)

    num_features = head_model.fc.in_features
    head_model.fc = nn.Linear(num_features, 2)
    num_features = helmet_model.fc.in_features
    helmet_model.fc = nn.Linear(num_features, 2)

elif 'VGG' in args.model_name: # GPU error
    print("We test VGG16 model!")
    head_model = models.vgg16(pretrained=True)
    head_model.classifier[6] = nn.Linear(in_features=4096, out_features=2)
    helmet_model = models.vgg16(pretrained=True)
    helmet_model.classifier[6] = nn.Linear(in_features=4096, out_features=2)
    
elif 'Mobile' in args.model_name:
    print("We test MobileNetV2 model!")
    head_model = models.mobilenet_v2(pretrained=True)
    head_model.classifier = nn.Linear(in_features=1280, out_features=2)
    helmet_model = models.mobilenet_v2(pretrained=True)
    helmet_model.classifier = nn.Linear(in_features=1280, out_features=2)

elif 'google' in args.model_name:
    print("We test GoogleNet model!")
    head_model = models.googlenet(pretrained=True)
    head_model.fc = nn.Linear(in_features=1024, out_features=2)
    helmet_model = models.googlenet(pretrained=True)
    helmet_model.fc = nn.Linear(in_features=1024, out_features=2)

def test_helmet(model,testloader):
    model.eval()
    test_loss = 0
    correct = 0
    # total = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(testloader):
                
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                test_loss += nn.functional.cross_entropy(outputs, targets,reduction='sum').item()
                pred = outputs.max(1, keepdim=True)[1]
                correct += pred.eq(targets.view_as(pred)).sum().item()
               
        epoch_loss = test_loss /len(testloader.dataset)
        epoch_acc = correct / len(testloader.dataset)
        print("## Test Results!! ##")
        print("total helmet dataset :", len(testloader.dataset))
        print("Correct :", correct)
        print('helmet test | Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

import cv2
def test_head(model,testloader):
    model.eval()
      
    test_loss = 0
    correct = 0
    # total = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(testloader):
                
                # plt.imshow(inputs[0]) #.transpose(2,0,1))
                # plt.show()
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                test_loss += nn.functional.cross_entropy(outputs, targets,reduction='sum').item()
                pred = outputs.max(1, keepdim=True)[1]
                # if pred
                correct += pred.eq(targets.view_as(pred)).sum().item()
               
        epoch_loss = test_loss /len(testloader.dataset)
        epoch_acc = correct / len(testloader.dataset)
        print("## Test Results!! ##")
        print("total head dataset :", len(testloader.dataset))
        print("Correct :", correct)
        print('head test | Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

BASE = '/home/ydm/4-2/project/sep/weight/'
# /home/ydm/4-2/project/sep/weight/new_one/head
# /home/ydm/4-2/project/sep/weight/head
# checkpoint_head = torch.load(BASE + 'head/best_model_res50.pth') #, map_location=torch.device('cpu'))
# best_model_MobileNetV2.pth
checkpoint_head = torch.load(BASE + 'new_one/head/best_model_' + args.model_name + '.pth') #, map_location=torch.device('cpu'))
checkpoint_helmet = torch.load(BASE + 'new_one/helmet/best_model_' + args.model_name + '.pth') #, map_location=torch.device('cpu'))

head_model.load_state_dict(checkpoint_head)   
helmet_model.load_state_dict(checkpoint_helmet)        

test_head(head_model,head_test_loader)
test_helmet(helmet_model,helmet_test_loader)