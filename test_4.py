# /home/ydm/4-2/project/dataset/multi-feature/test/data

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
# 헬멧o, 전방미주시(hY_fN)
# /home/ydm/4-2/project/dataset/final
ROOT = '/home/ydm/4-2/project/dataset/final'

# '/home/ydm/4-2/project/dataset/multi-feature/test/'
# data_dir_head = ROOT + 'head' # /home/ydm/4-2/project/dataset/multi-feature/test/head'
data_dir = ROOT 

classes = os.listdir(data_dir)

print('Classes :', classes)
# exit()

test_transform = transforms.Compose([transforms.Resize((244,224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


test_set = ImageFolder(data_dir, transform = test_transform)
total = len(test_set)
print('Size of all dataset :', total)

test_loader = DataLoader(test_set, 1, num_workers=2, pin_memory=True)

### EDIT ###
if 'res' in args.model_name:
    if '18' in args.model_name:
        print("We test res18 model!")
        model = models.resnet18(pretrained=True)

    elif '34' in args.model_name:
        print("We test res34 model!")
        model = models.resnet34(pretrained=True)

    else:
        print("We test res50 model!")
        model = models.resnet50(pretrained=True)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 4)

elif 'VGG' in args.model_name: # GPU error
    print("We test VGG16 model!")
    model = models.vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(in_features=4096, out_features=4)
    
elif 'Mobile' in args.model_name:
    print("We test MobileNetV2 model!")
    model = models.mobilenet_v2(pretrained=True)
    model.classifier = nn.Linear(in_features=1280, out_features=4)

elif 'google' in args.model_name:
    print("We test GoogleNet model!")
    model = models.googlenet(pretrained=True)
    model.fc = nn.Linear(in_features=1024, out_features=4)

def test(model,testloader):
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
        print("total dataset :", len(testloader.dataset))
        print("Correct :", correct)
        print('Test | Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

BASE = '/home/ydm/4-2/project/weight/'
# /home/ydm/4-2/project/sep/weight/new_one/head
# /home/ydm/4-2/project/sep/weight/head
# checkpoint_head = torch.load(BASE + 'head/best_model_res50.pth') #, map_location=torch.device('cpu'))
# best_model_Mobilenet.pth
checkpoint_head = torch.load(BASE + 'best_model_' + args.model_name + '.pth') #, map_location=torch.device('cpu'))

model.load_state_dict(checkpoint_head)   

test(model, test_loader)
