# crossroad & general road classification code
# dog =1, cat =0
# reference : https://www.kaggle.com/code/pmigdal/transfer-learning-with-resnet-50-in-pytorch/notebook

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
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary


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

data_dir = '/home/ydm/4-2/project/dataset/multi-feature'
# print('Folders :', os.listdir(data_dir))
classes = os.listdir(data_dir + "/train")
print('Classes :', classes)
# exit()

train_transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
test_transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


dataset = ImageFolder(data_dir + '/train', transform = train_transform)
total = len(dataset)
print('Size of all dataset :', total)
train_num = int(total*0.8)
val_num = int(total*0.1)
test_num = total - train_num - val_num
total_num = train_num + val_num + test_num

assert total == total_num, "# of total dataset must be same!"

train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_num, val_num, test_num])    
print(len(train_set), len(val_set), len(test_set))  
exit()

batch_size = 16
train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_set, batch_size, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_set, batch_size, num_workers=2, pin_memory=True)


### EDIT ###
if 'res' in args.model_name:
    if '18' in args.model_name:
        print("We train res18 model!")
        model = models.resnet18(pretrained=True)

    elif '34' in args.model_name:
        print("We train res34 model!")
        model = models.resnet34(pretrained=True)

    else:
        print("We train res50 model!")
        model = models.resnet50(pretrained=True)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 4)

elif 'VGG' in args.model_name: # GPU error
    print("We train VGG16 model!")
    model = models.vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(in_features=4096, out_features=4)
    
elif 'Mobile' in args.model_name:
    print("We train MobileNetV2 model!")
    model = models.mobilenet_v2(pretrained=True)
    model.classifier = nn.Linear(in_features=1280, out_features=4)

elif 'google' in args.model_name:
    print("We train GoogleNet model!")
    model = models.googlenet(pretrained=True)
    model.fc = nn.Linear(in_features=1024, out_features=4)

# print(model)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model.to(device)
# summary(model, (3, 224, 224))
# exit()

# num_features = model.fc.in_features
# model.fc = nn.Linear(num_features, 4)
# binary classification
# model.fc = nn.Sequential(
#                nn.Linear(2048, 1024),
#                nn.ReLU(inplace=True),
#                nn.Linear(1024, 128),
#                nn.ReLU(inplace=True),
#                nn.Linear(128, 4))

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model.to(device)
# summary(model, (3, 224, 224))
# exit()

EPOCH = 20
# criterion = nn.BCELoss()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
# optimizer = optim.Adam(model.fc.parameters())

class TrainModel():
    def __init__(self,model, criterion, optimizer, trainloader, valloader, num_epochs=10):
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = model.to(self.device)
        self.trainloader =trainloader
        self.valloader = valloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.best_acc_wts = copy.deepcopy(self.model.state_dict())
        self.best_acc =0.0

        print('## Start learning!! ##')
        for epoch in range(1, self.num_epochs+1):
            
            epoch_loss, epoch_acc = self.train()
            if epoch % 3 ==0 :
                print('Epoch {}/{}'.format(epoch, self.num_epochs))
                print('train | Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
            epoch_loss, epoch_acc = self.val()
            if epoch % 3 ==0 :
                print('val | Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
            scheduler.step()
        model.load_state_dict(self.best_acc_wts)
        
    def train(self):
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            # print(targets)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            # writer.add_scalar("Loss/train", loss, batch_idx)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.data.cpu().numpy()
            # total += targets.size(0)
            # correct += outputs.eq(targets).sum().item()
            pred = outputs.max(1, keepdim=True)[1]
            correct += pred.eq(targets.view_as(pred)).sum().item()

        epoch_loss = train_loss /len(self.trainloader.dataset)
        epoch_acc = correct / len(self.trainloader.dataset)
        return epoch_loss, epoch_acc 
        # print('train | Loss: {:.4f} Acc: {:.4f}'.format( epoch_loss, epoch_acc))

    def val(self):
        self.model.eval()
        val_loss = 0
        correct = 0
        # total = 0
        with torch.no_grad():
            for _, (inputs, targets) in enumerate(self.valloader):
                # transforms.ToTensor()
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                # val_loss += self.criterion(outputs, targets, reduction = 'sum').item()
                val_loss += nn.functional.cross_entropy(outputs, targets,reduction='sum').item()
                
                # loss_fn = self.criterion(reduction='sum') 
                # val_loss += loss_fn(outputs, targets).item()
                pred = outputs.max(1, keepdim=True)[1]
                correct += pred.eq(targets.view_as(pred)).sum().item()

            epoch_loss = val_loss /len(self.valloader.dataset)
            epoch_acc = correct / len(self.valloader.dataset)
            
            # print('val | Loss: {:.4f} Acc: {:.4f}'.format( epoch_loss, epoch_acc))
            
            if epoch_acc >= self.best_acc:
                self.best_acc = epoch_acc
                self.best_acc_wts = copy.deepcopy(self.model.state_dict())
                # torch.save(self.model.state_dict(), './sep/weight/' + args.predict + '/best_model_' + args.model_name + '.pth') 
                torch.save(self.model.state_dict(), './weight/best_model_' + args.model_name + '.pth') 
                print('Model Saved.')

            return epoch_loss, epoch_acc     

TrainModel(model, criterion=criterion, optimizer=optimizer,trainloader=train_loader,valloader=val_loader,num_epochs=EPOCH)    
# writer.close() ## 학습이 종료된 후에 반드시 선언할 것!

## Test part ##
def test(model,testloader,criterion):
    model.eval()
    test_loss = 0
    correct = 0
    # total = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
        print('test | Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

checkpoint = torch.load('./weight/best_model_' + args.model_name + '.pth')
model.load_state_dict(checkpoint)   
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)     
test(model,test_loader,criterion)