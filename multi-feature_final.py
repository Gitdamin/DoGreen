# multi feature train code
# head pose + helmet detect

import torch
import torch.nn as nn
import numpy as np
import os, glob, random
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.models as models
import time, copy
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

from data_loader import CustomDataset

# Data load!
data_dir = '/home/ydm/4-2/project/dataset/multi-feature'
# print('Folders :', os.listdir(data_dir))
# print("****************************************")
classes = os.listdir(data_dir + "/train")
print('classes :', classes)


train_transform = transforms.Compose([transforms.Resize((64,64)),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
test_transform = transforms.Compose([transforms.Resize((64,64)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# fix seed
def seed_everything(seed):
    torch.manual_seed(seed) # torch를 거치는 모든 난수들의 생성순서를 고정한다
    torch.cuda.manual_seed(seed) # cuda를 사용하는 메소드들의 난수시드는 따로 고정해줘야한다 
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True # 딥러닝에 특화된 CuDNN의 난수시드도 고정 
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed) # numpy를 사용할 경우 고정
    random.seed(seed) # 파이썬 자체 모듈 random 모듈의 시드 고정

seed_everything(42)


dataset = ImageFolder(data_dir + '/train', transform = train_transform)
total = len(dataset)
print('Size of all dataset :', total)

# 임시
train_num = int(total*0.7)
val_num = int(total*0.2)
test_num = total - train_num - val_num
check_sum = train_num + val_num + test_num

assert total == check_sum, "# of total dataset must be same!"

train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_num, val_num, test_num])    
print(len(train_set), len(val_set), len(test_set))  

# TEST
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

# model = models.resnet18(pretrained=False)
# model.fc = Identity()
# x = torch.randn(1, 3, 224, 224)
# output = model(x)
# print(model)
# print(output.shape)
# exit()

# model = models.resnet18(pretrained=False)
# model = nn.Sequential(*list(model.children())[:-1])
# x = torch.randn(1, 3, 224, 224)
# output = model(x)
# print(model)
# print(output.shape)

# model = models.resnet50(pretrained=True)
# model = nn.Sequential(*list(model.children())[:-2])
# print(model)

# out = model(torch.randn(1, 3, 64, 64))
# print(out.size())
# exit()
# t = []
# test = out.size()
# for i in test:
#     if i > 10:
#         pass
#     else:
#         i = 100
#     t.append(i)
# final = torch.tensor(t)
# print(final)
# exit()

batch_size = 16
# EDIT
# train_data_set = CustomDataset(train_set, classes, train_mode=True) 
train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_set, batch_size, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_set, batch_size, num_workers=2, pin_memory=True)


def conv2d(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=True):
    """`same` convolution with LeakyReLU, i.e. output shape equals input shape.
  Args:
    in_planes (int): The number of input feature maps.
    out_planes (int): The number of output feature maps.
    kernel_size (int): The filter size.
    dilation (int): The filter dilation factor.
    stride (int): The filter stride.
  """
    # compute new filter size after dilation
    # and necessary padding for `same` output size
    dilated_kernel_size = (kernel_size - 1) * (dilation - 1) + kernel_size
    same_padding = (dilated_kernel_size - 1) // 2

    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=same_padding,
            dilation=dilation,
            bias=bias,
        ),
        nn.LeakyReLU(0.1, inplace=True),
    )

class ImageEncoder(nn.Module):
    def __init__(self, z_dim): # z_dim -> output dim
        super(ImageEncoder,self).__init__()
        
        self.model = models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-2]) # delete fc layer
        
        # model.fc=nn.Linear(model.fc.in_features, 1024, bias=True)
        # self.model = models.vgg16(pretrained=True)
       
        self.helmet_feature = Helmet(z_dim)
        self.headpose_feature = HeadPose(z_dim)

        
    def forward(self, x): # x = input data
            out = self.model(x)
            
            helmet_out = self.helmet_feature(out)
            headpose_out = self.headpose_feature(out)
            
            helmet_out = F.softmax(helmet_out)
            headpose_out = F.softmax(headpose_out)
            
            return helmet_out, headpose_out

class Helmet(nn.Module):
    def __init__(self, z_dim): 
        super(Helmet,self).__init__()

        self.img_conv1 = conv2d(z_dim, 512, kernel_size=1, stride=1)
        # self.img_conv2 = conv2d(512, 1024, kernel_size=1, stride=1)

        self.linear = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(512, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 2),
        )
       
        
    def forward(self,x): 

            out_img_conv1 = self.img_conv1(x)
            # print(out_img_conv1.size())
            # out_img_conv2 = self.img_conv2(out_img_conv1)
            out = self.linear(out_img_conv1)
            return out 

class HeadPose(nn.Module):
    def __init__(self, z_dim):
        super(HeadPose,self).__init__()
       
        self.img_conv1 = conv2d(z_dim, 512, kernel_size=1, stride=1)
        # self.img_conv2 = conv2d(512, 1024, kernel_size=1, stride=1)
        # self.img_conv2 = conv2d(512, 1024, kernel_size=2, stride=2)

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(512, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 2),
        )
        
    def forward(self,x): # x = extracted feature

            out_img_conv1 = self.img_conv1(x)
            # out_img_conv2 = self.img_conv2(out_img_conv1)
            out = self.linear(out_img_conv1)
            return out 

    
model = ImageEncoder(z_dim = 2048) # z_dim : extracted feature dim. from backbone net.

def preproc(label):
    hel = []
    head = []

    for i in label:
        if i == 0 or i == 2:
            i = 0
        else:
            i = 1
        hel.append(i)
    f_hel = torch.tensor(hel)

    for j in label:
        if j == 0 or j == 3:
            j = 0
        else:
            j = 1
        head.append(j)
    f_head = torch.tensor(head)
    
    return f_hel, f_head



# HYPER PARAM.
# lr = 0.1
# lr = 0.0001
num_epochs = 100
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
                       
class TrainModel():
    def __init__(self, model, device, criterion, optimizer, trainloader, valloader, num_epochs=10):
        
        self.device = device
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.trainloader = trainloader
        self.valloader = valloader
        self.num_epochs = num_epochs
        self.best_acc_wts = copy.deepcopy(self.model.state_dict())
        self.best_acc =0.0

        print('## Start learning!! ##')
        for epoch in range(1, self.num_epochs+1):
            epoch_loss, epoch_acc = self.train()
            if epoch % 1 == 0 :
                print('Epoch {}/{}'.format(epoch, self.num_epochs))
                print('train | Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
            epoch_acc, best_acc = self.val()
            if epoch % 1 == 0 :
                print('val | Acc: {:.4f}'.format(epoch_acc))
        print('best Acc : ', best_acc)
        model.load_state_dict(self.best_acc_wts)
        
    def train(self):
        self.model.train()
        train_loss = 0
        correct_hel = 0
        correct_head = 0
        
        for _, (input_img, targets) in enumerate(self.trainloader):
            hel_l, head_l = preproc(targets)
            # print(hel_l, head_l)
            # exit()
            hel_l = hel_l.to(self.device)
            head_l = head_l.to(self.device)

            input_img = input_img.to(self.device)
           
            self.optimizer.zero_grad()
            hel_out, head_out = self.model(input_img)
            loss_hel = self.criterion(hel_out, hel_l)
            loss_head = self.criterion(head_out, head_l)
            loss = loss_hel + loss_hel

            loss.backward()
            self.optimizer.step()

            train_loss += loss.data.cpu().numpy()

            pred_hel = hel_out.argmax(1, keepdim=True)
            pred_head = head_out.argmax(1, keepdim=True)
           
            correct_hel += pred_hel.eq(hel_l.view_as(pred_hel)).sum().item()
            correct_head += pred_head.eq(head_l.view_as(pred_head)).sum().item()
            
        epoch_loss = train_loss / len(self.trainloader.dataset)
        epoch_acc = (correct_hel+correct_head) / (len(self.trainloader.dataset)*2)

        return epoch_loss, epoch_acc
    
    def val(self):
        self.model.eval()
        correct_hel = 0
        correct_head = 0
        
        with torch.no_grad():
            for _, (input_img, targets) in enumerate(self.valloader):
                hel_l, head_l = preproc(targets)

                hel_l = hel_l.to(self.device)
                head_l = head_l.to(self.device)

                input_img = input_img.to(self.device)
            
                self.optimizer.zero_grad()
                hel_out, head_out = self.model(input_img)
               
                pred_hel = hel_out.argmax(1, keepdim=True)
                pred_head = head_out.argmax(1, keepdim=True)
            
                correct_hel += pred_hel.eq(hel_l.view_as(pred_hel)).sum().item()
                correct_head += pred_head.eq(head_l.view_as(pred_head)).sum().item()
                
            epoch_acc = (correct_hel+correct_head) / (len(self.valloader.dataset)*2)
                        
            if epoch_acc >= self.best_acc:
                self.best_acc = epoch_acc
                self.best_acc_wts = copy.deepcopy(self.model.state_dict())

            return epoch_acc, self.best_acc       
 
# TrainModel(model, device, criterion=criterion, optimizer=optimizer, trainloader=train_loader, valloader=val_loader, num_epochs=num_epochs)   

PATH = './weights/'

# torch.save(model, PATH + 'multi_feature_resnet18.pt') 

def test():
# load pretrained model
    # Edit path
    checkpoint = torch.load('/home/damin/project/classification/weights/best_model_resnet50.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)   
    model.eval()
    model.to(device)  

    test_loss = 0
    correct = 0
    # total = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                test_loss += nn.functional.cross_entropy(outputs, targets,reduction='sum').item()
                pred = outputs.max(1, keepdim=True)[1]
                correct += pred.eq(targets.view_as(pred)).sum().item()
               
        epoch_loss = test_loss /len(test_loader.dataset)
        epoch_acc = correct / len(test_loader.dataset)

        print("## Test Results!! ##")
        print('test | Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))