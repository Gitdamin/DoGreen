
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
import argparse
import numpy as np
from typing import Any, List, Optional, Union

from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from torchvision.models.mobilenetv2 import InvertedResidual, ConvBNReLU, MobileNetV2, model_urls
from torch.quantization import QuantStub, DeQuantStub, fuse_modules
# from utils import _fuse_modules, _replace_relu, quantize_model

parser = argparse.ArgumentParser(description='kind of train/model name')
# parser.add_argument("--predict", type=str, default="head", help="head/helmet")
parser.add_argument("--model_name", type=str, default="res18", help="res*/VGG*")
args = parser.parse_args()

def _replace_relu(module: nn.Module) -> None:
    reassign = {}
    for name, mod in module.named_children():
        _replace_relu(mod)
        # Checking for explicit type instead of instance
        # as we only want to replace modules of the exact type
        # not inherited classes
        if type(mod) is nn.ReLU or type(mod) is nn.ReLU6:
            reassign[name] = nn.ReLU(inplace=False)

    for key, value in reassign.items():
        module._modules[key] = value

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

data_dir = '/home/ydm/4-2/project/dataset/scene/dataset'
# print('Folders :', os.listdir(data_dir))
classes = os.listdir(data_dir)
print('Classes :', classes)
# exit()

train_transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
test_transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


dataset = ImageFolder(data_dir, transform = train_transform)
total = len(dataset)
print('Size of all dataset :', total)
train_num = int(total*0.8)
val_num = int(total*0.1)
test_num = total - train_num - val_num
total_num = train_num + val_num + test_num

assert total == total_num, "# of total dataset must be same!"

train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_num, val_num, test_num])    
print(len(train_set), len(val_set), len(test_set))  

batch_size = 16 # Heavy model..
train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_set, batch_size, num_workers=2, pin_memory=True)

test_dir = '/home/ydm/4-2/project/dataset/scene/test'
test_set = ImageFolder(test_dir, transform = test_transform)
print('Size of test dataset :', len(test_set))
# exit()
test_loader = DataLoader(test_set, 1, num_workers=2, pin_memory=True)

# writer = SummaryWriter('logs/res101/')

if 'res' in args.model_name:
    if '18' in args.model_name:
        print("We train res18 model!")
        model = models.resnet18(pretrained=True)

    elif '34' in args.model_name:
        print("We train res34 model!")
        model = models.resnet34(pretrained=True)

    elif '50' in args.model_name:
        print("We train res50 model!")
        model = models.resnet50(pretrained=True)

    elif '101' in args.model_name:
        print("We train res101 model!")
        model = models.resnet101(pretrained=True)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)

elif 'VGG' in args.model_name: # GPU error
    print("We train VGG16 model!")
    model = models.vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(in_features=4096, out_features=2)
    
elif 'Mobile' in args.model_name:
    print("We train MobileNetV2 model!")
    model = models.mobilenet_v2(pretrained=True)
    # model = models.mobilenet_v2(pretrained=True, progress=True, quantize=True)
    model.classifier = nn.Linear(in_features=1280, out_features=2)

elif 'google' in args.model_name:
    print("We train GoogleNet model!")
    model = models.googlenet(pretrained=True)
    model.fc = nn.Linear(in_features=1024, out_features=2)


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model.to(device)
# summary(model, (3, 224, 224))
# exit()


# binary classification
# model.fc = nn.Sequential(
#                nn.Linear(2048, 1024),
#                nn.ReLU(inplace=True),
#                nn.Linear(1024, 128),
#                nn.ReLU(inplace=True),
#                nn.Linear(128, 4))

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model.to(device)
# summary(model, (3, 64, 64))
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
                torch.save(self.model.state_dict(), './scene/weight/best_model_' + args.model_name + '.pth') 
                print('Model Saved.')

            return epoch_loss, epoch_acc     

# TrainModel(model, criterion=criterion, optimizer=optimizer,trainloader=train_loader,valloader=val_loader,num_epochs=EPOCH)    
# writer.close() ## 학습이 종료된 후에 반드시 선언할 것!

def calibrate_model(model, loader, device=torch.device("cpu:0")):
    print("Calibration!")

    model.to(device)
    model.eval()
    # print(model)
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        _ = model(inputs)


def measure_inference_latency(model,
                              device,
                              input_size=(1, 101, 40),
                              num_samples=100,
                              num_warmups=10):

    model.to(device)
    model.eval()

    x = torch.rand(size=input_size).to(device)

    with torch.no_grad():
        for _ in range(num_warmups):
            _ = model(x)
    torch.cuda.synchronize()

    with torch.no_grad():
        start_time = time.time()
        for _ in range(num_samples):
            _ = model(x)
            torch.cuda.synchronize()
        end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_ave = elapsed_time / num_samples

    return elapsed_time_ave


class QuantizableInvertedResidual(InvertedResidual):
    def __init__(self, *args, **kwargs):
        super(QuantizableInvertedResidual, self).__init__(*args, **kwargs)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.use_res_connect:
            return self.skip_add.add(x, self.conv(x))
        else:
            return self.conv(x)

    def fuse_model(self):
        for idx in range(len(self.conv)):
            if type(self.conv[idx]) == nn.Conv2d:
                fuse_modules(self.conv, [str(idx), str(idx + 1)], inplace=True)

class QuantizableMobileNetV2(MobileNetV2):
    def __init__(self, *args, **kwargs):
        """
        MobileNet V2 main class

        Args:
           Inherits args from floating point MobileNetV2
        """
        super(QuantizableMobileNetV2, self).__init__(*args, **kwargs)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self._forward_impl(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNReLU:
                fuse_modules(m, ['0', '1', '2'], inplace=True)
            if type(m) == QuantizableInvertedResidual:
                m.fuse_model()


class QuantizedModel(nn.Module):
    def __init__(self, model_fp32):
        super(QuantizedModel, self).__init__()

        self.quant = torch.quantization.QuantStub()
        self.model_fp32 = model_fp32
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model_fp32(x)
        x = self.dequant(x)
        return x

# CPU ver.
def quantize_evaluate(model,testloader,criterion):

    start = time.time()
    cuda_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu:0")
  
    model_fp32 = copy.deepcopy(model)
   
    quantized_model = QuantizableMobileNetV2(block=QuantizableInvertedResidual) # (model_fp32=model_fp32) #  # 
    _replace_relu(quantized_model)

    quantized_model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack') # fbgemm, qnnpack
    # torch.backends.quantized.engine = 'qnnpack'
    quantized_model = torch.quantization.prepare(quantized_model, inplace=True)
    calibrate_model(model=quantized_model, loader=test_loader, device = cpu_device)
    quantized_model = quantized_model.to(cpu_device)
    quantized_model_int8 = torch.quantization.convert(quantized_model.eval(), inplace=True)
    # print(quantized_model_int8)
    # exit()
    # test_loss = 0
    correct = 0

    with torch.no_grad():
        for _, (inputs, targets) in enumerate(testloader):
                inputs = inputs.to(cpu_device)
                targets = targets.to(cpu_device)
                outputs = quantized_model_int8(inputs)
                # test_loss += nn.functional.cross_entropy(outputs, targets,reduction='sum').item()
                pred = outputs.max(1, keepdim=True)[1]
                correct += pred.eq(targets.view_as(pred)).sum().item()
               
        # epoch_loss = test_loss /len(testloader.dataset)
        epoch_acc = correct / len(testloader.dataset)
        print("## Test Results!! ##")
        print("total dataset :", len(testloader.dataset))
        print("Correct :", correct)
        print('Quantize test | Acc: {:.4f}'.format(epoch_acc))


## Test part ##
def test(model,testloader,criterion):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)   
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
        print("total dataset :", len(testloader.dataset))
        print("Correct :", correct)
        print('test | Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

checkpoint = torch.load('./scene/weight/best_model_' + args.model_name + '.pth')
model.load_state_dict(checkpoint)   

quantize_evaluate(model,test_loader,criterion)
print('#'*20)
# test(model,test_loader,criterion)