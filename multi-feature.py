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

# Data load!
data_dir = ''
# print('Folders :', os.listdir(data_dir))
# print("****************************************")
classes = os.listdir(data_dir + "/train")
print('classes :', classes)


train_transform = transforms.Compose([transforms.Resize((64,64)),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                                transforms.Normalize([0.6255728 , 0.6089708, 0.57252455], [0.18989876, 0.1860236, 0.18625423])])
test_transform = transforms.Compose([transforms.Resize((64,64)),
                                transforms.ToTensor(),
                                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                                transforms.Normalize([0.6456329, 0.6196691, 0.57742184], [0.17780653, 0.17758076, 0.1820412])])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

train_transform = transforms.Compose([transforms.Resize((64,64)),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
test_transform = transforms.Compose([transforms.Resize((64,64)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


dataset = ImageFolder(data_dir + '/train', transform = test_transform)
total = len(dataset)
print('Size of all dataset :', total)
train_num = int(total*0.6)
val_num = int(total*0.2)
test_num = int(total*0.2) +2
total_num = train_num + val_num + test_num

assert total == total_num, "# of total dataset must be same!"

train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_num, val_num, test_num])    
print(len(train_set), len(val_set), len(test_set))  

batch_size = 16
train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_set, batch_size, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_set, batch_size, num_workers=2, pin_memory=True)

## reference
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
        
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
        
#         self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
#         self.drop = nn.Dropout2d(0.25)
#         self.fc2 = nn.Linear(in_features=600, out_features=120)
#         self.fc3 = nn.Linear(in_features=120, out_features=10)
        
#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = out.view(out.size(0), -1)
#         out = self.fc1(out)
#         out = self.drop(out)
#         out = self.fc2(out)
#         out = self.fc3(out)
        
#         return out

# EDIT
class ImageEncoder(nn.Module):
    def __init__(self, z_dim): # z_dim -> output dim
        super(ImageEncoder,self).__init__()
        
        self.model = models.resnet18(pretrained=True)
        # self.model = models.vgg16(pretrained=True)
        # print(self.model)
        # exit()

        num_classes = 2*z_dim
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        # self.model.classifier[6] = nn.Linear(in_features=4096, out_features=2*z_dim)
        # print(self.model)
        # exit()
        
    def forward(self,x): # x = extracted feature
            out = self.model(x)
            out_img_conv1 = self.img_conv1(out)
            out_img_conv2 = self.img_conv2(out_img_conv1)

            return out_img_conv2 

class Helmet(nn.Module):
    def __init__(self, z_dim): # z_dim -> 모델의 깊이로 설정하기..
        super(Helmet,self).__init__()

        self.img_conv1 = conv2d(3, 16, kernel_size=7, stride=2)
        self.img_conv2 = conv2d(16, 32, kernel_size=5, stride=2)

        num_classes = 2*z_dim
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        # self.model.classifier[6] = nn.Linear(in_features=4096, out_features=2*z_dim)
        # print(self.model)
        # exit()
        
    def forward(self,x): # x = extracted feature
            out = self.model(x)
            out_img_conv1 = self.img_conv1(out)
            out_img_conv2 = self.img_conv2(out_img_conv1)

            return out_img_conv2 

class HeadPose(nn.Module):
    def __init__(self, z_dim):
        super(HeadPose,self).__init__()
       
        num_classes = 2*z_dim
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        # self.model.classifier[6] = nn.Linear(in_features=4096, out_features=2*z_dim)
        # print(self.model)
        # exit()
        
    def forward(self,x): # x = extracted feature
            out = self.model(x)
            return out 


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
    
# class ImageEncoder(nn.Module):
#     def __init__(self, z_dim):
#         """
#         Image encoder taken from Making Sense of Vision and Touch
#         """
#         super().__init__()
#         self.z_dim = z_dim

#         self.img_conv1 = conv2d(3, 16, kernel_size=7, stride=2)
#         self.img_conv2 = conv2d(16, 32, kernel_size=5, stride=2)
#         self.img_conv3 = conv2d(32, 64, kernel_size=5, stride=2)
#         self.img_conv4 = conv2d(64, 64, stride=2)
#         self.img_conv5 = conv2d(64, 128, stride=2)
#         self.img_conv6 = conv2d(128, self.z_dim, stride=2)
#         self.img_encoder = nn.Linear(self.z_dim, 2 * self.z_dim)
#         # self.flatten = Flatten()

#         # if initailize_weights:
#         #     init_weights(self.modules())

#     def forward(self, image):
#         # image encoding layers
#         out_img_conv1 = self.img_conv1(image)
#         out_img_conv2 = self.img_conv2(out_img_conv1)
#         out_img_conv3 = self.img_conv3(out_img_conv2)
#         out_img_conv4 = self.img_conv4(out_img_conv3)
#         out_img_conv5 = self.img_conv5(out_img_conv4)
#         out_img_conv6 = self.img_conv6(out_img_conv5)

#         img_out_convs = (
#             out_img_conv1,
#             out_img_conv2,
#             out_img_conv3,
#             out_img_conv4,
#             out_img_conv5,
#             out_img_conv6,
#         )

#         # image embedding parameters
#         flattened = out_img_conv6.reshape(out_img_conv6.size(0), -1)
#         img_out = self.img_encoder(flattened)
#         # print(img_out.shape)
#         # print('--------')
#         # print('img encoder_output')
#         # print(img_out)
#         return img_out
    
            
    
           
class SensorFusion(nn.Module):
   
    def __init__(
        self, device, z_dim=128):
        super().__init__()

        self.z_dim = z_dim
        self.device = device

        # -----------------------
        # Modality Encoders
        # -----------------------
        self.img_encoder = ImgEncoder(self.z_dim)
        self.frc_encoder = ForceEncoder(self.z_dim)
        self.pos_encoder = PosEncoder(self.z_dim)

        # -----------------------
        # modality fusion network
        # -----------------------
        # N Total modalities each 
        self.fusion_fc1 = nn.Sequential(
            nn.Linear(3 * 2 * self.z_dim, 128), nn.LeakyReLU(0.1, inplace=True)
        )
        self.fusion_fc2 = nn.Sequential(
            nn.Linear(self.z_dim, self.z_dim), nn.LeakyReLU(0.1, inplace=True)
        )
        
        # self.Avg = nn.AvgPool1d(9)
        
        self.linear = nn.Sequential(
            nn.Linear(self.z_dim, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(32, 4),
        )

    def forward(self, vis_in, frc_in, pos_in):

        # image = rescaleImage(vis_in)
        # depth = filter_depth(depth_in)

        # Get encoded outputs
        img_out = self.img_encoder(vis_in)
        # print(img_out)
        frc_out = self.frc_encoder(frc_in)
        # print(frc_out)
        pos_out = self.pos_encoder(pos_in)

        # multimodal embedding
        mm_f1 = torch.cat([img_out, frc_out, pos_out], 1).squeeze()
        # print('123')
        # mm_f1 = img_out+frc_out
        mm_f2 = self.fusion_fc1(mm_f1)
        out = self.fusion_fc2(mm_f2)
        out = self.linear(out)
   
        return F.softmax(out)
    
model = SensorFusion(device, z_dim = 128)
# lr = 0.1
lr = 0.0001
num_epochs = 20
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr = lr, momentum=0.9)
# optimizer = optim.Adam(model.parameters(),lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0) # betas=(0.9, 0.999),weight_decay=0.0,)
# optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-2)
optimizer = optim.Adam(model.parameters(),lr=lr)
                       
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
            if epoch % 1 ==0 :
                print('Epoch {}/{}'.format(epoch, self.num_epochs))
                print('train | Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
            epoch_acc, best_acc = self.val()
            if epoch % 1 ==0 :
                print('val | Acc: {:.4f}'.format(epoch_acc))
        print('best Acc : ', best_acc)
        model.load_state_dict(self.best_acc_wts)
        
    def train(self):
        self.model.train()
        train_loss = 0
        correct = 0
        pred = []
        
        for i, (input_img, input_frc, input_pos, targets) in enumerate(self.trainloader):
            targets = targets.to(self.device)
            # print(targets)
            input_img = input_img.to(self.device)
            # print(input_img)
            input_frc = input_frc.unsqueeze(dim = 1)
            input_frc = input_frc.to(self.device, dtype=torch.float)
            
            input_pos = input_pos.unsqueeze(dim = 1)
            input_pos = input_pos.to(self.device, dtype=torch.float)
            # print(input_frc)
            # print(input_pos)
            # exit()
            # print(input_frc.type)
            # exit()
            self.optimizer.zero_grad()
           
            outputs = self.model(input_img, input_frc, input_pos)
            loss = self.criterion(outputs, targets)
            # exit()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.data.cpu().numpy()
            pred = outputs.argmax(1, keepdim=True)
            # print(pred)
            # exit()
            correct += pred.eq(targets.view_as(pred)).sum().item()
            
        epoch_loss = train_loss / len(self.trainloader.dataset)
        epoch_acc = correct / len(self.trainloader.dataset)
        return epoch_loss, epoch_acc 
    
    def val(self):
        self.model.eval()
        val_loss = 0
        correct = 0
        
        with torch.no_grad():
            for _, (input_img, input_frc, input_pos, targets) in enumerate(self.valloader):
                # print(len(val_data))
                
                targets = targets.to(self.device)
                input_img = input_img.to(self.device)
                
                input_frc = input_frc.unsqueeze(dim = 1)
                input_frc = input_frc.to(self.device, dtype=torch.float)
                
                input_pos = input_pos.unsqueeze(dim = 1)
                input_pos = input_pos.to(self.device, dtype=torch.float)
                
                self.optimizer.zero_grad()
                outputs = self.model(input_img, input_frc, input_pos)
                pred = outputs.argmax(1, keepdim=True)
                correct += pred.eq(targets.view_as(pred)).sum().item()
               
            # epoch_loss = val_loss /len(val_data)
            epoch_acc = correct / len(self.valloader.dataset)
                        
            if epoch_acc >= self.best_acc:
                self.best_acc = epoch_acc
                self.best_acc_wts = copy.deepcopy(self.model.state_dict())

            return epoch_acc, self.best_acc       
 
TrainModel(model, device, criterion=criterion, optimizer=optimizer, trainloader=train_loader, valloader=vali_loader, num_epochs=num_epochs)   

PATH = './weights/'

torch.save(model, PATH + 'multimodal_model_RES_3in_2.pt') 