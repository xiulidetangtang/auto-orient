import torch.utils.data
from torchvision import transforms
import torch.nn as nn
import torch
class Train_CNN_Model(nn.Module):
    def __init__(self,num_classes=8):
        super(Train_CNN_Model,self).__init__()
        self.conv1=nn.Conv2d(1,32,kernel_size=3,padding=1)
        self.bn1=nn.BatchNorm2d(32)
        self.conv2=nn.Conv2d(32,32,kernel_size=3,padding=1)
        self.bn2=nn.BatchNorm2d(32)
        self.conv3=nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.bn3=nn.BatchNorm2d(64)
        self.maxpool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.avgpool=nn.AvgPool2d(kernel_size=2,stride=2)
        self.flatten=nn.Flatten()
        self.linear1=nn.Linear(64*16*16,64)
        self.linear2=nn.Linear(64,num_classes)
        self.softmax=nn.Softmax()
        self.relu = nn.ReLU()
    def forward(self,x):
        x=self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x=self.maxpool(self.relu(self.bn2(self.conv2(x))))
        x=self.maxpool(self.relu(self.bn3(self.conv3(x))))
        x=self.avgpool(x)
        x=self.flatten(x)
        x=self.linear1(x)
        x=self.linear2(x)
        x=self.softmax(x)
        return x