import torch.utils.data
from torchvision import transforms
import torch.nn as nn
import torch
import os
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim

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
class double_conv2d(nn.Module):
    def __init__(self,input_channel,output_channel,kernel=3,striding=1,padding=1):
        super(double_conv2d,self).__init__()
        self.conv1=nn.Conv2d(input_channel,output_channel,kernel_size=kernel,stride=striding,padding=padding,bias=True)
        self.conv2=nn.Conv2d(output_channel,output_channel,kernel_size=kernel,stride=striding,padding=padding,bias=True)
        self.bn1=nn.BatchNorm2d(output_channel)
        self.bn2=nn.BatchNorm2d(output_channel)
        self.ReLu=nn.ReLU()
    def forward(self,x):
        x=self.ReLu(self.bn1(self.conv1(x)))
        x=self.ReLu(self.bn2(self.conv2(x)))
        return x
class deconv2d_bn(nn.Module):
    def __init__(self,input_channel,output_channel,kernel=2,striding=2):
        super(deconv2d_bn,self).__init__()
        self.deconv=nn.ConvTranspose2d(input_channel,output_channel,kernel_size=kernel,stride=striding)
        self.bn1=nn.BatchNorm2d(output_channel)
        self.ReLu=nn.ReLU()
    def forward(self,x):
        x=self.ReLu(self.bn1(self.deconv(x)))
        return x

class Unet(nn.Module):
    def __init__(self,input_channel=1,output_channel=256):
        super(Unet,self).__init__()
        self.conv1=double_conv2d(input_channel,64)
        self.conv2=double_conv2d(64,128)
        self.conv3=double_conv2d(128,256)
        self.conv8=double_conv2d(256,128)
        self.conv9=double_conv2d(128,64)
        self.conv10=nn.Conv2d(64,5,kernel_size=3,stride=1,padding=1)
        self.deconv2=deconv2d_bn(256,128)
        self.deconv1=deconv2d_bn(128,64)
        self.MaxPool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.SoftMax=nn.Softmax()

    def forward(self, x):
        enc1 = self.conv1(x)
        enc2 = self.conv2(self.MaxPool(enc1))
        enc3 = self.conv3(self.MaxPool(enc2))
        dec2 = self.deconv2(enc3)  # 反卷积
        concat1 = torch.cat([dec2, enc2], dim=1)  # 确保enc2和dec2的特征图大小匹配
        enc4 = self.conv8(concat1)
        dec1 = self.deconv1(enc4)
        concat2 = torch.cat([dec1, enc1], dim=1)  # 确保enc1和dec1的特征图大小匹配
        enc5 = self.conv9(concat2)
        output = self.conv10(enc5)  # 输出层
        return output

class CombinedModel(nn.Module):
    def __init__(self, unet_model, cnn_model):
        super(CombinedModel, self).__init__()
        self.unet = unet_model
        self.cnn = cnn_model

    def forward(self, x):
        unet_output = self.unet(x)
        probabilities = torch.softmax(unet_output, dim=1)
        attention_map, _ = torch.max(probabilities, dim=1, keepdim=True)
        combined_input = (attention_map+1) * x
        output = self.cnn(combined_input)
        return output