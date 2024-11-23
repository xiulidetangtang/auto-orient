import torch.utils.data
from torch.distributed.tensor.parallel.input_reshard import input_reshard
from torchvision import transforms
import torch.nn as nn
import torch
import os
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import random
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
from unet import Unet
from CNN_module import Train_CNN_Model


DataTransform=transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop((256),scale=(0.7,1),ratio=(0.8,1.2)),
    transforms.ToTensor()
])


data_dir_list=[r"C:\Users\apoll\Desktop\university\third\CV\Project_orient\1_MSCMR_orient\C0_raw_2D",r"C:\Users\apoll\Desktop\university\third\CV\Project_orient\1_MSCMR_orient\LGE_adjusted_output_2D",r"C:\Users\apoll\Desktop\university\third\CV\Project_orient\1_MSCMR_orient\T2_adjusted_output_2D"]
unet_param_list=['best_unet_model_C0.pth','best_unet_model_DE.pth','best_unet_model_T2.pth']
CNN_param_list=['best_combined_model_C0_new.pth','best_combined_model_DE_new.pth','best_combined_model_T2_new.pth']
output_list=['finetune_C0_new.pth','finetune_DE_new.pth','finetune_T2_new.pth']
early_stop_list=['best_finetune_C0_new.pth','best_finetune_DE_new.pth','best_finetune_T2_new.pth']


def train_combined_model(train_loader, valid_loader, model, optimizer,early_stop_dir,output_dir, loss_fn, num_epochs):
    best_score=0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Validation Accuracy: {100 * correct / total:.2f}%")
        score=100*correct/total
        patience=5
        if score > best_score:
            best_score=score
            torch.save(model.state_dict(), early_stop_dir)
            print(f"Saved best model with accuracy: {score:.2f}%")
            epochs_no_improve = 0  # 重置计数器，因为验证集准确率有所提高
        else:
            epochs_no_improve += 1
        if epochs_no_improve>=patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
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
if __name__=='__main__':
    for data_dir,unet_dir,cnn_dir,output_dir,early_stop_dir in zip(data_dir_list,unet_param_list,CNN_param_list,output_list,early_stop_list):
        dataset=datasets.ImageFolder(root=data_dir,transform=DataTransform)
        train_size=int(0.7*len(dataset))
        valid_size=int(0.15*len(dataset))
        test_size=int(len(dataset)-train_size-valid_size)
        train_dataset,valid_dataset,test_dataset=torch.utils.data.random_split(dataset,[train_size,valid_size,test_size])
        batch_size=8
        train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
        valid_loader=DataLoader(valid_dataset,batch_size=batch_size,shuffle=False)
        test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
        unet_model=Unet(input_channel=1,output_channel=8)
        unet_model.load_state_dict(torch.load(unet_dir))
        for parm in unet_model.parameters():
            parm.requires_grad=True
        for param in unet_model.conv1.parameters():
            param.requires_grad = False

        for param in unet_model.conv2.parameters():
            param.requires_grad = False

        for param in unet_model.conv3.parameters():
            param.requires_grad = False
        CNN_model=Train_CNN_Model(num_classes=8)
        CNN_model.load_state_dict((torch.load(cnn_dir)))
        for parm in CNN_model.parameters():
            parm.requires_grad=True

        device = torch.device('cuda')
        combined_model = CombinedModel(unet_model, CNN_model).to(device)
        # 仅训练解码部分：包括 deconv2、deconv1、conv8、conv9
        decoder_params = list(unet_model.deconv2.parameters()) + \
                         list(unet_model.deconv1.parameters()) + \
                         list(unet_model.conv8.parameters()) + \
                         list(unet_model.conv9.parameters()) + \
                         list(unet_model.conv10.parameters())
        # 为 UNet 解码部分设置较小的学习率
        optimizer = optim.SGD([
            {'params': decoder_params, 'lr': 1e-5},  # 仅为解码部分设置较小的学习率
            {'params': CNN_model.parameters(), 'lr': 1e-3}  # 为 CNN 使用较大的学习率
        ])
        loss_fn = nn.CrossEntropyLoss()
        train_combined_model(train_loader, valid_loader, combined_model, optimizer, early_stop_dir,output_dir,loss_fn, num_epochs=40)
        torch.save(combined_model.state_dict(),output_dir)

