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





def train_model(train_loader,valid_loader,model,optimizer,loss_fn,num_epoch=40):
    best_score = 0
    device = torch.device('cuda')
    model = model.to(device)
    for epoch in range(num_epoch):
        model.train()
        running_loss=0.0
        for input,label in train_loader:
            input,label=input.to(device),label.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                unet_output=Unet_model(input)
                probabilities = torch.softmax(unet_output, dim=1)
                attention_map,_=torch.max(probabilities,dim=1,keepdim=True)
            combined_input=(attention_map+1)*input
            output=model(combined_input)
            loss=loss_fn(output,label)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()*input.size(0)
        epoch_loss=running_loss/len(train_loader.dataset)
        print(f"Epoch [{epoch + 1}/{num_epoch}], Loss: {epoch_loss:.4f}")
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs,labels=inputs.to(device),labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Validation Accuracy: {100 * correct / total:.2f}%")
        score=100*correct/total
        patience=5
        if score > best_score:
            best_score=score
            torch.save(model.state_dict(), 'best_combined_model_DE_new.pth')
            print(f"Saved best model with accuracy: {score:.2f}%")
            epochs_no_improve = 0  # 重置计数器，因为验证集准确率有所提高
        else:
            epochs_no_improve += 1
        if epochs_no_improve>=patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

# 开始训练
if __name__ == '__main__':

    DataTransform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop((256), scale=(0.7, 1), ratio=(0.8, 1.2)),
        transforms.ToTensor()
    ])

    data_dir = r"C:\Users\apoll\Desktop\university\third\CV\Project_orient\1_MSCMR_orient\LGE_adjusted_output_2D"

    dataset = datasets.ImageFolder(root=data_dir, transform=DataTransform)
    train_size = int(0.7 * len(dataset))
    valid_size = int(0.15 * len(dataset))
    test_size = int(len(dataset) - train_size - valid_size)
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                               [train_size, valid_size, test_size])
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    Unet_model = Unet(input_channel=1, output_channel=256)
    Unet_model.load_state_dict(torch.load('best_unet_model_DE.pth'))
    device = torch.device('cuda')
    Unet_model = Unet_model.to(device)
    Unet_model.eval()
    for parm in Unet_model.parameters():
        parm.requires_grad = False
    model = Train_CNN_Model(num_classes=8)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    train_model( train_loader, valid_loader, model, optimizer,loss_fn, num_epoch=40)
    torch.save(model.state_dict(), 'combined_model_DE_new.pth')