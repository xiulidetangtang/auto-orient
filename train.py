import torch.utils.data
from torchvision import transforms
import torch.nn as nn
import torch
import os
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim

DataTransform=transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop((256),scale=(0.7,1),ratio=(0.8,1.2)),
    transforms.ToTensor()
])

data_dir=r"C:\Users\apoll\Desktop\university\third\CV\Project_orient\1 MSCMR orient\T2_adjusted_output_2D"

dataset=datasets.ImageFolder(root=data_dir,transform=DataTransform)
train_size=int(0.7*len(dataset))
valid_size=int(0.15*len(dataset))
test_size=int(len(dataset)-train_size-valid_size)
train_dataset,valid_dataset,test_dataset=torch.utils.data.random_split(dataset,[train_size,valid_size,test_size])
batch_size=8
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
valid_loader=DataLoader(valid_dataset,batch_size=batch_size,shuffle=False)
test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

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

model=Train_CNN_Model(num_classes=8)
loss_fn=nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.01)

def train_model(train_loader,valid_loader,model,optimizer,loss_fn,num_epoch=40):
    for epoch in range(num_epoch):
        model.train()
        running_loss=0.0
        for input,label in train_loader:
            optimizer.zero_grad()
            output=model(input)
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
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Validation Accuracy: {100 * correct / total:.2f}%")

# 开始训练

train_model( train_loader, valid_loader, model, optimizer,loss_fn, num_epoch=40)
torch.save(model.state_dict(), 'model_T2_final.pth')