import torch.utils.data
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
def Data_transform(img,label):
    if random.random() < 0.5:
        angle = random.uniform(-10, 10)
        img = TF.rotate(img, angle)
        label = TF.rotate(label, angle, interpolation=TF.InterpolationMode.NEAREST)

    i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=(0.7, 1), ratio=(0.8, 1.2))
    img = TF.resized_crop(img, i, j, h, w, size=(256, 256))
    label = TF.resized_crop(label, i, j, h, w, size=(256, 256), interpolation=TF.InterpolationMode.NEAREST)
    img = TF.to_tensor(img)
    label = torch.tensor(np.array(label), dtype=torch.long)  # 确保标签为 long 类型
    return img, label




class promoteDataset(torch.utils.data.Dataset):
    def __init__(self,img_dir,label_dir,transform):
        self.img_folder=sorted([os.path.join(img_dir,folder) for folder in os.listdir(img_dir)])
        self.label_folder=sorted([os.path.join(label_dir,folder) for folder in os.listdir(label_dir)])
        self.img_paths=[]
        self.label_paths=[]
        self.transform=transform

        for img_folder,label_folder in zip(self.img_folder,self.label_folder):
            img_file_list=sorted([os.path.join(img_folder,file) for file in os.listdir(img_folder)])
            label_file_list=sorted([os.path.join(label_folder,file) for file in os.listdir(label_folder)])
            self.img_paths.extend(img_file_list)
            self.label_paths.extend(label_file_list)
    def __len__(self):
        return len(self.img_paths)


    def __getitem__(self, idx):
        # 打开图像并转换为灰度图像
        image = Image.open(self.img_paths[idx]).convert('L')
        label = Image.open(self.label_paths[idx]).convert('L')

        # 将图像和标签进行自定义的 transform（如果有）
        if self.transform:
            image, label = self.transform(image, label)

        # 将标签转换为 NumPy 数组
        label_np = np.array(label)

        # 定义灰度值区间的最小值（5 个区间）
        bins = [0, 52, 103, 154, 205]  # 区间的最小值
        cols,rows=label_np.shape
        flatten_label=label_np.flatten()
        new_label=[]
        for axis in flatten_label:
            if axis<52 and axis >=0:
                new_axis=0
            elif axis >= 52 and axis < 103:
                new_axis = 1
            elif axis >= 103 and axis < 154:
                new_axis = 2
            elif axis >= 154 and axis<205:
                new_axis = 3
            else:new_axis = 4
            new_label.append(new_axis)
        new_label=np.array(new_label)
        new_label=new_label.reshape(cols,rows)


        # 将映射后的 NumPy 数组转换为 PyTorch 的 long 张量
        label_tensor = torch.tensor(new_label, dtype=torch.long)

        return image, label_tensor


img_dir=r"C:\Users\apoll\Desktop\university\third\CV\Project_orient\2 MSCMR seg\train_data\DE_2D"
label_dir=r"C:\Users\apoll\Desktop\university\third\CV\Project_orient\2 MSCMR seg\train_data_manual\DE_2D"
dataset=promoteDataset(img_dir,label_dir,Data_transform)
train_size=int(0.7*len(dataset))
valid_size=int(0.15*len(dataset))
test_size=int(len(dataset)-train_size-valid_size)
train_dataset,valid_dataset,test_dataset=torch.utils.data.random_split(dataset,[train_size,valid_size,test_size])
batch_size=8
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
valid_loader=DataLoader(valid_dataset,batch_size=batch_size,shuffle=False)
test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
epoch_num=40
model=Unet()


optimizer=optim.SGD(model.parameters(),lr=0.01)
loss_fn=nn.CrossEntropyLoss()

def train_model(train_loader,valid_loader,model,optimizer,loss_fn,num_epoch):
    device = torch.device('cuda')
    model = model.to(device)
    best_val_acc = 0
    for epoch in range(epoch_num):
        model.train()
        running_loss=0.0
        for img,label in train_loader:
            img,label=img.to(device),label.to(device)

            optimizer.zero_grad()
            output=model(img)
            loss=loss_fn(output,label)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()*img.size(0)
        epoch_loss=running_loss/len(train_loader.dataset)
        print(f"Epoch [{epoch + 1}/{num_epoch}], Loss: {epoch_loss:.4f}")
        torch.cuda.empty_cache()
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs,labels=inputs.to(device),labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.numel()
                correct += (predicted == labels).sum().item()

        print(f"Validation Accuracy: {100 * correct / total:.2f}%")
        val_accuracy = 100 * correct / total
        patience=5

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), 'best_unet_model_DE.pth')
            print(f"Saved best model with accuracy: {val_accuracy:.2f}%")
            epochs_no_improve = 0  # 重置计数器，因为验证集准确率有所提高
        else:
            epochs_no_improve += 1

        # 如果验证集准确率没有提升并且超过耐心值，停止训练
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
train_model( train_loader, valid_loader, model, optimizer,loss_fn, num_epoch=40)
torch.save(model.state_dict(), 'unet_model_DE.pth')