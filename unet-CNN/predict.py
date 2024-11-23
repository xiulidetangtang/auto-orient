import os
from copy import deepcopy
from tkinter.tix import IMAGE
from PIL import Image
from torchvision import transforms
import SimpleITK as sitk
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import dataloader
from collections import Counter
from CNN_module import Train_CNN_Model
from finetune import output_list
from unet import Unet
import json

# class CombinedModel(nn.Module):
#     def __init__(self, unet_model, cnn_model):
#         super(CombinedModel, self).__init__()
#         self.unet = unet_model
#         self.cnn = cnn_model
#
#     def forward(self, x):
#         unet_output = self.unet(x)
#         probabilities = torch.softmax(unet_output, dim=1)
#         attention_map, _ = torch.max(probabilities, dim=1, keepdim=True)
#         combined_input = attention_map * x
#         output = self.cnn(combined_input)
#         return output
# class CombinedModel(nn.Module):
#     def __init__(self, unet_model, cnn_model):
#         super(CombinedModel, self).__init__()
#         self.unet = unet_model
#         self.cnn = cnn_model
#
#     def forward(self, x):
#         unet_output = self.unet(x)
#         probabilities = torch.softmax(unet_output, dim=1)
#         attention_map, _ = torch.max(probabilities, dim=1, keepdim=True)
#         combined_input = (attention_map+1) * x
#         output = self.cnn(combined_input)
#         return output
# combined_model=load_trained_model('best_unet_model_C0.pth','best_combined_model_C0_new.pth','best_finetune_C0_new.pth','cuda')

# unet_model=Unet(input_channel=1,output_channel=8)
# unet_model.load_state_dict(torch.load('best_unet_model_C0.pth'))
# for parm in unet_model.parameters():
#     parm.requires_grad=True
# CNN_model=Train_CNN_Model(num_classes=8)
# CNN_model.load_state_dict((torch.load('best_combined_model_C0.pth')))
# for parm in CNN_model.parameters():
#     parm.requires_grad=True
# combined_model=CombinedModel(unet_model,CNN_model)
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
DataTransform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop((256), scale=(0.7, 1), ratio=(0.8, 1.2)),
    transforms.ToTensor()
])
CNN_model_C0=Train_CNN_Model(num_classes=8)
CNN_model_C0.load_state_dict((torch.load('best_combined_model_C0_new.pth')))
Unet_C0=Unet(input_channel=1)
Unet_C0.load_state_dict(torch.load('best_unet_model_C0.pth'))
combined_model_C0=CombinedModel(Unet_C0,CNN_model_C0)
combined_model_C0.load_state_dict(torch.load('best_finetune_C0_new.pth'))
CNN_model_LGE=Train_CNN_Model(num_classes=8)
CNN_model_LGE.load_state_dict((torch.load('best_combined_model_DE_new.pth')))
Unet_LGE=Unet(input_channel=1)
Unet_LGE.load_state_dict(torch.load('best_unet_model_DE.pth'))
combined_model_LGE=CombinedModel(Unet_LGE,CNN_model_LGE)
combined_model_LGE.load_state_dict(torch.load('best_finetune_DE_new.pth'))
CNN_model_T2=Train_CNN_Model(num_classes=8)
CNN_model_T2.load_state_dict((torch.load('best_combined_model_T2_new.pth')))
Unet_T2=Unet(input_channel=1)
Unet_T2.load_state_dict(torch.load('best_unet_model_T2.pth'))
combined_model_T2=CombinedModel(Unet_T2,CNN_model_T2)
combined_model_T2.load_state_dict(torch.load('best_finetune_T2_new.pth'))

# def load_trained_model(unet_path, cnn_path, combined_model_path, device):
#     unet_model = Unet(input_channel=1)
#     unet_model.load_state_dict(torch.load(unet_path))
#
#     CNN_model = Train_CNN_Model(num_classes=8)
#     CNN_model.load_state_dict(torch.load(cnn_path))
#
#     combined_model = CombinedModel(unet_model, CNN_model)
#     combined_model.load_state_dict(torch.load(combined_model_path))
#     combined_model.eval()  # 切换到评估模式
#     return combined_model
label_list=['000','001','010','011','100','101','110','111']
def preprocess_data(slice):
    rows, cols = slice.shape
    equalized_img = image_equalization(slice)
    channel_list = [0.6, 0.8, 1.0]
    result = np.zeros((rows, cols, 3))
    flattened_slice = equalized_img.flatten()
    G = max(flattened_slice)
    for idx, i in enumerate(channel_list):
        mask = G * i
        mask_result = np.where((flattened_slice >= mask), mask, flattened_slice).astype(np.uint8)
        result[:, :, idx] = mask_result.reshape(rows, cols)
    return result


def image_equalization(img):
    img = img.astype(np.uint8)
    cols, rows = img.shape
    flattened_img = img.flatten()
    histogram = np.zeros(256, dtype=int)
    for i in flattened_img:
        histogram[i] += 1
    cdf = histogram.cumsum()
    cdf = 255 * cdf / cdf[-1]
    new_img = np.zeros_like(flattened_img)
    for index in range(len(flattened_img)):
        new_img[index] = cdf[flattened_img[index]]
    return new_img.reshape(cols, rows)


def open_Image(dir):
    img = sitk.ReadImage(dir)
    img_array = sitk.GetArrayFromImage(img)
    result_list = []
    for i in range(img_array.shape[0]):
        slice = deepcopy(img_array[i, :, :])
        slice_matrix = preprocess_data(slice)
        for j in range(slice_matrix.shape[2]):
            new_slice = deepcopy(slice_matrix[:, :, j])
            img = Image.fromarray(new_slice)
            DataTransform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.RandomRotation(10),
                transforms.RandomResizedCrop((256), scale=(0.7, 1), ratio=(0.8, 1.2)),
                transforms.ToTensor()
            ])
            slice_tensor = DataTransform(img).unsqueeze(0)
            slice_tensor = slice_tensor.float()
            result_list.append(slice_tensor)
    return result_list


def autoadjusted(img_nii, label):
    img = sitk.ReadImage(img_nii)
    img_array = sitk.GetArrayFromImage(img)
    processed_slice = []
    for i in range(img_array.shape[0]):
        slice = deepcopy(img_array[i, :, :])
        slice = transform_nii(label, slice)
        processed_slice.append(slice)
    processed_array = np.stack(processed_slice, axis=0)
    return processed_array


def transform_nii(type, img):
    if type == '000':
        target = img
    elif type == '001':
        target = np.fliplr(img)
    elif type == '010':
        target = np.flipud(img)
    elif type == '011':
        target = np.fliplr(np.flipud(img))
    elif type == '100':
        target = img.transpose((1, 0))
    elif type == '101':
        target = np.flipud(img.transpose(1, 0))
    elif type == '110':
        target = np.fliplr(img.transpose(1, 0))
    elif type == '111':
        target = np.fliplr(np.flipud(img.transpose(1, 0)))
    else:
        return 0
    return target


def download_img(processed_array, original_img, dir):
    processed_img = sitk.GetImageFromArray(processed_array)
    processed_img.SetDirection(original_img.GetDirection())
    processed_img.SetOrigin(original_img.GetOrigin())
    processed_img.SetSpacing(original_img.GetSpacing())
    sitk.WriteImage(processed_img, dir)


def main(type,dir,model,output_dir):
    results_dict={}
    Correct = 0
    Total = 0
    TP = [0 for i in range(8)]
    FP = [0 for i in range(8)]
    FN = [0 for i in range(8)]
    recall = [0 for _ in range(8)]
    precision = [0 for _ in range(8)]
    F1 = [0 for _ in range(8)]
    for i,real_label in enumerate(label_list):

        folder_path=os.path.join(dir,type,real_label)
        print('working on:',folder_path)
        file_path=[f for f in os.listdir(folder_path) if f.endswith('.nii.gz')]
        for file in file_path:
            file_dir=os.path.join(folder_path,file)
            img_nii_gz_list=open_Image(file_dir)
            origin_img=sitk.ReadImage(file_dir)
            result_label_list = []
            for img_nii_gz in img_nii_gz_list:
                with torch.no_grad():
                    output = model(img_nii_gz)
                label=label_list[output.argmax()]
                result_label_list.append(label)
            img_label=Counter(result_label_list).most_common(1)[0][0]
            print('now dealing with:', file,img_label)
            Total+=1
            if img_label == real_label:
                Correct+=1
                TP[i]+=1
            else:
                FN[i]+=1
                FP[label_list.index(img_label)]+=1

    accuracy = Correct / Total
    for i in range(8):
        recall[i]=TP[i]/(TP[i]+FN[i])
        precision[i]=TP[i]/(TP[i]+FP[i])
        F1[i]=2*precision[i]*recall[i]/(precision[i]+recall[i])

        results_dict[label_list[i]] = {
        "accuracy": accuracy,
        "TP": TP[i],
        "FP": FP[i],
        "FN": FN[i],
        "recall": recall[i],
        "precision": precision[i],
        "F1": F1[i],
        "type": type
    }
    # Save the results dictionary to a JSON file
    with open(output_dir, "w") as json_file:
        json.dump(results_dict, json_file, indent=4)

    print("Results saved to results.json")

        # new_nii_img = autoadjusted(file_dir, img_label)
        # output_dir = os.path.join(dir, 'output_combined', file)
        # download_img(new_nii_img, origin_img, output_dir)
type_list=['C0','LGE','T2']
model_list=[combined_model_C0,combined_model_LGE,combined_model_T2]
output_list=['result_C0.json','result_LGE.json','result_T2.json']
for type,model,output in zip(type_list,model_list,output_list):
    main(type, r"C:\Users\apoll\Desktop\university\third\CV\Project_orient\1_MSCMR_orient",model,output)