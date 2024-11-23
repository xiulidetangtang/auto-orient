from collections import Counter
import SimpleITK as sitk
from PIL import Image
from torchvision import transforms
import numpy as np
from copy import deepcopy
import os
import sys
import torch
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QVBoxLayout, QWidget, QLabel, QComboBox, \
        QFileDialog, QMessageBox
from model import Train_CNN_Model
from model import Unet
from model import CombinedModel
class ImageProcessor:
    def __init__(self, label_list):
        self.label_list = label_list
        self.result_list = None
        self.original_img = None
    def preprocess_data(self, slice):
        """图像预处理"""
        rows, cols = slice.shape
        equalized_img = self.image_equalization(slice)
        channel_list = [0.6, 0.8, 1.0]
        result = np.zeros((rows, cols, 3))
        flattened_slice = equalized_img.flatten()
        G = max(flattened_slice)
        for idx, i in enumerate(channel_list):
            mask = G * i
            mask_result = np.where((flattened_slice >= mask), mask, flattened_slice).astype(np.uint8)
            result[:, :, idx] = mask_result.reshape(rows, cols)
        return result

    def image_equalization(self, img):
        """图像直方图均衡化"""
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

    def open_Image(self, dir):
        """打开并预处理图像"""
        img = sitk.ReadImage(dir)
        img_array = sitk.GetArrayFromImage(img)
        self.original_img = sitk.ReadImage(dir)

        self.result_list = []
        for i in range(img_array.shape[0]):
            slice = deepcopy(img_array[i, :, :])
            slice_matrix = self.preprocess_data(slice)
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
                self.result_list.append(slice_tensor)
        return self.result_list

    def predict(self, model):
        """使用模型对 result_list 进行预测"""
        if self.result_list is None:
            raise ValueError("图像尚未预处理，请先调用 open_Image")

        result_label_list = []

        for img_nii_gz in self.result_list:
            with torch.no_grad():
                output = model(img_nii_gz)
            label = self.label_list[output.argmax()]
            result_label_list.append(label)
        img_label = Counter(result_label_list).most_common(1)[0][0]
        return img_label

    def transform_nii(self, label, img):
        """根据标签对图像切片进行旋转翻转"""
        if label == '000':
            target = img
        elif label == '001':
            target = np.fliplr(img)
        elif label == '010':
            target = np.flipud(img)
        elif label == '011':
            target = np.fliplr(np.flipud(img))
        elif label == '100':
            target = img.transpose((1, 0))
        elif label == '101':
            target = np.flipud(img.transpose(1, 0))
        elif label == '110':
            target = np.fliplr(img.transpose(1, 0))
        elif label == '111':
            target = np.fliplr(np.flipud(img.transpose(1, 0)))
        return target

    def auto_adjust_and_save(self, label, output_dir):
        """根据标签自动调整图像并保存为 .nii.gz"""
        if self.original_img is None:
            raise ValueError("原始图像未加载")

        img_array = sitk.GetArrayFromImage(self.original_img)
        processed_slices = []

        for i in range(img_array.shape[0]):
            slice = deepcopy(img_array[i, :, :])
            transformed_slice = self.transform_nii(label, slice)
            processed_slices.append(transformed_slice)
        processed_array = np.stack(processed_slices, axis=0)
        processed_img = sitk.GetImageFromArray(processed_array)
        processed_img.SetDirection(self.original_img.GetDirection())
        processed_img.SetOrigin(self.original_img.GetOrigin())
        processed_img.SetSpacing(self.original_img.GetSpacing())
        sitk.WriteImage(processed_img, output_dir)
        print(f"图像已保存至: {output_dir}")

def load_models():
    '''加载模型'''
    global C0_model, LGE_model, T2_model
    global Unet_C0_model, Unet_LGE_model, Unet_T2_model

    C0_model = Train_CNN_Model()
    C0_model.load_state_dict(torch.load('model_C0_final.pth'))
    LGE_model = Train_CNN_Model()
    LGE_model.load_state_dict(torch.load('model_LGE_final.pth'))
    T2_model = Train_CNN_Model()
    T2_model.load_state_dict(torch.load('model_T2_final.pth'))


    Unet_C0_Unet = Unet()
    Unet_C0_CNN = Train_CNN_Model()
    Unet_C0_model = CombinedModel(Unet_C0_Unet, Unet_C0_CNN)
    Unet_C0_model.load_state_dict(torch.load('best_finetune_C0_new.pth'))
    Unet_LGE_Unet = Unet()
    Unet_LGE_CNN = Train_CNN_Model()
    Unet_LGE_model = CombinedModel(Unet_LGE_Unet, Unet_LGE_CNN)
    Unet_LGE_model.load_state_dict(torch.load('best_finetune_DE_new.pth'))
    Unet_T2_Unet = Unet()
    Unet_T2_CNN = Train_CNN_Model()
    Unet_T2_model = CombinedModel(Unet_T2_Unet, Unet_T2_CNN)
    Unet_T2_model.load_state_dict(torch.load('best_finetune_T2_new.pth'))


class ModelSelectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("医学图像预测")
        self.setGeometry(100, 100, 400, 300)
        self.label_list = ['000', '001', '010', '011', '100', '101', '110', '111']
        self.processor = ImageProcessor(self.label_list)
        layout = QVBoxLayout()
        self.model_type_selector = QComboBox(self)
        self.model_type_selector.addItems(["C0", "LGE", "T2"])
        layout.addWidget(QLabel("选择模型类型:"))
        layout.addWidget(self.model_type_selector)
        self.model_selector = QComboBox(self)
        self.model_selector.addItems(["CNN", "Unet-CNN"])
        layout.addWidget(QLabel("选择模型结构:"))
        layout.addWidget(self.model_selector)
        self.image_button = QPushButton("选择图像", self)
        self.image_button.clicked.connect(self.open_image)
        layout.addWidget(self.image_button)
        self.predict_button = QPushButton("开始预测", self)
        self.predict_button.clicked.connect(self.predict)
        layout.addWidget(self.predict_button)
        self.auto_adjust_button = QPushButton("自动调整并下载", self)
        self.auto_adjust_button.clicked.connect(self.auto_adjust_and_download)
        layout.addWidget(self.auto_adjust_button)
        self.result_label = QLabel("预测结果将在此处显示", self)
        layout.addWidget(self.result_label)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.image_path = None

    def open_image(self):
        """打开 .nii.gz 文件并加载医学图像"""
        file_name, _ = QFileDialog.getOpenFileName(self, "打开医学图像文件", "", "NII文件 (*.nii *.nii.gz)")
        if file_name:
            self.image_path = file_name
            # 调用图像处理类中的 open_Image 函数进行预处理
            self.processor.open_Image(self.image_path)
            QMessageBox.information(self, "信息", f"成功打开图像文件: {os.path.basename(file_name)}")

    def predict(self):
        """根据选择的模型和类别进行预测"""
        if not self.image_path:
            QMessageBox.warning(self, "错误", "请先选择输入图像")
            return

        model_type = self.model_type_selector.currentText()
        model_structure = self.model_selector.currentText()

        if model_structure == "CNN":
            if model_type == "C0":
                model = C0_model
            elif model_type == "LGE":
                model = LGE_model
            elif model_type == "T2":
                model = T2_model
        else:
            if model_type == "C0":
                model = Unet_C0_model
            elif model_type == "LGE":
                model = Unet_LGE_model
            elif model_type == "T2":
                model = Unet_T2_model
        final_label = self.processor.predict(model)
        self.result_label.setText(f"预测结果: {final_label}")

    def auto_adjust_and_download(self):
        """根据预测标签调整图像并下载"""
        if not self.image_path:
            QMessageBox.warning(self, "错误", "请先选择输入图像")
            return

        model_type = self.model_type_selector.currentText()
        model_structure = self.model_selector.currentText()

        if model_structure == "CNN":
            if model_type == "C0":
                model = C0_model
            elif model_type == "LGE":
                model = LGE_model
            elif model_type == "T2":
                model = T2_model
        else:
            if model_type == "C0":
                model = Unet_C0_model
            elif model_type == "LGE":
                model = Unet_LGE_model
            elif model_type == "T2":
                model = Unet_T2_model

        final_label = self.processor.predict(model)

        save_file, _ = QFileDialog.getSaveFileName(self, "保存调整后的图像", "", "NII文件 (*.nii.gz)")
        if save_file:

            self.processor.auto_adjust_and_save(final_label, save_file)
            QMessageBox.information(self, "信息", f"图像已保存至: {save_file}")

if __name__ == "__main__":
    load_models()
    app = QApplication(sys.argv)
    window = ModelSelectionGUI()
    window.show()
    sys.exit(app.exec_())
