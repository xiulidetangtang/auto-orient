import sys
import os
from PIL import Image
from collections import Counter
from copy import deepcopy
import numpy as np
import SimpleITK as sitk
import torch
from torchvision import transforms
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QVBoxLayout,QHeaderView, QHBoxLayout, QWidget, QLabel, QComboBox, QFileDialog, QMessageBox,QTableWidget,QTableWidgetItem
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from model import Train_CNN_Model, Unet, CombinedModel  # 确保正确导入模型文件

class ImageProcessor:
    def __init__(self, label_list):
        self.label_list = label_list
        self.result_list = None
        self.original_img = None

    def preprocess_data(self, slice):
        """图像预处理"""
        # 图像均衡化和三通道构建
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
        flattened_img = img.flatten()
        histogram = np.zeros(256, dtype=int)
        for i in flattened_img:
            histogram[i] += 1
        cdf = histogram.cumsum()
        cdf = 255 * cdf / cdf[-1]
        new_img = np.zeros_like(flattened_img)
        for index in range(len(flattened_img)):
            new_img[index] = cdf[flattened_img[index]]
        return new_img.reshape(img.shape)

    def open_image(self, dir):
        """打开并预处理图像"""
        img = sitk.ReadImage(dir)
        img_array = sitk.GetArrayFromImage(img)
        self.original_img = img  # 保持原始 SimpleITK 图像

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
                slice_tensor = DataTransform(img).unsqueeze(0).float()
                self.result_list.append(slice_tensor)
        return self.result_list

    def predict(self, model):
        """使用模型对 result_list 进行预测"""
        if self.result_list is None:
            raise ValueError("图像尚未预处理，请先调用 open_image")

        result_label_list = []
        for img_tensor in self.result_list:
            with torch.no_grad():
                output = model(img_tensor)
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
class MedicalImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("医学图像处理工具")
        self.setGeometry(100, 100, 900, 800)
        self.image_index = 0
        self.image_array = None
        self.default_image_path = "default_preview.png"
        self.instruction_image1 = "instruction1.png"
        self.processor = ImageProcessor(['000', '001', '010', '011', '100', '101', '110', '111'])
        self.model_type = None
        self.model_structure = None
        self.model = None
        self.adjusted_image = None
        self.condition=0
        self.initUI()  # 初始化UI

    def initUI(self):
        # 设置左侧布局（包含各种功能按钮）
        left_layout = QVBoxLayout()
        self.open_button = QPushButton("打开 (Open)")
        self.open_button.clicked.connect(self.open_image)
        left_layout.addWidget(self.open_button)

        self.predict_button = QPushButton("预测 (Predict)")
        self.predict_button.clicked.connect(self.predict)
        left_layout.addWidget(self.predict_button)

        self.adjust_button = QPushButton("调整 (Adjust)")
        self.adjust_button.clicked.connect(self.auto_adjust)
        left_layout.addWidget(self.adjust_button)

        self.preview_button = QPushButton("预览 (Preview)")
        self.preview_button.clicked.connect(self.preview_image)
        left_layout.addWidget(self.preview_button)

        self.original_button = QPushButton("原图 (Original)")
        self.original_button.clicked.connect(self.show_original)
        left_layout.addWidget(self.original_button)

        self.save_button = QPushButton("保存 (Save)")
        self.save_button.clicked.connect(self.download_image)
        left_layout.addWidget(self.save_button)

        left_widget = QWidget()
        left_widget.setLayout(left_layout)

        # 设置顶部导航栏（包含翻页和模型选择器）
        top_layout = QHBoxLayout()
        self.prev_button = QPushButton("上一张")
        self.prev_button.clicked.connect(self.show_prev_image)
        top_layout.addWidget(self.prev_button)

        self.index_label = QLabel("第 0 张 / 共 0 张")
        top_layout.addWidget(self.index_label)

        self.next_button = QPushButton("下一张")
        self.next_button.clicked.connect(self.show_next_image)
        top_layout.addWidget(self.next_button)

        self.model_type_selector = QComboBox(self)
        self.model_type_selector.addItems(["C0", "LGE", "T2"])
        top_layout.addWidget(QLabel("选择模型类型:"))
        top_layout.addWidget(self.model_type_selector)

        self.model_selector = QComboBox(self)
        self.model_selector.addItems(["CNN", "Unet-CNN"])
        top_layout.addWidget(QLabel("选择模型结构:"))
        top_layout.addWidget(self.model_selector)

        top_widget = QWidget()
        top_widget.setLayout(top_layout)

        # 设置中心图片展示区
        self.image_label = QLabel()
        self.image_label.setFixedSize(512, 512)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: lightgray;")  # 默认背景色
        if os.path.exists(self.default_image_path):
            default_pixmap = QPixmap(self.default_image_path).scaled(512, 512, Qt.KeepAspectRatio)
            self.image_label.setPixmap(default_pixmap)
        else:
            self.image_label.setText("No Image")
        # 设置最下方信息框
        self.info_label = QLabel("图像信息: 长度=0, 宽度=0, 总层数=0,目前预测结果：无")
        self.info_label.setAlignment(Qt.AlignCenter)
        # 右布局
        right_layout = QVBoxLayout()

        # 第一张说明图片
        self.instruction_label1 = QLabel()
        self.instruction_label1.setFixedSize(200, 200)  # 调整为适合大小
        if os.path.exists(self.instruction_image1):
            instruction_pixmap1 = QPixmap(self.instruction_image1).scaled(200,200 , Qt.KeepAspectRatio)
            self.instruction_label1.setPixmap(instruction_pixmap1)
        else:
            self.instruction_label1.setText("Instruction 1")

        right_layout.addWidget(self.instruction_label1)

        self.instruction_table = QTableWidget()
        self.instruction_table.setRowCount(8)
        self.instruction_table.setColumnCount(3)
        self.instruction_table.setHorizontalHeaderLabels(["No", "Mark", "Description"])
        for row in range(8):
            self.instruction_table.setRowHeight(row, 60)  # 将行高设置为 60 像素，以适应多行内容


        # 填充表格内容
        data = [
            ["000", "1 2\n3 4", "Target[x,y,z]=Source[x,y,z]"],
            ["001", "2 1\n4 3", "Target[x,y,z]=Source[sx-x,y,z]"],
            ["010", "3 4\n1 2", "Target[x,y,z]=Source[x,sy-y,z]"],
            ["011", "4 3\n2 1", "Target[x,y,z]=Source[sx-x,sy-y,z]"],
            ["100", "1 3\n2 4", "Target[x,y,z]=Source[y,x,z]"],
            ["101", "3 1\n4 2", "Target[x,y,z]=Source[sx-y,x,z]"],
            ["110", "2 4\n1 3", "Target[x,y,z]=Source[y,sy-x,z]"],
            ["111", "4 2\n3 1", "Target[x,y,z]=Source[sx-y,sy-x,z]"]
        ]
        for row, row_data in enumerate(data):
            for col, item in enumerate(row_data):
                table_item = QTableWidgetItem(item)
                table_item.setTextAlignment(Qt.AlignCenter)
                self.instruction_table.setItem(row, col, table_item)

        # 设置固定的表格大小
        self.instruction_table.setFixedSize(200, 200)  # 宽度400像素，高度300像素

        # 禁用表头自动调整
        self.instruction_table.horizontalHeader().setStretchLastSection(False)
        self.instruction_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.instruction_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
        self.instruction_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Fixed)

        # 设置列宽度
        self.instruction_table.setColumnWidth(0, 50)
        self.instruction_table.setColumnWidth(1, 80)
        self.instruction_table.setColumnWidth(2, 300)


        for row, (no, mark, description) in enumerate(data):
            self.instruction_table.setItem(row, 0, QTableWidgetItem(no))
            self.instruction_table.setItem(row, 1, QTableWidgetItem(mark))
            self.instruction_table.setItem(row, 2, QTableWidgetItem(description))

        right_layout.addWidget(self.instruction_table)

        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        # 主布局
        main_layout = QHBoxLayout()
        main_layout.addWidget(left_widget)
        main_layout.addWidget(self.image_label)
        main_layout.addWidget(right_widget)

        # 中心布局
        central_layout = QVBoxLayout()
        central_layout.addWidget(top_widget)
        central_layout.addLayout(main_layout)
        central_layout.addWidget(self.info_label)

        central_widget = QWidget()
        central_widget.setLayout(central_layout)
        self.setCentralWidget(central_widget)

    def open_image(self):
        """打开医学图像并进行预处理"""
        file_name, _ = QFileDialog.getOpenFileName(self, "打开医学图像文件", "", "NII文件 (*.nii *.nii.gz)")
        if file_name:
            # 使用 SimpleITK 读取图像并获取 numpy 数组
            image = sitk.ReadImage(file_name)
            self.image_array = sitk.GetArrayFromImage(image)  # 获取所有切片
            self.image_index = 0  # 重置切片索引

            # 更新最下方信息标签
            self.depth, self.height, self.width = self.image_array.shape
            self.info_label.setText(f"图像信息: 长度={self.width}, 宽度={self.height}, 总层数={self.depth},目前预测结果：无")

            # 确保图像被预处理并存储在 result_list 中
            self.processor.open_image(file_name)
            if not self.processor.result_list:
                QMessageBox.warning(self, "错误", "图像预处理失败，请检查图像格式和预处理流程")
            else:
                QMessageBox.information(self, "信息", f"成功打开图像文件: {os.path.basename(file_name)}")

            # 更新图像显示
            self.update_image_view()
    def normalize_to_uint8(self, array):
        array = (array - np.min(array)) / (np.max(array) - np.min(array))
        return (array * 255).astype(np.uint8)


    def predict(self):
        try:
            """根据选择的模型和类别进行预测"""
            if self.image_array is None or self.image_array.size == 0:
                QMessageBox.warning(self, "错误", "请先选择输入图像")
                return
            model_type = self.model_type_selector.currentText()
            model_structure = self.model_selector.currentText()

            # 根据选择加载模型
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

            self.final_label = self.processor.predict(model)
            self.info_label.setText(f"图像信息: 长度={self.width}, 宽度={self.height}, 总层数={self.depth},目前预测结果：{self.final_label}")
            self.info_label.setAlignment(Qt.AlignCenter)
            QMessageBox.information(self, "预测结果", f"预测标签: {self.final_label}")
        except Exception as e:
            print(e)

    def auto_adjust(self):
        """根据预测标签自动调整图像"""
        try:
            if self.image_array is None or self.image_array.size == 0:
                QMessageBox.warning(self, "错误", "请先选择输入图像")
                return
            if not hasattr(self, 'final_label'):
                QMessageBox.warning(self, "错误", "请先进行预测以获得标签")
                return

            # 使用预测标签对所有切片进行调整
            adjusted_slices = []
            for i in range(self.image_array.shape[0]):
                slice = deepcopy(self.image_array[i, :, :])
                adjusted_slice = self.processor.transform_nii(self.final_label, slice)
                adjusted_slices.append(adjusted_slice)

            # 将所有调整好的切片堆叠为三维图像
            self.adjusted_image = np.stack(adjusted_slices, axis=0)
            QMessageBox.information(self, "信息", "图像已调整")
        except Exception as e:
            print(e)

    def download_image(self):
        """下载调整后的图像"""
        if self.adjusted_image is None or self.adjusted_image.size==0:
            QMessageBox.warning(self, "错误", "请先进行图像调整")
            return

        save_file, _ = QFileDialog.getSaveFileName(self, "保存调整后的图像", "", "NII文件 (*.nii.gz)")
        if save_file:
            processed_img = sitk.GetImageFromArray(self.adjusted_image)
            # 设置图像的方向、原点和分辨率，如果原始图像存在
            if self.processor.original_img is not None:
                processed_img.SetDirection(self.processor.original_img.GetDirection())
                processed_img.SetOrigin(self.processor.original_img.GetOrigin())
                processed_img.SetSpacing(self.processor.original_img.GetSpacing())

            sitk.WriteImage(processed_img, save_file)
            QMessageBox.information(self, "信息", f"图像已保存至: {save_file}")

    def preview_image(self):
        """预览调整后的图像"""
        if self.adjusted_image is not None:
            self.image_index = 0  # 重置切片索引
            self.update_image_view(self.adjusted_image[self.image_index])
            self.update_slice_label()  # 更新标签
            self.condition=1
        else:
            QMessageBox.warning(self, "错误", "没有可预览的调整图像")

    def show_original(self):
        """显示原始图像"""
        if self.image_array is not None:
            self.image_index = 0  # 重置切片索引
            self.update_image_view(self.image_array[self.image_index])
            self.update_slice_label()  # 更新标签
            self.condition=0

    def update_slice_label(self):
        """更新顶端切片信息标签"""
        total_slices = self.image_array.shape[0]
        self.index_label.setText(f"第 {self.image_index + 1} 张 / 共 {total_slices} 张")


    def update_image_view(self, image_slice=None):
        """更新中心图片区域以显示当前切片"""
        if image_slice is None:
            image_slice = self.image_array[self.image_index]

        # 确保图像数据为 uint8 格式
        image_slice = self.normalize_to_uint8(image_slice)
        height, width = image_slice.shape
        q_image = QImage(image_slice.data, width, height, width, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), aspectRatioMode=Qt.KeepAspectRatio))

        # 更新顶端切片信息标签
        total_slices = self.image_array.shape[0]
        self.index_label.setText(f"第 {self.image_index + 1} 张 / 共 {total_slices} 张")

    def show_prev_image(self):
        """显示前一个切片"""
        if self.condition == 0:
            if self.image_array is not None and self.image_index > 0:
                self.image_index -= 1
                self.update_image_view(self.image_array[self.image_index])
        if self.condition == 1:
            if self.adjusted_image is not None and self.image_index >0:
                self.image_index-=1
                self.update_image_view(self.adjusted_image[self.image_index])

    def show_next_image(self):
        """显示下一个切片"""
        if self.condition == 0:
            if self.image_array is not None and self.image_index < self.image_array.shape[0] - 1:
                self.image_index += 1
                self.update_image_view(self.image_array[self.image_index])
        elif self.condition == 1:
            if self.adjusted_image is not None and self.image_index<self.adjusted_image.shape[0]-1:
                self.image_index+=1
                self.update_image_view(self.adjusted_image[self.image_index])

def load_models():
    """加载模型"""
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

if __name__ == "__main__":
    # 加载模型
    load_models()
    app = QApplication(sys.argv)
    viewer = MedicalImageViewer()
    viewer.show()
    sys.exit(app.exec_())