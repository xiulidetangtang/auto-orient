import sys
import os
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QFileDialog, QVBoxLayout, QWidget, QMessageBox
import SimpleITK as sitk


class MedicalImageProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("医学图像处理工具")
        self.setGeometry(100, 100, 400, 200)
        layout = QVBoxLayout()
        self.open_button = QPushButton("打开医学图像 (.nii.gz)", self)
        self.open_button.clicked.connect(self.open_image)
        layout.addWidget(self.open_button)
        self.predict_button = QPushButton("预测图像", self)
        self.predict_button.clicked.connect(self.predict_image)
        layout.addWidget(self.predict_button)
        self.adjust_button = QPushButton("调整图像", self)
        self.adjust_button.clicked.connect(self.adjust_image)
        layout.addWidget(self.adjust_button)
        self.save_button = QPushButton("保存处理后的图像", self)
        self.save_button.clicked.connect(self.save_image)
        layout.addWidget(self.save_button)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.image = None  # 用于存储加载的医学图像
        self.adjusted_image = None  # 用于存储处理后的图像

    def open_image(self):
        """打开 .nii.gz 文件并加载医学图像"""
        file_name, _ = QFileDialog.getOpenFileName(self, "打开医学图像文件", "", "NII文件 (*.nii *.nii.gz)")
        if file_name:
            self.image = sitk.ReadImage(file_name)
            QMessageBox.information(self, "信息", f"成功打开图像文件: {os.path.basename(file_name)}")

    def predict_image(self):
        """模拟医学图像预测功能"""
        if self.image is None:
            QMessageBox.warning(self, "错误", "请先打开医学图像文件")
            return

        # 模拟预测操作（这里可以替换为实际的模型推理）
        QMessageBox.information(self, "预测结果", "图像预测结果：这是一个示例预测")

    def adjust_image(self):
        """对医学图像进行调整（例如旋转）"""
        if self.image is None:
            QMessageBox.warning(self, "错误", "请先打开医学图像文件")
            return

        # 这里进行图像调整操作，示例中进行图像的旋转
        self.adjusted_image = sitk.Flip(self.image, [True, False, False])  # 例如对X轴进行翻转
        QMessageBox.information(self, "调整成功", "图像已调整")

    def save_image(self):
        """保存调整后的医学图像到文件"""
        if self.adjusted_image is None:
            QMessageBox.warning(self, "错误", "没有可保存的图像，请先进行调整")
            return

        save_file_name, _ = QFileDialog.getSaveFileName(self, "保存医学图像", "", "NII文件 (*.nii.gz);;所有文件 (*)")
        if save_file_name:
            sitk.WriteImage(self.adjusted_image, save_file_name)
            QMessageBox.information(self, "保存成功", f"图像已保存为: {save_file_name}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MedicalImageProcessor()
    window.show()
    sys.exit(app.exec_())
