# auto-orient
这个工具利用两种神经网络传统CNN以及复合UNET-CNN模型，以MSCMR作为训练集，实现了多模态的心脏图像纠正。利用注意力机制实现了迁移学习，从单一模态迁移到多模态中。并实现了一个简易的GUI界面来简化工具的使用。
将2维图像四个角分别命名为1234，则每幅2维图形共有八种不同的形式。这对于人工检查抑或后续处理均较为不利。这个工具基于此，对八种不同的图像实现纠正，将其还原回统一样式。
# 大致流程：
## 预处理：
以100%，80%，60%进行截断处理，并通过随机旋转，随机旋转的形式增强泛化程度。利用直方图均衡化加强图像。
## 注意力：
使用一个预训练的UNET网络，获得注意力图谱。使用了注意力图谱的CNN网络虽然在单模态上表现与基础CNN模型相近，但是多模态迁移上会优于传统CNN模型
# 数据集：
MSCMR orient：链接: https://pan.baidu.com/s/1cE5i68YUNhXrzUpldTV6ow 提取码: mj6p
# 引用
这个项目源自复旦大学ZMIC实验室庄吓海教授。根据论文Recognition and Standardization of Cardiac MRI Orientation via Multi-tasking Learning and Deep Neural Networks完成。该论文原github网址为https://github.com/BWGZK/Orientation-Adjust-Tool。在论文中提到了利用注意力机制的CNN模型。但是出于某些原因原github只完成了基础的cnn模型。我在该模型基础上利用unet作为注意力机制，进一步提示了模型的迁移能力。
