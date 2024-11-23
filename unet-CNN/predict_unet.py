import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from unet import Unet
# 假设你已经定义了Unet类并加载了训练好的模型
# 加载训练好的Unet模型
unet_model = Unet(input_channel=1, output_channel=256)
unet_model.load_state_dict(torch.load('best_unet_model_C0.pth'))  # 加载已保存的模型权重
unet_model.eval()  # 进入评估模式
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
unet_model = unet_model.to(device)

# 假设我们有一张灰度图像输入
img_path = r"C:\Users\apoll\Desktop\university\third\CV\Project_orient\2 MSCMR seg\train_data\C0_2D\010\subject3_C0.nii.gzslice6.png" # 替换为你的图像路径
img = Image.open(img_path).convert('L')  # 转为灰度图像
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()  # 转为Tensor  # 归一化
])

input_img = transform(img).unsqueeze(0).to(device)  # 添加batch维度，并移动到GPU（如果可用）

# 执行推理
with torch.no_grad():
    output = unet_model(input_img)  # 得到模型输出
    attention_map = torch.softmax(output, dim=1)  # 通过softmax得到每个类别的概率分布

# 可视化结果 - 使用最大概率类别生成注意力图
# 选取注意力图中每个像素对应的最大概率类别
attention_map_np = attention_map.cpu().numpy()  # 转换为numpy数组，便于可视化
attention_map_np=attention_map_np*50
predicted_labels = np.argmax(attention_map_np, axis=1)  # 获取最大概率的类别

# 显示预测结果
plt.imshow(predicted_labels[0], cmap='gray')
plt.title('Predicted Attention Map')
plt.show()
