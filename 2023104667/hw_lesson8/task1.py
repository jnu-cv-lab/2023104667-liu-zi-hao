# 1. 导入需要的库
from sklearn.datasets import load_digits  # 加载手写数字数据集
import matplotlib.pyplot as plt           # 用于图像可视化
import numpy as np                        # 数值计算

# 2. 加载手写数字数据集
digits = load_digits()

# ==================== 任务1：数据探索 ====================
# 3. 查看数据集中图像的数量
image_count = digits.images.shape[0]
print(f"数据集中总图像数量：{image_count} 张")

# 4. 查看每张图像的大小 (8x8 灰度图)
image_shape = digits.images.shape[1:]
print(f"单张图像的尺寸：{image_shape[0]} × {image_shape[1]} 像素")

# 5. 查看特征向量维度 (8x8=64维)
feature_dim = digits.data.shape[1]
print(f"每张图像展开后的特征向量维度：{feature_dim} 维")

# 6. 查看类别标签 (0~9)
labels = np.unique(digits.target)
print(f"数据集中的所有类别标签：{labels}")

# 7. 查看标签的数量统计（每个数字有多少张样本）
label_counts = np.bincount(digits.target)
print("每个类别对应的样本数量：")
for num, count in enumerate(label_counts):
    print(f"数字 {num}：{count} 张")

# ==================== 可视化样本图像 ====================
# 8. 显示前16张样本图像及其真实标签
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 确保数字正常显示
plt.rcParams['axes.unicode_minus'] = False        # 解决负号显示问题

plt.figure(figsize=(8, 8))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(digits.images[i], cmap='gray')
    plt.title(f"Label: {digits.target[i]}")  # 用英文"Label:"替代中文，彻底避免乱码
    plt.axis('off')

plt.tight_layout()
plt.show()