from sklearn.datasets import load_digits

# 加载数据集
digits = load_digits()

# 原始图像：8×8 二维矩阵
image = digits.images[0]
print("一张 8×8 原始图像（二维矩阵）：")
print(image)
print("图像形状：", image.shape)

print("\n========================================")

# 转换为 64 维特征向量（模型可直接使用）
feature_vector = digits.data[0]
print("转换后的 64 维特征向量（一维）：")
print(feature_vector)
print("向量形状：", feature_vector.shape)