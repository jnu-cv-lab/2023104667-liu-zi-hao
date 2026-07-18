import cv2
import numpy as np
import os
# 切换到代码所在目录（避免路径问题）
# os.chdir("/home/hhhkinggoder1/cv-course/homework")

#  任务1: 使用OpenCV读取一张测试图片 
img = cv2.imread("test.jpg")
if img is None:
    raise ValueError("无法读取图片，请检查路径")

#  任务2: 输出图像基本信息 
h, w, channels = img.shape
dtype = img.dtype
print(f"图像尺寸: 宽度={w}, 高度={h}")
print(f"图像通道数: {channels}")
print(f"像素数据类型: {dtype}")

# 任务3: 显示原图
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # OpenCV BGR转RGB
plt.title("Original Image")
plt.axis("off")
plt.show()

#  任务4: 转换为灰度图并显示 -
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(8, 6))
plt.imshow(gray_img, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")
plt.show()

#  任务5: 保存灰度图为新文件
cv2.imwrite("gray_test.jpg", gray_img)
print("灰度图已保存为: gray_test.jpg")

#  任务6: NumPy简单操作（裁剪左上角区域） 
# 1. 输出某个像素值（以(0,0)为例）
pixel_val = img[0, 0]
print(f"左上角像素值(BGR格式): {pixel_val}")

# 2. 裁剪左上角100x100区域并保存
crop_img = img[0:100, 0:100]
cv2.imwrite("crop_test.jpg", crop_img)
print("左上角裁剪区域已保存为: crop_test.jpg")
