import cv2
import numpy as np
import math
import os

# 强制切换到exercise文件夹，解决路径问题
#os.chdir("/home/hhhkinggoder1/cv-course/exercise")

#  1. 读入彩色图像 
img = cv2.imread("two.jpg")
if img is None:
    raise ValueError("无法读取图片，请检查路径")
h, w = img.shape[:2]
print(f"原图尺寸: {w}x{h}")

# 2. 转换到YCbCr色彩空间 
img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
Y, Cr, Cb = cv2.split(img_ycrcb)

#  3. 对Cb、Cr通道进行下采样（2倍下采样） 
Cr_down = Cr[::2, ::2]
print(f"下采样后Cb/Cr尺寸: {Cb_down.shape[1]}x{Cb_down.shape[0]}")

#  4. 插值恢复原尺寸（双线性插值） 
Cb_up = cv2.resize(Cb_down, (w, h), interpolation=cv2.INTER_LINEAR)
Cr_up = cv2.resize(Cr_down, (w, h), interpolation=cv2.INTER_LINEAR)

# 5. 与原Y通道重建图像
img_ycrcb_recon = cv2.merge((Y, Cr_up, Cb_up))
img_rgb_recon = cv2.cvtColor(img_ycrcb_recon, cv2.COLOR_YCrCb2BGR)

# 6. 计算PSNR 
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

psnr_value = calculate_psnr(img, img_rgb_recon)
print(f"PSNR值: {psnr_value:.2f} dB")

# - 7. 保存结果（避免WSL显示问题） 
cv2.imwrite("original.jpg", img)
cv2.imwrite("reconstructed.jpg", img_rgb_recon)
print("已保存原图和重建图到exercise文件夹")
