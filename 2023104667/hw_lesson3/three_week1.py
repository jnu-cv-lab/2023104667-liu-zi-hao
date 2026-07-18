import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct, idct, fft, ifft
import cv2
import os

# -------------------------- 1. 解决路径问题 --------------------------
os.chdir(os.path.dirname(__file__))
img_path = "ocean.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"当前目录: {os.getcwd()}, 文件夹内文件: {os.listdir('.')}")
    raise ValueError("图片读取失败，请检查文件名是否为ocean.jpg")

# 取图像中间行作为一维测试信号
h, w = img.shape
x = img[h//2, :]
N = len(x)
print(f"照片尺寸：{w}x{h}，取中间行，信号长度N={N}")

# -------------------------- 2. 延拓方式对比 --------------------------
x_dft_ext = np.tile(x, 2)
x_dct_ext = np.concatenate([x[::-1], x])

plt.figure(figsize=(12, 8))
plt.subplot(3,1,1)
plt.plot(np.arange(N), x, "k-", lw=1.5)
plt.title(f"Original Signal (Middle Row of ocean.jpg, N={N})")
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(np.arange(2*N), x_dft_ext, "r-", lw=1.5)
plt.axvline(N-0.5, c="b", ls="--", label="Boundary Jump (DFT)")
plt.title("DFT Periodic Extension")
plt.legend()
plt.grid(True)

plt.subplot(3,1,3)
plt.plot(np.arange(2*N), x_dct_ext, "g-", lw=1.5)
plt.axvline(N-0.5, c="b", ls="--", label="Smooth Mirror (DCT)")
plt.title("DCT Even Symmetric Extension")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("extension_compare.png", dpi=300)
plt.show(block=False)

# -------------------------- 3. 频谱对比（修复stem参数） --------------------------
X_dft = fft(x)
X_dct = dct(x, norm="ortho")
mag_dft = np.abs(X_dft)
mag_dct = np.abs(X_dct)

plt.figure(figsize=(12, 8))
plt.subplot(2,1,1)
# 移除不兼容的use_line_collection参数
plt.stem(np.arange(N//2), mag_dft[:N//2], basefmt=" ")
plt.title("DFT Magnitude Spectrum (Positive Frequencies)")
plt.grid(True)

plt.subplot(2,1,2)
plt.stem(np.arange(N), mag_dct, basefmt=" ")
plt.title("DCT-II Magnitude Spectrum")
plt.grid(True)

plt.tight_layout()
plt.savefig("spectrum_compare.png", dpi=300)
plt.show(block=False)

# -------------------------- 4. 能量集中性量化 --------------------------
E_total = np.sum(x**2)
print(f"\n总能量: {E_total:.2f}")

K_list = [int(N*0.05), int(N*0.1), int(N*0.2), N]
eta_dft, eta_dct = [], []
for K in K_list:
    E_dft = np.sum(mag_dft[:K]**2)
    eta_dft.append(E_dft / E_total * 100)
    E_dct = np.sum(mag_dct[:K]**2)
    eta_dct.append(E_dct / E_total * 100)

print("\n前K个系数能量占比:")
print(f"K\tDFT(%)\tDCT(%)")
for K, ed, ec in zip(K_list, eta_dft, eta_dct):
    print(f"{K}\t{ed:.2f}\t{ec:.2f}")

plt.figure(figsize=(8,5))
plt.plot(K_list, eta_dft, "o-", label="DFT", lw=2)
plt.plot(K_list, eta_dct, "s-", label="DCT-II", lw=2)
plt.xlabel("Number of Top Coefficients")
plt.ylabel("Energy Ratio (%)")
plt.title("Energy Concentration Comparison")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("energy_concentration.png", dpi=300)
plt.show(block=False)

# -------------------------- 5. 逆变换重建验证 --------------------------
K = int(N*0.1)
X_dft_trunc = np.zeros_like(X_dft)
X_dft_trunc[:K] = X_dft[:K]
X_dft_trunc[-K+1:] = X_dft[-K+1:]
x_dft_recon = ifft(X_dft_trunc).real

X_dct_trunc = np.zeros_like(X_dct)
X_dct_trunc[:K] = X_dct[:K]
x_dct_recon = idct(X_dct_trunc, norm="ortho")

plt.figure(figsize=(10,5))
plt.plot(np.arange(N), x, "k-", label="Original", lw=2)
plt.plot(np.arange(N), x_dft_recon, "r--", label=f"DFT Reconstruct (Top {K})", lw=2)
plt.plot(np.arange(N), x_dct_recon, "g--", label=f"DCT Reconstruct (Top {K})", lw=2)
plt.xlabel("n")
plt.ylabel("x[n]")
plt.title("Reconstruction Comparison")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("reconstruction_compare.png", dpi=300)
plt.show(block=False)

# 最后阻塞，防止程序退出
plt.show()