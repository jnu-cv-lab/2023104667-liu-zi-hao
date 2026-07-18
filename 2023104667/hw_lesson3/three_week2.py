import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------- 1. Image Read --------------------------
img_path = "/home/hhhkinggoder1/cv-course/homework/book.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError("Image load failed!")
h, w = img.shape

# -------------------------- 2. Downsampling --------------------------
scale = 0.5
new_h, new_w = int(h * scale), int(w * scale)

small_no_filter = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
img_gauss = cv2.GaussianBlur(img, (5, 5), 1.5)
small_gauss = cv2.resize(img_gauss, (new_w, new_h), interpolation=cv2.INTER_AREA)

# -------------------------- 3. Restoration (3 interpolations) --------------------------
restored = {}
methods = {
    "Nearest": cv2.INTER_NEAREST,
    "Bilinear": cv2.INTER_LINEAR,
    "Bicubic": cv2.INTER_CUBIC
}

for down_type, small_img in [("no_filter", small_no_filter), ("gauss", small_gauss)]:
    restored[down_type] = {}
    for name, inter in methods.items():
        restored[down_type][name] = cv2.resize(small_img, (w, h), interpolation=inter)

# -------------------------- 4. MSE & PSNR --------------------------
def mse_psnr(i1, i2):
    mse = np.mean((i1.astype(float) - i2.astype(float)) **2)
    psnr = 10 * np.log10(255**2 / mse) if mse !=0 else float('inf')
    return mse, psnr

print("=== MSE & PSNR Results ===")
for down in ["no_filter", "gauss"]:
    print(f"\n[{down}]")
    for name, res_img in restored[down].items():
        mse, psnr = mse_psnr(img, res_img)
        print(f"{name}: MSE={mse:.1f}, PSNR={psnr:.1f} dB")

# -------------------------- 5. FFT --------------------------
def fft(img_in):
    f = np.fft.fft2(img_in)
    f = np.fft.fftshift(f)
    mag = 20 * np.log(np.abs(f) + 1)
    return cv2.normalize(mag, None, 0,255,cv2.NORM_MINMAX,cv2.CV_8U)

fft_ori = fft(img)
fft_small = fft(cv2.resize(small_no_filter, (w,h)))
fft_restore = fft(restored["no_filter"]["Bilinear"])

# -------------------------- 6. DCT --------------------------
def dct(img_in):
    d = cv2.dct(np.float32(img_in))
    d_log = 20 * np.log(np.abs(d)+1)
    return cv2.normalize(d_log, None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)

dct_ori = dct(img)
dct_near = dct(restored["no_filter"]["Nearest"])
dct_bili = dct(restored["no_filter"]["Bilinear"])
dct_bicu = dct(restored["no_filter"]["Bicubic"])
dct_gau = dct(restored["gauss"]["Bilinear"])

# -------------------------- 7. PLOT (ALL ENGLISH ✅) --------------------------
plt.rcParams['axes.unicode_minus'] = False

# ---------------- Spatial Domain ----------------
fig1, axes = plt.subplots(2,5,figsize=(20,9))
axes[0,0].imshow(img,cmap='gray'); axes[0,0].set_title("Original"); axes[0,0].axis('off')
axes[0,1].imshow(small_no_filter,cmap='gray'); axes[0,1].set_title("Down No Filter"); axes[0,1].axis('off')
axes[0,2].imshow(restored["no_filter"]["Nearest"],cmap='gray'); axes[0,2].set_title("Nearest"); axes[0,2].axis('off')
axes[0,3].imshow(restored["no_filter"]["Bilinear"],cmap='gray'); axes[0,3].set_title("Bilinear"); axes[0,3].axis('off')
axes[0,4].imshow(restored["no_filter"]["Bicubic"],cmap='gray'); axes[0,4].set_title("Bicubic"); axes[0,4].axis('off')

axes[1,0].imshow(img,cmap='gray'); axes[1,0].set_title("Original"); axes[1,0].axis('off')
axes[1,1].imshow(small_gauss,cmap='gray'); axes[1,1].set_title("Down Gaussian"); axes[1,1].axis('off')
axes[1,2].imshow(restored["gauss"]["Nearest"],cmap='gray'); axes[1,2].set_title("Nearest"); axes[1,2].axis('off')
axes[1,3].imshow(restored["gauss"]["Bilinear"],cmap='gray'); axes[1,3].set_title("Bilinear"); axes[1,3].axis('off')
axes[1,4].imshow(restored["gauss"]["Bicubic"],cmap='gray'); axes[1,4].set_title("Bicubic"); axes[1,4].axis('off')

plt.tight_layout()
fig1.savefig("/home/hhhkinggoder1/cv-course/homework/spatial.png",dpi=300)

# ---------------- FFT ----------------
fig2, ax2 = plt.subplots(1,3,figsize=(15,5))
ax2[0].imshow(fft_ori,cmap='gray'); ax2[0].set_title("FFT Original"); ax2[0].axis('off')
ax2[1].imshow(fft_small,cmap='gray'); ax2[1].set_title("FFT Down"); ax2[1].axis('off')
ax2[2].imshow(fft_restore,cmap='gray'); ax2[2].set_title("FFT Restored"); ax2[2].axis('off')
fig2.savefig("/home/hhhkinggoder1/cv-course/homework/fft.png",dpi=300)

# ---------------- DCT ----------------
fig3, ax3 = plt.subplots(2,3,figsize=(15,10))
ax3[0,0].imshow(dct_ori,cmap='gray'); ax3[0,0].set_title("DCT Original"); ax3[0,0].axis('off')
ax3[0,1].imshow(dct_near,cmap='gray'); ax3[0,1].set_title("DCT Nearest"); ax3[0,1].axis('off')
ax3[0,2].imshow(dct_bili,cmap='gray'); ax3[0,2].set_title("DCT Bilinear"); ax3[0,2].axis('off')
ax3[1,0].imshow(dct_bicu,cmap='gray'); ax3[1,0].set_title("DCT Bicubic"); ax3[1,0].axis('off')
ax3[1,1].imshow(dct_gau,cmap='gray'); ax3[1,1].set_title("DCT Gaussian+Bl"); ax3[1,1].axis('off')
ax3[1,2].axis('off')
fig3.savefig("/home/hhhkinggoder1/cv-course/homework/dct.png",dpi=300)

plt.show()
print("✅ ALL DONE! NO BOXES!")