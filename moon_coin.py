import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
import os

SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) + "/"

def plot_hist(ax, img, title):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    ax.plot(hist, color='r')
    ax.set_title(title)
    ax.set_xlim([0, 256])

def process_image(img, name):
    eq_global = cv2.equalizeHist(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    eq_clahe = clahe.apply(img)

    fig, axes = plt.subplots(2, 3, figsize=(15,8))
    
    axes[0,0].imshow(img, cmap='gray')
    axes[0,0].set_title('Original')
    axes[0,0].axis('off')

    axes[0,1].imshow(eq_global, cmap='gray')
    axes[0,1].set_title('Global Equalization')
    axes[0,1].axis('off')

    axes[0,2].imshow(eq_clahe, cmap='gray')
    axes[0,2].set_title('CLAHE')
    axes[0,2].axis('off')

    plot_hist(axes[1,0], img, 'Original Histogram')
    plot_hist(axes[1,1], eq_global, 'Global Histogram')
    plot_hist(axes[1,2], eq_clahe, 'CLAHE Histogram')

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}{name}_result.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    moon = data.moon()
    process_image(moon, "Moon")

    coin = data.coins()
    process_image(coin, "Coin")

    print("处理完成！")