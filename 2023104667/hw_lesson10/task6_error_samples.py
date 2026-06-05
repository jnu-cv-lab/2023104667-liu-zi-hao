import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ===================== 你的模型 =====================
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ===================== 设备 & 数据 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ===================== 加载训练好的模型 =====================
model = CNN().to(device)
model.eval()

# ===================== 自动找错图 =====================
wrong_images = []
true_labels = []
pred_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        # 找出错的
        wrong_idx = (preds != labels)
        wrong_imgs = images[wrong_idx]
        wrong_trues = labels[wrong_idx]
        wrong_preds = preds[wrong_idx]
        
        wrong_images.extend(wrong_imgs.cpu())
        true_labels.extend(wrong_trues.cpu().numpy())
        pred_labels.extend(wrong_preds.cpu().numpy())
        
        if len(wrong_images) >= 8:
            break

# ===================== 显示 8 张错图 =====================
plt.figure(figsize=(12, 6))
for i in range(8):
    plt.subplot(2, 4, i+1)
    img = wrong_images[i].squeeze()
    plt.imshow(img, cmap='gray')
    plt.title(f"True:{true_labels[i]} \nPred:{pred_labels[i]}")
    plt.axis("off")

plt.suptitle("Wrong Classification Samples")
plt.tight_layout()
plt.show()