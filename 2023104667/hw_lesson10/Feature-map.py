import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 模型结构（和你之前的CNN保持一致）
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
        # 为了获取第一层特征图，单独返回conv1的输出
        x = self.conv1(x)
        x = self.relu(x)
        return x  # 这里直接返回第一层卷积后的特征图

# 设备与数据加载
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# 加载模型（如果有训练好的权重，取消下面注释）
model = CNN().to(device)
# model.load_state_dict(torch.load("你的模型路径.pth"))
model.eval()

# 选一张测试图片（取第一个batch）
img, label = next(iter(test_loader))
img = img.to(device)

# 前向传播，获取第一层特征图
with torch.no_grad():
    feature_maps = model(img)  # shape: [1, 16, 28, 28]

# 显示原图 + 前8张特征图
plt.figure(figsize=(12, 6))
# 先显示原图
plt.subplot(2, 5, 1)
plt.imshow(img[0][0].cpu().numpy(), cmap='gray')
plt.title(f"Original Image (Label: {label.item()})")
plt.axis('off')

# 显示前8张特征图
for i in range(8):
    fm = feature_maps[0, i].cpu().numpy()
    plt.subplot(2, 5, i + 2)
    plt.imshow(fm, cmap='gray')
    plt.title(f"Feature Map {i+1}")
    plt.axis('off')

plt.suptitle("First Layer Feature Maps Visualization")
plt.tight_layout()
plt.show()