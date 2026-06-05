import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 模型结构（和你之前一样）
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

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)

# ===================== 任务4：卷积核可视化 =====================
print("正在显示第一层卷积核...")

# 取出 conv1 的权重
kernels = model.conv1.weight.data.cpu()

# 画出前 8 个卷积核
plt.figure(figsize=(10, 5))
for i in range(8):
    kernel = kernels[i, 0]  # 第i个卷积核
    kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())
    
    plt.subplot(2, 4, i+1)
    plt.imshow(kernel, cmap="gray")
    plt.title(f"Kernel {i+1}")
    plt.axis("off")

plt.suptitle("第1层卷积核（前8个）")
plt.show()