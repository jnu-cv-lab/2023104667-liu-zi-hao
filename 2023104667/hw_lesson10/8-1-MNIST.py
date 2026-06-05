import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# ===================== 固定环境 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ===================== CNN模型 =====================
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

# ===================== 训练函数（支持3种优化器） =====================
def train_and_evaluate(optimizer_name, lr=0.001):
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()

    if optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == "sgd_momentum":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)

    epochs = 5
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    for epoch in range(epochs):
        # 训练
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # 验证
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100 * correct_val / total_val

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

        print(f"[{optimizer_name}] Epoch {epoch+1} | Train Loss:{train_loss:.4f} | Train Acc:{train_acc:.2f}% | Val Loss:{val_loss:.4f} | Val Acc:{val_acc:.2f}%")

    # 最终测试准确率
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct_test += (predicted == labels).sum().item()
            total_test += labels.size(0)
    test_acc = 100 * correct_test / total_test

    return train_loss_list, val_loss_list, train_acc_list, val_acc_list, test_acc

# ===================== 运行3种优化器 =====================
print("\n========== 训练：SGD ==========")
sgd = train_and_evaluate("sgd")

print("\n========== 训练：SGD+Momentum ==========")
sgdm = train_and_evaluate("sgd_momentum")

print("\n========== 训练：Adam ==========")
adam = train_and_evaluate("adam")

# ===================== 输出最终测试准确率 =====================
print("\n===== 任务2 最终测试准确率对比 =====")
print(f"SGD            : {sgd[4]:.2f}%")
print(f"SGD+Momentum   : {sgdm[4]:.2f}%")
print(f"Adam           : {adam[4]:.2f}%")

# ===================== 画图对比 =====================
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(sgd[1], label="SGD Val Loss")
plt.plot(sgdm[1], label="SGD+M Val Loss")
plt.plot(adam[1], label="Adam Val Loss")
plt.title("Validation Loss Comparison")
plt.legend()

plt.subplot(1,2,2)
plt.plot(sgd[3], label="SGD Val Acc")
plt.plot(sgdm[3], label="SGD+M Val Acc")
plt.plot(adam[3], label="Adam Val Acc")
plt.title("Validation Accuracy Comparison")
plt.legend()

plt.show()