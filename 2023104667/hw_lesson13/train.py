import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import BadmintonDataset
from model import SkeletonTransformer

# 配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
epoch_num = 20
lr = 1e-3
npy_dir = "/home/hhhkinggoder1/cv-course/homework13/task-13/saved_npy"
ckpt_dir = "/home/hhhkinggoder1/cv-course/homework13/task-13/checkpoints"
vis_dir = "/home/hhhkinggoder1/cv-course/homework13/task-13/vis"
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(vis_dir, exist_ok=True)

# 加载数据集
train_set = BadmintonDataset(os.path.join(npy_dir, "X_train.npy"), os.path.join(npy_dir, "y_train.npy"))
test_set = BadmintonDataset(os.path.join(npy_dir, "X_test.npy"), os.path.join(npy_dir, "y_test.npy"))
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# 模型、损失、优化器
model = SkeletonTransformer().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

best_acc = 0.0
train_loss_list, test_acc_list = [], []

# 训练循环
for epoch in range(epoch_num):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    train_loss_list.append(avg_loss)

    # 验证
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            pred_idx = torch.argmax(pred, dim=1)
            correct += (pred_idx == y).sum().item()
    test_acc = correct / len(test_set)
    test_acc_list.append(test_acc)
    print(f"Epoch[{epoch+1:2d}] Loss:{avg_loss:.4f}  Test Acc:{test_acc:.4f}")

    # 保存最优模型
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_model.pth"))

# 绘制曲线
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(train_loss_list, label="Train Loss")
plt.xlabel("Epoch")
plt.legend()
plt.subplot(1,2,2)
plt.plot(test_acc_list, color="orange", label="Test Acc")
plt.xlabel("Epoch")
plt.legend()
plt.savefig(os.path.join(vis_dir, "train_curve.png"))
plt.close()
print(f"训练结束，最优准确率：{best_acc:.4f}，模型已保存至checkpoints")