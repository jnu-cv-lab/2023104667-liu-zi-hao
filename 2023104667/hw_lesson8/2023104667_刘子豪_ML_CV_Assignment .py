
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# ====================== 1. 加载数据集 ======================
digits = load_digits()
X = digits.data  # 特征 (64维)
y = digits.target  # 标签 (0-9)

print("===== 数据集信息 =====")
print("样本总数:", len(X))
print("图像尺寸: 8x8")
print("特征向量形状:", X.shape)
print("类别数量: 10")

# ====================== 2. 划分训练集/测试集 ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print("\n训练集大小:", len(X_train))
print("测试集大小:", len(X_test))

# ====================== 3. 训练6个模型 ======================
models = [
    ("KNN", KNeighborsClassifier()),
    ("Naive Bayes", GaussianNB()),
    ("Logistic Regression", LogisticRegression(max_iter=10000)),
    ("SVM", SVC()),
    ("Decision Tree", DecisionTreeClassifier()),
    ("Random Forest", RandomForestClassifier())
]

print("\n===== 各模型测试集准确率 =====")
acc_list = []
for name, model in models:
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    acc_list.append((name, acc))
    print(f"{name:20s}: {acc:.4f}")

# ====================== 4. 绘制KNN混淆矩阵 ======================
best_model = KNeighborsClassifier()
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=digits.target_names)
plt.figure(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix (KNN)")
plt.show()

# ====================== 5. 错误样本可视化 ======================
wrong_idx = np.where(y_pred != y_test)[0]
print("\n错误样本数量:", len(wrong_idx))

plt.figure(figsize=(10, 4))
for i, idx in enumerate(wrong_idx[:4]):
    plt.subplot(1, 4, i+1)
    img = X_test[idx].reshape(8,8)
    plt.imshow(img, cmap='gray')
    plt.title(f"True:{y_test[idx]}\nPred:{y_pred[idx]}")
    plt.axis('off')
plt.tight_layout()
plt.show()