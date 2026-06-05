# 1. 导入需要的库
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split  # 划分数据集的工具

# 2. 加载数据集
digits = load_digits()
X = digits.data  # 特征（64维向量）
y = digits.target  # 标签（0-9）

# 3. 划分数据集：测试集占 25%，训练集占 75%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.25,  # 测试集比例 25%
    random_state=42  # 固定随机种子，保证结果可复现
)

# 4. 输出划分结果
print("===== 数据集划分结果 =====")
print(f"总样本数量：{len(X)}")
print(f"训练集样本数量：{len(X_train)}")
print(f"测试集样本数量：{len(X_test)}")
print(f"测试集比例：{len(X_test)/len(X):.2%}")