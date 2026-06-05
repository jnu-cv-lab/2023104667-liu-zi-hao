# ==================== 任务4：5种传统机器学习模型训练与评估 ====================
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# 1. 加载数据 + 划分数据集
digits = load_digits()
X = digits.data
y = digits.target

# 75%训练集 25%测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 2. 导入需要的5种+1种分类器
from sklearn.neighbors import KNeighborsClassifier       # KNN
from sklearn.naive_bayes import GaussianNB               # 朴素贝叶斯
from sklearn.linear_model import LogisticRegression      # 逻辑回归
from sklearn.svm import SVC                              # SVM
from sklearn.tree import DecisionTreeClassifier          # 决策树
from sklearn.ensemble import RandomForestClassifier      # 随机森林

# 3. 定义所有要训练的模型（名字 + 模型）
models = [
    ("KNN 最近邻", KNeighborsClassifier()),
    ("朴素贝叶斯", GaussianNB()),
    ("逻辑回归", LogisticRegression(max_iter=10000)),
    ("SVM 支持向量机", SVC()),
    ("决策树", DecisionTreeClassifier()),
    ("随机森林", RandomForestClassifier())
]

# 4. 遍历训练所有模型 + 计算准确率
print("=" * 60)
print("          传统机器学习分类器 测试集准确率对比")
print("=" * 60)

for name, model in models:
    # 训练模型
    model.fit(X_train, y_train)
    
    # 在测试集上预测
    y_pred = model.predict(X_test)
    
    # 计算准确率
    acc = model.score(X_test, y_test)
    
    # 输出结果（保留4位小数，方便写报告）
    print(f"{name:12s} : 准确率 = {acc:.4f}")

print("=" * 60)