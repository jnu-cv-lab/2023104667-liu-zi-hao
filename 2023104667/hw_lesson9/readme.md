\# 第10次课程作业｜CNN图像分类实验

姓名：刘子豪

学号：2023104667

提交日期：2026.06.06



\## 一、文件目录

hw\_lesson10/

├── code-MNIST.py # MNIST 手写数字分类 CNN 源码

├── code-CIFAR-10.py # CIFAR-10 彩色图像分类 CNN 源码

├── 曲线图.png # 训练 loss/acc 变化曲线图

├── 输出结果.png # 控制台日志输出截图

├── 训练过程.png # Epoch 训练过程截图

├── 预测图.png # 样本预测效果图

└── 实验报告 (第十周).doc # 实验原理、数据分析、总结报告





\## 二、环境依赖

```bash

pip install torch torchvision matplotlib numpy



三、运行指令

bash

\# 手写数字实验

python code-MNIST.py

\# 彩色图片分类实验

python code-CIFAR-10.py



四、实验说明

实验内容：搭建基础 CNN 网络，分别在 MNIST 灰度数据集、CIFAR-10 彩色数据集完成图像分类。

网络结构：卷积层 + 池化层 + 全连接层，Adam 优化器，交叉熵损失。

结果：MNIST 识别精度优于 CIFAR-10，彩色图像特征复杂、分类难度更高。



五、文件说明

两张 py 为完整可运行代码；各类 png 为运行实拍结果；doc 为完整作业实验报告

全选直接复制粘贴README.md。

