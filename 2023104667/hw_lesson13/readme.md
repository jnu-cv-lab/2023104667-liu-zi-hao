羽毛球动作识别实验（hw\_lesson13）

项目简介

计算机视觉实验：基于 MediaPipe Pose 提取人体骨骼关键点，使用 Skeleton Transformer 实现 6 类羽毛球击球动作时序分类。



文件说明

plaintext

hw\_lesson13/

├─ preprocess.py   # 视频骨骼提取、数据集制作

├─ dataset.py      # 数据集加载类

├─ model.py        # Skeleton Transformer模型

├─ train.py        # 模型训练脚本

├─ infer.py        # 测试评估+单视频推理

├─ checkpoints.zip # 最优训练权重best\_model.pth

├─ train\_curve.png # 训练损失\&准确率曲线

├─ confusion\_matrix.png # 分类混淆矩阵

└─ 第十三次报告.docx  # 完整实验报告



环境依赖

Python、PyTorch、MediaPipe、OpenCV、NumPy、Matplotlib、Seaborn



运行流程

预处理：python preprocess.py 生成 npy 骨架数据集

训练模型：python train.py，自动保存最优权重

推理测试：python infer.py，输出准确率、混淆矩阵、单视频预测结果



实验结果

最优测试准确率：30.36%

模型问题：样本量不足，相似动作易混淆

改进方向：扩充数据集、增加运动特征、延长训练轮数



作者信息

姓名：刘子豪

学号：2023104667

课程：计算机视觉实验

