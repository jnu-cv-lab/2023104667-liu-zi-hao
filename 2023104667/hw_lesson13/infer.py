import os
import cv2
import mediapipe as mp
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import BadmintonDataset
from model import SkeletonTransformer

# 类别对应
class_names = [
    "backhand_drive",
    "backhand_net_shot",
    "forehand_clear",
    "forehand_drive",
    "forehand_lift",
    "forehand_net_shot"
]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt_dir = "/home/hhhkinggoder1/cv-course/homework13/task-13/checkpoints"
ckpt_path = os.path.join(ckpt_dir, "best_model.pth")
vis_dir = "/home/hhhkinggoder1/cv-course/homework13/task-13/vis"

# 加载模型
model = SkeletonTransformer().to(device)
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()

# ---------------------- 1. 测试集整体评估 + 混淆矩阵 ----------------------
def eval_testset():
    # 原来的 test_x = "./saved_npy/X_test.npy"
    test_x = "/home/hhhkinggoder1/cv-course/homework13/task-13/saved_npy/X_test.npy"
    test_y = "/home/hhhkinggoder1/cv-course/homework13/task-13/saved_npy/y_test.npy"
    test_set = BadmintonDataset(test_x, test_y)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=False)
    all_pred = []
    all_true = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            out = model(x)
            pred = torch.argmax(out, dim=1).cpu().numpy()
            all_pred.extend(pred)
            all_true.extend(y.numpy())
    # 混淆矩阵
    cm = confusion_matrix(all_true, all_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "confusion_matrix.png"))
    print(classification_report(all_true, all_pred, target_names=class_names))

# ---------------------- 2. 单视频推理函数 ----------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
target_frames = 30

def get_video_feat(vid_path):
    cap = cv2.VideoCapture(vid_path)
    frame_kps = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            kps = []
            for lm in results.pose_landmarks.landmark:
                kps.append([lm.x, lm.y])
            frame_kps.append(kps)
    cap.release()

    target_frames = 30
    n_frame = len(frame_kps)
    # ========== 修复核心：帧数不够则补零，防止索引越界 ==========
    if n_frame < target_frames:
        pad_num = target_frames - n_frame
        zero_pad = [[[0.0, 0.0] for _ in range(33)] for __ in range(pad_num)]
        frame_kps += zero_pad
        n_frame = len(frame_kps)

    indices = np.linspace(0, n_frame - 1, target_frames, dtype=int)
    seq = np.array([frame_kps[i] for i in indices]).astype(np.float32)
    return seq

def predict_video(vid_path):
    feat = get_video_feat(vid_path)
    with torch.no_grad():
        # 1. numpy数组转tensor，放到对应设备
feat_tensor = torch.from_numpy(feat).float().to(device)
# 2. 增加batch维度：[30,33,2] -> [1,30,33,2]
feat_tensor = feat_tensor.unsqueeze(0)

# 推理
with torch.no_grad():
    out = model(feat_tensor)
        prob = torch.softmax(out, dim=1)
        idx = torch.argmax(prob, dim=1).item()
        conf = prob[0, idx].item()
    print(f"预测类别：{class_names[idx]}  置信度：{conf:.4f}")

if __name__ == "__main__":
    eval_testset()
    # 测试你现有的视频 backhand_drive/001.mp4
    test_video = "./backhand_drive/001.mp4"
    print("\n===== 单视频推理结果 =====")
    predict_video(test_video)