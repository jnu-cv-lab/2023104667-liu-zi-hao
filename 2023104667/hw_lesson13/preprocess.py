import os
import cv2
import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split

# ===================== 配置（和你的文件夹完全对应）====================
root_dir = "/home/hhhkinggoder1/cv-course/homework13/task-13"
class_names = [
    "backhand_drive",
    "backhand_net_shot",
    "forehand_clear",
    "forehand_drive",
    "forehand_lift",
    "forehand_net_shot"
]
num_classes = len(class_names)
target_frames = 30  # 统一帧数30
# 替换原来的 save_dir = "./saved_npy"
save_dir = "/home/hhhkinggoder1/cv-course/homework13/task-13/saved_npy"
os.makedirs(save_dir, exist_ok=True)

# MediaPipe姿态初始化
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

def extract_pose_keypoints(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)
    if not res.pose_landmarks:
        return np.zeros(33 * 4)
    kps = []
    for lm in res.pose_landmarks.landmark:
        kps.extend([lm.x, lm.y, lm.z, lm.visibility])
    return np.array(kps)

def uniform_resample(seq, target_len):
    n = len(seq)
    if n == target_len:
        return seq
    indices = np.linspace(0, n - 1, target_len, dtype=int)
    return [seq[i] for i in indices]

# 遍历所有视频
all_data = []
all_label = []
for label_idx, cls in enumerate(class_names):
    cls_path = os.path.join(root_dir, cls)
    for vid_name in os.listdir(cls_path):
        if not vid_name.endswith(".mp4"):
            continue
        vid_path = os.path.join(cls_path, vid_name)
        cap = cv2.VideoCapture(vid_path)
        frame_kps_list = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            kp = extract_pose_keypoints(frame)
            frame_kps_list.append(kp)
        cap.release()
        if len(frame_kps_list) < 5:
            continue
        frame_kps_list = uniform_resample(frame_kps_list, target_frames)
        all_data.append(np.array(frame_kps_list))
        all_label.append(label_idx)

# 划分训练测试集
X = np.array(all_data)
y = np.array(all_label)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 保存npy
np.save(os.path.join(save_dir, "X_train.npy"), X_train)
np.save(os.path.join(save_dir, "y_train.npy"), y_train)
np.save(os.path.join(save_dir, "X_test.npy"), X_test)
np.save(os.path.join(save_dir, "y_test.npy"), y_test)
print(f"预处理完成！训练集:{X_train.shape}, 测试集:{X_test.shape}")