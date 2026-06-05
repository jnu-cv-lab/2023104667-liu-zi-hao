import cv2
import numpy as np

# ---------------------- 1. 读取图片（已修改你的路径） ----------------------
# 请把路径补全到括号内，保持格式一致
img_box = cv2.imread("/home/hhhkinggoder1/cv-course/homework6/box.png", 0)
img_scene = cv2.imread("/home/hhhkinggoder1/cv-course/homework6/box_in_scene.png", 0)

# ---------------------- 2. 创建ORB检测器 ----------------------
orb = cv2.ORB_create(nfeatures=1000)

# ---------------------- 3. 检测关键点 + 计算描述子 ----------------------
kp_box, des_box = orb.detectAndCompute(img_box, None)
kp_scene, des_scene = orb.detectAndCompute(img_scene, None)

# ---------------------- 4. 可视化关键点 ----------------------
img_box_kp = cv2.drawKeypoints(img_box, kp_box, None, color=(0, 255, 0), flags=0)
img_scene_kp = cv2.drawKeypoints(img_scene, kp_scene, None, color=(0, 255, 0), flags=0)

# 保存图片到你的作业目录
cv2.imwrite('/home/hhhkinggoder1/cv-course/homework6/box_keypoints.png', img_box_kp)
cv2.imwrite('/home/hhhkinggoder1/cv-course/homework6/box_in_scene_keypoints.png', img_scene_kp)

# ---------------------- 5. 输出结果 ----------------------
print("===== box.png 信息 =====")
print(f"关键点数量：{len(kp_box)}")
print(f"描述子维度：{des_box.shape}")

print("\n===== box_in_scene.png 信息 =====")
print(f"关键点数量：{len(kp_scene)}")
print(f"描述子维度：{des_scene.shape}")

# 显示图片
cv2.imshow('box keypoints', img_box_kp)
cv2.imshow('scene keypoints', img_scene_kp)
cv2.waitKey(0)
cv2.destroyAllWindows()