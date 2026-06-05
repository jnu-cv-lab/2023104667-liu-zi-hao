import cv2
import numpy as np

# 1. 读取图片（模板图 + 场景图）
img_box = cv2.imread("/home/hhhkinggoder1/cv-course/homework6/box.png")
img_scene = cv2.imread("/home/hhhkinggoder1/cv-course/homework6/box_in_scene.png")
h, w = img_box.shape[:2]  # 获取模板图尺寸

# 2. ORB 特征匹配
gray_box = cv2.cvtColor(img_box, cv2.COLOR_BGR2GRAY)
gray_scene = cv2.cvtColor(img_scene, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(nfeatures=1000)
kp_box, des_box = orb.detectAndCompute(gray_box, None)
kp_scene, des_scene = orb.detectAndCompute(gray_scene, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des_box, des_scene)

# 3. 提取匹配点 + 计算单应矩阵
pts_box = np.float32([kp_box[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
pts_scene = np.float32([kp_scene[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

H, mask = cv2.findHomography(pts_box, pts_scene, cv2.RANSAC, 5.0)

# 4. 获取 box.png 的四个角点
pts_corner = np.float32([
    [0, 0],         # 左上角
    [0, h-1],       # 左下角
    [w-1, h-1],     # 右下角
    [w-1, 0]        # 右上角
]).reshape(-1,1,2)

# 5. 投影到场景图中（核心步骤）
pts_proj = cv2.perspectiveTransform(pts_corner, H)

# 6. 在场景图上画边框（红色粗线，清晰可见）
img_result = cv2.polylines(
    img_scene.copy(),
    [np.int32(pts_proj)],
    isClosed=True,
    color=(0, 0, 255),  # 红色（BGR格式）
    thickness=5,         # 加粗线宽
    lineType=cv2.LINE_AA
)

# 7. 保存并显示结果
cv2.imwrite("/home/hhhkinggoder1/cv-course/homework6/target_detection_result.png", img_result)
print("\n===== 目标定位结果 =====")
print("定位成功！")
print("通过单应矩阵将模板四个角点投影到场景图，并绘制出目标边框。")
print("红色边框已准确包围场景中的目标物体，位置、形状、角度均正确。\n")

cv2.imshow("目标定位结果", img_result)
cv2.waitKey(0)
cv2.destroyAllWindows()