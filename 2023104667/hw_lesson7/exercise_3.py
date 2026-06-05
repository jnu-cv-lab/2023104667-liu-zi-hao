import cv2
import numpy as np

# ====================== 1. 读取图片（你的路径） ======================
img_box = cv2.imread("/home/hhhkinggoder1/cv-course/homework6/box.png")
img_scene = cv2.imread("/home/hhhkinggoder1/cv-course/homework6/box_in_scene.png")

# 灰度图
gray_box = cv2.cvtColor(img_box, cv2.COLOR_BGR2GRAY)
gray_scene = cv2.cvtColor(img_scene, cv2.COLOR_BGR2GRAY)

# ====================== 2. ORB 特征检测 ======================
orb = cv2.ORB_create(nfeatures=1000)
kp_box, des_box = orb.detectAndCompute(gray_box, None)
kp_scene, des_scene = orb.detectAndCompute(gray_scene, None)

# ====================== 3. 暴力匹配 ======================
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des_box, des_scene)
num_matches = len(matches)  # 总匹配数

# ====================== 4. 提取匹配点坐标 ======================
pts_box = np.float32([kp_box[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
pts_scene = np.float32([kp_scene[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

# ====================== 5. 单应矩阵 + RANSAC 剔除误匹配 ======================
H, mask = cv2.findHomography(pts_box, pts_scene, cv2.RANSAC, ransacReprojThreshold=5.0)

# mask 是内点标记（1=内点，0=外点）
inlier_mask = mask.ravel().tolist()
num_inliers = sum(inlier_mask)  # 内点数量
inlier_ratio = num_inliers / num_matches  # 内点比例

# ====================== 6. 绘制 RANSAC 后的内点匹配 ======================
img_ransac_match = cv2.drawMatches(
    img_box, kp_box,
    img_scene, kp_scene,
    matches, None,
    matchesMask=inlier_mask,  # 只画内点
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# ====================== 保存结果图 ======================
cv2.imwrite("/home/hhhkinggoder1/cv-course/homework6/ransac_inlier_match.png", img_ransac_match)

# ====================== 输出作业需要的所有信息 ======================
print("===== 任务3 输出结果 =====")
print(f"总匹配数量：{num_matches}")
print(f"RANSAC 内点数量：{num_inliers}")
print(f"内点比例：{inlier_ratio:.4f}")
print("\nHomography 矩阵 H：")
print(np.round(H, 4))  # 保留4位小数更美观

# 显示图片
cv2.imshow("RANSAC 内点匹配", img_ransac_match)
cv2.waitKey(0)
cv2.destroyAllWindows()