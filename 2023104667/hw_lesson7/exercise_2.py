import cv2
import numpy as np

# 读取图片
img_box = cv2.imread("/home/hhhkinggoder1/cv-course/homework6/box.png", 0)
img_scene = cv2.imread("/home/hhhkinggoder1/cv-course/homework6/box_in_scene.png", 0)

# ORB
orb = cv2.ORB_create(nfeatures=1000)
kp_box, des_box = orb.detectAndCompute(img_box, None)
kp_scene, des_scene = orb.detectAndCompute(img_scene, None)

# 暴力匹配器 + 汉明距离 + 交叉校验
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des_box, des_scene)

# 1. 绘制【初始全部匹配图】（未筛选、未排序）
img_match_all = cv2.drawMatches(
    img_box, kp_box,
    img_scene, kp_scene,
    matches, None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# 2. 按距离升序排序
matches_sorted = sorted(matches, key=lambda x: x.distance)

# 3. 绘制【前30优质匹配图】
img_match_top30 = cv2.drawMatches(
    img_box, kp_box,
    img_scene, kp_scene,
    matches_sorted[:30], None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# 保存两张图（作业提交）
cv2.imwrite("/home/hhhkinggoder1/cv-course/homework6/orb_all_match.png", img_match_all)    # 初始匹配图
cv2.imwrite("/home/hhhkinggoder1/cv-course/homework6/orb_top30_match.png", img_match_top30)# 前30匹配图

# 输出信息
print(f"总匹配数量：{len(matches)}")

# 展示
cv2.imshow("初始全部匹配", img_match_all)
cv2.imshow("排序后前30匹配", img_match_top30)
cv2.waitKey(0)
cv2.destroyAllWindows()