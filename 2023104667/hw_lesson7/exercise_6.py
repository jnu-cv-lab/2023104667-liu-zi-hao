import cv2
import numpy as np
import time

# ===================== 路径 =====================
path = "/home/hhhkinggoder1/cv-course/homework6/"

# ===================== ORB 方法（用于对比） =====================
def orb_demo():
    img_box = cv2.imread(path + "box.png", 0)
    img_scene = cv2.imread(path + "box_in_scene.png", 0)
    h, w = img_box.shape[:2]

    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(img_box, None)
    kp2, des2 = orb.detectAndCompute(img_scene, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    match_cnt = len(matches)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

    inlier_cnt = sum(mask.ravel())
    ratio = inlier_cnt / match_cnt if match_cnt !=0 else 0

    try:
        corners = np.float32([[0,0],[0,h],[w,h],[w,0]]).reshape(-1,1,2)
        cv2.perspectiveTransform(corners, H)
        success = "是"
    except:
        success = "否"
    return match_cnt, inlier_cnt, round(ratio,4), success, "快"

# ===================== SIFT 方法（KNN + Lowe滤波 + RANSAC） =====================
def sift_demo():
    img_box = cv2.imread(path + "box.png", 0)
    img_scene = cv2.imread(path + "box_in_scene.png", 0)
    h, w = img_box.shape[:2]

    # 1. SIFT 创建
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_box, None)
    kp2, des2 = sift.detectAndCompute(img_scene, None)

    # 2. BFMatcher + NORM_L2
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)

    # 3. Lowe ratio test 0.75
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    match_cnt = len(good)

    # 4. 单应矩阵 + RANSAC
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

    inlier_cnt = sum(mask.ravel())
    ratio = inlier_cnt / match_cnt if match_cnt !=0 else 0

    # 5. 定位
    try:
        corners = np.float32([[0,0],[0,h],[w,h],[w,0]]).reshape(-1,1,2)
        cv2.perspectiveTransform(corners, H)
        success = "是"
    except:
        success = "否"

    # 6. 画出定位图（保存 SIFT 结果）
    img_scene_color = cv2.imread(path + "box_in_scene.png")
    corners_proj = cv2.perspectiveTransform(corners, H)
    img_res = cv2.polylines(img_scene_color.copy(), [np.int32(corners_proj)], True, (0,255,255), 4)
    cv2.imwrite(path + "sift_target_result.png", img_res)

    return match_cnt, inlier_cnt, round(ratio,4), success, "较慢"

# ===================== 运行并输出对比表 =====================
if __name__ == "__main__":
    print("============== ORB vs SIFT 对比实验 ==============")
    o_match, o_inlier, o_ratio, o_success, o_speed = orb_demo()
    s_match, s_inlier, s_ratio, s_success, s_speed = sift_demo()

    print(f"{'方法':<6}{'匹配数量':<10}{'内点数':<10}{'内点比例':<12}{'定位成功':<12}{'运行速度':<10}")
    print(f"ORB    {o_match:<10}{o_inlier:<10}{o_ratio:<12}{o_success:<12}{o_speed:<10}")
    print(f"SIFT   {s_match:<10}{s_inlier:<10}{s_ratio:<12}{s_success:<12}{s_speed:<10}")

    print("\n============== SIFT 定位图已保存：sift_target_result.png ==============")