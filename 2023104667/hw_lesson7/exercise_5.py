import cv2
import numpy as np

# ===================== 固定路径 =====================
path = "/home/hhhkinggoder1/cv-course/homework6/"

def run_experiment(nfeatures):
    # 1. 读取图片
    img_box = cv2.imread(path + "box.png")
    img_scene = cv2.imread(path + "box_in_scene.png")
    h, w = img_box.shape[:2]

    gray_box = cv2.cvtColor(img_box, cv2.COLOR_BGR2GRAY)
    gray_scene = cv2.cvtColor(img_scene, cv2.COLOR_BGR2GRAY)

    # 2. ORB 特征检测（使用传入的 nfeatures）
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp_box, des_box = orb.detectAndCompute(gray_box, None)
    kp_scene, des_scene = orb.detectAndCompute(gray_scene, None)

    # 3. 特征匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_box, des_scene)
    num_matches = len(matches)

    # 4. 单应矩阵 + RANSAC
    pts_box = np.float32([kp_box[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    pts_scene = np.float32([kp_scene[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    H, mask = cv2.findHomography(pts_box, pts_scene, cv2.RANSAC, 5.0)

    num_inliers = sum(mask.ravel())
    inlier_ratio = num_inliers / num_matches if num_matches != 0 else 0

    # 5. 定位判断
    try:
        pts_corner = np.float32([[0,0],[0,h],[w,h],[w,0]]).reshape(-1,1,2)
        pts_proj = cv2.perspectiveTransform(pts_corner, H)
        is_success = "是"
    except:
        is_success = "否"

    # 返回结果
    return [
        nfeatures,
        len(kp_box),
        len(kp_scene),
        num_matches,
        num_inliers,
        round(inlier_ratio, 4),
        is_success
    ]

# ===================== 运行三组实验 =====================
print("========== ORB 参数对比实验 ==========")
print("nfeatures\t模板关键点\t场景关键点\t匹配数\t内点数\t内点比例\t定位成功")

for n in [500, 1000, 2000]:
    res = run_experiment(n)
    print(f"{res[0]}\t\t{res[1]}\t\t{res[2]}\t\t{res[3]}\t{res[4]}\t{res[5]}\t\t{res[6]}")

print("\n========== 实验完成 ==========")