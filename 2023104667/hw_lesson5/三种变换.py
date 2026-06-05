import cv2
import numpy as np

# 读取你的原图
img = cv2.imread("/home/hhhkinggoder1/cv-course/homework5/5-1.jpg")
h, w = img.shape[:2]

# ==================== 1. 相似变换 ====================
center = (w//2, h//2)
angle = 30       # 旋转30度
scale = 0.8      # 缩放到0.8倍
M = cv2.getRotationMatrix2D(center, angle, scale)
similar_img = cv2.warpAffine(img, M, (w, h))

# 保存相似变换结果图
cv2.imwrite("/home/hhhkinggoder1/cv-course/homework5/similarity.jpg", similar_img)

# 显示
cv2.imshow("Similarity 相似变换", similar_img)

# ==================== 2. 仿射变换 ====================
pts1 = np.float32([[50,50], [200,50], [50,200]])
pts2 = np.float32([[10,100], [200,50], [100,250]])
M_affine = cv2.getAffineTransform(pts1, pts2)
affine_img = cv2.warpAffine(img, M_affine, (w, h))

# 保存仿射变换结果图
cv2.imwrite("/home/hhhkinggoder1/cv-course/homework5/affine.jpg", affine_img)

# 显示
cv2.imshow("Affine 仿射变换", affine_img)

# ==================== 3. 透视变换 ====================
# 这里用测试图做透视
pts1_p = np.float32([[50, 50], [w-50, 50], [50, h-50], [w-50, h-50]])
pts2_p = np.float32([[80, 100], [w-10, 50], [100, h-80], [w-50, h-20]])
M_persp = cv2.getPerspectiveTransform(pts1_p, pts2_p)
persp_img = cv2.warpPerspective(img, M_persp, (w, h))

# 保存透视变换结果图
cv2.imwrite("/home/hhhkinggoder1/cv-course/homework5/perspective.jpg", persp_img)

# 显示
cv2.imshow("Perspective 透视变换", persp_img)

# 等待按键后关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()

print("全部完成！三张结果图已保存")