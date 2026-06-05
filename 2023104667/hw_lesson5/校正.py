import cv2
import numpy as np

# 读取你的斜拍A4照片
img = cv2.imread("/home/hhhkinggoder1/cv-course/homework5/5-1-2.jpg")

# 用你点击的四个角点坐标
pts1 = np.float32([
    (435, 85),    # 左上
    (1270, 367),  # 右上
    (30, 438),    # 左下
    (1016, 995)   # 右下
])

# 校正为标准A4比例的矩形
width, height = 420, 594
pts2 = np.float32([
    [0, 0],
    [width, 0],
    [0, height],
    [width, height]
])

# 透视变换
M = cv2.getPerspectiveTransform(pts1, pts2)
corrected = cv2.warpPerspective(img, M, (width, height))

corrected = cv2.rotate(corrected, cv2.ROTATE_180)

# 保存最终结果
cv2.imwrite("/home/hhhkinggoder1/cv-course/homework5/a4_corrected.jpg", corrected)

# 显示结果
cv2.imshow("Corrected (Final)", corrected)
cv2.waitKey(0)
cv2.destroyAllWindows()