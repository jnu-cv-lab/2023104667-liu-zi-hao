import cv2
import numpy as np
from PIL import Image

# 9x6 内角点 = 10x7 个方格，方格 25mm，横向 A4 打印
square_mm = 25
cols_squares, rows_squares = 10, 7  # 方格数量
dpi = 300

mm2px = dpi / 25.4
square_px = int(round(square_mm * mm2px))

# A4 横向：297mm 宽 x 210mm 高
a4_w_px = int(round(297 * mm2px))  # 横向宽度
a4_h_px = int(round(210 * mm2px))  # 横向高度

board_w = cols_squares * square_px
board_h = rows_squares * square_px

print(f'=== 9x6 内角点棋盘格 (方格 {square_mm}mm, 横向 A4) ===')
print(f'方格大小: {square_px} px ({square_mm} mm)')
print(f'方格数量: {cols_squares} x {rows_squares}')
print(f'内角点数: {cols_squares-1} x {rows_squares-1}')
print(f'棋盘尺寸: {board_w} x {board_h} px = {cols_squares * square_mm} x {rows_squares * square_mm} mm')
print(f'A4 纸张(横向): {a4_w_px} x {a4_h_px} px = 297 x 210 mm')

if board_w <= a4_w_px and board_h <= a4_h_px:
    print(f'✓ 可以完整打印在横向 A4 上')
else:
    print(f'✗ 棋盘尺寸: {board_w / mm2px:.0f} x {board_h / mm2px:.0f} mm')
    print(f'  需要: 宽 {board_w / mm2px:.0f} mm (A4横向有 297mm) ✓')
    print(f'  需要: 高 {board_h / mm2px:.0f} mm (A4横向有 210mm) ✓')

ox = (a4_w_px - board_w) // 2
oy = (a4_h_px - board_h) // 2

# 创建白色背景
img = np.ones((a4_h_px, a4_w_px), dtype=np.uint8) * 255

# 绘制棋盘格
for r in range(rows_squares):
    for c in range(cols_squares):
        if (r + c) % 2 == 0:  # 黑色方格
            y1 = oy + r * square_px
            y2 = oy + (r + 1) * square_px
            x1 = ox + c * square_px
            x2 = ox + (c + 1) * square_px
            img[y1:y2, x1:x2] = 0

# 添加标注
label = f"9x6 inner corners | square = {square_mm} mm | print LANDSCAPE at 100%"
cv2.putText(img, label, (ox, oy + board_h + 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2, cv2.LINE_AA)

# 保存
cv2.imwrite("chessboard_9x6_25mm_landscape.png", img)
Image.fromarray(img).save("chessboard_9x6_25mm_A4_landscape.pdf", "PDF", resolution=dpi)

# 验证
ret, corners = cv2.findChessboardCorners(img, (9, 6))
print(f'\n验证: 检测到 9x6 内角点 = {ret}')

if ret:
    print('\n✓ 成功！文件已保存:')
    print(f'  - chessboard_9x6_25mm_landscape.png')
    print(f'  - chessboard_9x6_25mm_A4_landscape.pdf')
    print(f'\n重要：打印时请：')
    print(f'  1. 选择"横向"打印（Landscape）')
    print(f'  2. 选择"100% 实际大小"，不要缩放')
    print(f'  3. 打印后用尺子测量：每个方格应该是 {square_mm}mm × {square_mm}mm')
else:
    print('\n✗ 警告: 无法检测到 9x6 内角点')
