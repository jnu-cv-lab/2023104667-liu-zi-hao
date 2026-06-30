"""
Camera calibration with a 9x6 inner-corner checkerboard.

Usage:
    python calibrate.py                # default: images/, square=25mm, pattern=9x6
    python calibrate.py --square 25
    python calibrate.py --pattern 9 6 --square 25 --images images

Outputs (in ./output):
    corners_*.jpg          每张图的角点检测可视化
    undistort_compare.jpg  原图 vs 去畸变 对比
    calibration_result.txt 内参 K、畸变 D、重投影误差
"""
import argparse
import glob
import os
import sys
import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images", default="images", help="标定图片所在目录")
    p.add_argument("--pattern", nargs=2, type=int, default=[9, 6],
                   metavar=("COLS", "ROWS"),
                   help="内角点数量 (列, 行)，默认 9 6")
    p.add_argument("--square", type=float, default=25.0,
                   help="方格边长（毫米），默认 25")
    p.add_argument("--output", default="output", help="输出目录")
    p.add_argument("--ext", default="jpg,jpeg,png,JPG,JPEG,PNG",
                   help="图片扩展名（逗号分隔）")
    p.add_argument("--max-side", type=int, default=1600,
                   help="检测前先缩放到的最长边像素，提速；0 表示不缩放")
    return p.parse_args()


def find_image_files(images_dir, exts):
    files = []
    for e in exts.split(","):
        files.extend(glob.glob(os.path.join(images_dir, f"*.{e}")))
    return sorted(set(files))


def build_object_points(cols, rows, square_mm):
    """棋盘格在标定板坐标系中的 3D 角点：z=0，单位 mm"""
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_mm
    return objp


def detect_corners(gray, pattern_size):
    """先用快速 flag 找；找不到再退回普通 flag"""
    flags_fast = (cv2.CALIB_CB_ADAPTIVE_THRESH +
                  cv2.CALIB_CB_NORMALIZE_IMAGE +
                  cv2.CALIB_CB_FAST_CHECK)
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags_fast)
    if not found:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    return found, corners


def main():
    args = parse_args()
    cols, rows = args.pattern
    pattern_size = (cols, rows)
    os.makedirs(args.output, exist_ok=True)

    files = find_image_files(args.images, args.ext)
    if not files:
        print(f"[ERROR] 在 {args.images}/ 没找到图片")
        sys.exit(1)

    print(f"[INFO] 找到 {len(files)} 张图片")
    print(f"[INFO] 棋盘格内角点: {cols} x {rows}, 方格边长: {args.square} mm")

    objp = build_object_points(cols, rows, args.square)
    obj_points, img_points, used_files = [], [], []
    image_size = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    for fp in files:
        img = cv2.imread(fp)
        if img is None:
            print(f"  [skip] 读不出: {fp}")
            continue

        # 可选：缩放后检测，加快速度，但角点坐标会换算回原图
        h0, w0 = img.shape[:2]
        scale = 1.0
        if args.max_side > 0 and max(h0, w0) > args.max_side:
            scale = args.max_side / max(h0, w0)
            small = cv2.resize(img, (int(w0 * scale), int(h0 * scale)))
        else:
            small = img

        gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        found, corners_small = detect_corners(gray_small, pattern_size)

        if not found:
            print(f"  [FAIL] 未检测到角点: {os.path.basename(fp)}")
            continue

        # 把缩放图找到的角点映射回原图坐标
        corners = corners_small / scale

        # 用原图灰度做亚像素精化（结果更准）
        gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners = cv2.cornerSubPix(
            gray_full, corners, winSize=(11, 11), zeroZone=(-1, -1),
            criteria=criteria)

        obj_points.append(objp)
        img_points.append(corners)
        used_files.append(fp)
        if image_size is None:
            image_size = (w0, h0)

        # 保存角点可视化
        vis = img.copy()
        cv2.drawChessboardCorners(vis, pattern_size, corners, True)
        out_name = "corners_" + os.path.splitext(os.path.basename(fp))[0] + ".jpg"
        cv2.imwrite(os.path.join(args.output, out_name), vis)
        print(f"  [OK]   {os.path.basename(fp)}")

    n_used = len(obj_points)
    print(f"\n[INFO] 成功使用 {n_used} / {len(files)} 张图做标定")
    if n_used < 8:
        print("[ERROR] 有效图片太少，至少需要 8 张（推荐 15+）")
        sys.exit(1)

    print("[INFO] 正在调用 cv2.calibrateCamera ...")
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None)

    # 重投影误差（每张图 + 平均）
    total_err, total_pts = 0.0, 0
    per_image_err = []
    for i, (objp_i, imgp_i) in enumerate(zip(obj_points, img_points)):
        proj, _ = cv2.projectPoints(objp_i, rvecs[i], tvecs[i], K, D)
        err = cv2.norm(imgp_i, proj, cv2.NORM_L2) / len(proj)
        per_image_err.append(err)
        total_err += err * err * len(proj)
        total_pts += len(proj)
    mean_err = np.sqrt(total_err / total_pts)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    k1, k2, p1, p2, k3 = D.ravel()[:5]
    cx_img, cy_img = image_size[0] / 2, image_size[1] / 2

    # 打印 + 写文件
    lines = []
    lines.append("=" * 60)
    lines.append("Camera Calibration Result")
    lines.append("=" * 60)
    lines.append(f"Images used        : {n_used} / {len(files)}")
    lines.append(f"Image size (WxH)   : {image_size[0]} x {image_size[1]}")
    lines.append(f"Pattern (inner)    : {cols} x {rows}")
    lines.append(f"Square size        : {args.square} mm")
    lines.append("")
    lines.append("Intrinsic matrix K =")
    lines.append(f"  [[{K[0,0]:12.4f}, {K[0,1]:12.4f}, {K[0,2]:12.4f}],")
    lines.append(f"   [{K[1,0]:12.4f}, {K[1,1]:12.4f}, {K[1,2]:12.4f}],")
    lines.append(f"   [{K[2,0]:12.4f}, {K[2,1]:12.4f}, {K[2,2]:12.4f}]]")
    lines.append("")
    lines.append(f"fx = {fx:.4f}    fy = {fy:.4f}    |fx-fy| = {abs(fx-fy):.4f}")
    lines.append(f"cx = {cx:.4f}    cy = {cy:.4f}")
    lines.append(f"image center      = ({cx_img:.1f}, {cy_img:.1f})")
    lines.append(f"|cx - W/2| = {abs(cx-cx_img):.2f}  |cy - H/2| = {abs(cy-cy_img):.2f}")
    lines.append("")
    lines.append("Distortion D = [k1, k2, p1, p2, k3]")
    lines.append(f"  k1 = {k1:+.6f}")
    lines.append(f"  k2 = {k2:+.6f}")
    lines.append(f"  p1 = {p1:+.6f}")
    lines.append(f"  p2 = {p2:+.6f}")
    lines.append(f"  k3 = {k3:+.6f}")
    lines.append("")
    lines.append(f"Reprojection error (calibrateCamera RMS) : {rms:.4f} px")
    lines.append(f"Reprojection error (manual mean)         : {mean_err:.4f} px")
    lines.append("")
    lines.append("Per-image reprojection error (px):")
    for fp, e in zip(used_files, per_image_err):
        lines.append(f"  {os.path.basename(fp):30s}  {e:.4f}")
    lines.append("=" * 60)

    result_txt = "\n".join(lines)
    print("\n" + result_txt)
    with open(os.path.join(args.output, "calibration_result.txt"), "w",
              encoding="utf-8") as f:
        f.write(result_txt)

    # 保存 K, D 到 npz，方便后续加载
    np.savez(os.path.join(args.output, "calibration.npz"),
             K=K, D=D, image_size=np.array(image_size))

    # 去畸变对比图：选第一张成功的
    sample = cv2.imread(used_files[0])
    h, w = sample.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=1, newImgSize=(w, h))
    undist = cv2.undistort(sample, K, D, None, new_K)

    # 同尺寸拼接
    label_h = 60
    canvas = np.ones((h + label_h, w * 2 + 20, 3), dtype=np.uint8) * 255
    canvas[label_h:label_h + h, :w] = sample
    canvas[label_h:label_h + h, w + 20:] = undist
    cv2.putText(canvas, "Original", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Undistorted", (w + 40, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(args.output, "undistort_compare.jpg"), canvas)
    cv2.imwrite(os.path.join(args.output, "undistorted_sample.jpg"), undist)

    print(f"\n[DONE] 结果写入 {args.output}/")
    print("       - corners_*.jpg      角点检测可视化")
    print("       - undistort_compare.jpg  去畸变对比")
    print("       - calibration_result.txt 标定结果文本")
    print("       - calibration.npz    K/D 数据")


if __name__ == "__main__":
    main()
