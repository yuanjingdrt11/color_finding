#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成测试图像：包含多种颜色（红黄蓝绿黑浅蓝）的圆形和方形物体
用于验证 color_shape_detector.py 的检测效果
"""

import cv2
import numpy as np
import math
import random

# 输出图像尺寸
WIDTH = 1400
HEIGHT = 1000

# 颜色 BGR 定义（用于绘制测试图形）
COLORS_BGR = {
    "red":        (0, 0, 255),
    "yellow":     (0, 255, 255),
    "blue":       (255, 0, 0),
    "green":      (0, 255, 0),
    "black":      (30, 30, 30),    # 接近黑但不是纯黑（便于观察）
    "light_blue": (255, 255, 150),
}

def draw_rotated_square(canvas, center, size, angle_deg, color, thickness=-1):
    """绘制旋转的方形"""
    cx, cy = center
    half = size // 2
    # 未旋转的角点
    corners = np.array([
        [-half, -half],
        [ half, -half],
        [ half,  half],
        [-half,  half],
    ], dtype=np.float32)

    # 旋转矩阵
    angle_rad = math.radians(angle_deg)
    rot_mat = np.array([
        [math.cos(angle_rad), -math.sin(angle_rad)],
        [math.sin(angle_rad),  math.cos(angle_rad)],
    ])
    rotated = np.dot(corners, rot_mat.T) + np.array([cx, cy])
    pts = rotated.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(canvas, [pts], color)
    # 轮廓
    cv2.polylines(canvas, [pts], isClosed=True, color=(0, 0, 0), thickness=1)

def draw_circle(canvas, center, radius, color, thickness=-1):
    """绘制圆形"""
    cv2.circle(canvas, center, radius, color, thickness)
    # 轮廓
    cv2.circle(canvas, center, radius, (0, 0, 0), 1)

def generate_test_image():
    """生成包含所有颜色圆形和方形的测试图像"""
    canvas = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 240  # 浅灰背景

    # 标题
    cv2.putText(canvas, "Color & Shape Test Image", (WIDTH//2 - 200, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(canvas, "Red | Yellow | Blue | Green | Black | Light Blue",
                (WIDTH//2 - 340, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)

    color_names = ["red", "yellow", "blue", "green", "black", "light_blue"]
    shapes = ["circle", "square"]

    # 布局：上面三列圆形，下面三列方形
    # 每行6个（每种颜色一个），两行
    cols = len(color_names)
    start_x = 120
    spacing_x = (WIDTH - 2 * start_x) // cols
    circle_y = 250
    square_y = 550

    # 添加一些随机噪声模拟真实场景
    noise = np.random.randint(0, 15, canvas.shape, dtype=np.uint8)
    canvas = cv2.add(canvas, noise)
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    # 绘制圆形（带不同大小变化）
    np.random.seed(42)
    for i, color_name in enumerate(color_names):
        cx = start_x + i * spacing_x + spacing_x // 2
        cy = circle_y + random.randint(-20, 20)
        radius = random.randint(35, 55)
        draw_circle(canvas, (cx, cy), radius, COLORS_BGR[color_name])
        # 标签
        cv2.putText(canvas, f"{color_name} circle", (cx - 45, cy - radius - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # 绘制方形（带旋转角度）
    for i, color_name in enumerate(color_names):
        cx = start_x + i * spacing_x + spacing_x // 2
        cy = square_y + random.randint(-20, 20)
        size = random.randint(70, 100)
        angle = random.uniform(-30, 30)  # 随机旋转角度
        draw_rotated_square(canvas, (cx, cy), size, angle, COLORS_BGR[color_name])
        # 标签
        cv2.putText(canvas, f"{color_name} square ({angle:.1f}deg)",
                    (cx - 65, cy - size//2 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # 额外添加几个混合位置的图形（在中间区域）
    extra_coords = [
        (200, 400, "red", "circle", 30, 0),
        (500, 420, "blue", "square", 55, 15),
        (800, 400, "green", "circle", 40, 0),
        (1100, 420, "yellow", "square", 60, -20),
        (350, 750, "black", "circle", 35, 0),
        (650, 780, "light_blue", "square", 50, 25),
        (950, 750, "red", "square", 45, -10),
        (1200, 780, "green", "circle", 32, 0),
    ]

    for ex, ey, color_name, shape, size, angle in extra_coords:
        if shape == "circle":
            draw_circle(canvas, (ex, ey), size, COLORS_BGR[color_name])
        else:
            draw_rotated_square(canvas, (ex, ey), size*2, angle, COLORS_BGR[color_name])

    # 图例
    legend_y = 920
    legend_x = 30
    for i, color_name in enumerate(color_names):
        x = legend_x + i * 220
        cv2.rectangle(canvas, (x, legend_y - 12), (x + 16, legend_y + 4),
                      COLORS_BGR[color_name], -1)
        cv2.rectangle(canvas, (x, legend_y - 12), (x + 16, legend_y + 4),
                      (0, 0, 0), 1)
        cv2.putText(canvas, color_name, (x + 22, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    return canvas


if __name__ == "__main__":
    img = generate_test_image()
    output_path = "test_image.png"
    cv2.imwrite(output_path, img)
    print(f"[生成] 测试图像已保存至: {output_path}")
    print(f"  尺寸: {img.shape[1]}×{img.shape[0]}")
    print(f"  包含 12 个主要物体 + 8 个额外物体（红黄蓝绿黑浅蓝的圆形和方形）")

    # 显示
    cv2.imshow("Test Image", img)
    print("按任意键关闭...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()