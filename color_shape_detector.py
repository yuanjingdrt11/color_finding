#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
颜色识别与形状检测系统
功能：
  1. 识别颜色：红色、黄色、蓝色、黑色、绿色、浅蓝色
  2. 使用 findContours + 旋转矩阵 (minAreaRect) 寻找边界
  3. 判断圆形与方形
  4. 通过对角线连线确定中心
  5. 计算倾斜角（旋转角度）
  6. 世界坐标解算（基于标定参数的简化模型）
  7. 可视化标注 + 边缘描绘
  8. 输出检测数据 (JSON / CSV)
"""

import cv2
import numpy as np
import json
import csv
import math
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional

# ============================================================
# 数据结构
# ============================================================

@dataclass
class DetectedObject:
    """单个检测结果"""
    id: int
    color_name: str          # 颜色名称
    color_hsv_range: Tuple   # 使用的 HSV 范围
    shape: str               # "circle" 或 "square"
    center_px: Tuple[float, float]   # 图像坐标中心 (cx, cy)
    center_alt_px: Tuple[float, float]  # 对角连线确定的中心
    angle_deg: float         # 倾斜角度（度）
    width_px: float          # 旋转矩形的宽
    height_px: float         # 旋转矩形的高
    area_px: float           # 轮廓面积
    perimeter_px: float      # 轮廓周长
    circularity: float       # 圆形度 (4π*area/perimeter²)
    rectangularity: float    # 矩形度 (area / (w*h))
    contour_approx_points: int  # 多边形拟合顶点数
    world_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # 世界坐标
    box_points: List[Tuple[float, float]] = field(default_factory=list)  # 旋转矩形的4个顶点
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # 正外接矩形


# ============================================================
# 颜色定义（HSV 范围）
# 注：OpenCV HSV 范围 H∈[0,179], S∈[0,255], V∈[0,255]
# ============================================================

COLOR_RANGES = {
    "red": {
        # 红色在 HSV 中跨越 0 度边界，需要两个区间
        "ranges": [
            {"lower": (0, 70, 50),   "upper": (10, 255, 255)},
            {"lower": (170, 70, 50), "upper": (179, 255, 255)},
        ],
        "bgr": (0, 0, 255),      # 标注颜色 BGR
    },
    "yellow": {
        "ranges": [
            {"lower": (20, 70, 50),  "upper": (35, 255, 255)},
        ],
        "bgr": (0, 255, 255),
    },
    "blue": {
        "ranges": [
            {"lower": (100, 70, 50), "upper": (130, 255, 255)},
        ],
        "bgr": (255, 0, 0),
    },
    "green": {
        "ranges": [
            {"lower": (36, 70, 50),  "upper": (80, 255, 255)},
        ],
        "bgr": (0, 255, 0),
    },
    "black": {
        "ranges": [
            {"lower": (0, 0, 0),     "upper": (179, 255, 50)},
        ],
        "bgr": (50, 50, 50),     # 深灰色标注
    },
    "light_blue": {
        "ranges": [
            {"lower": (90, 40, 120), "upper": (115, 200, 255)},
        ],
        "bgr": (255, 255, 150),  # 浅蓝标注
    },
}


# ============================================================
# 图像预处理
# ============================================================

class ImagePreprocessor:
    """图像预处理器 - 严谨的多步骤预处理"""

    @staticmethod
    def preprocess(image: np.ndarray) -> np.ndarray:
        """
        对输入图像进行预处理：
        1. 转灰度（可选，此处保留彩色）
        2. 高斯模糊去噪
        3. 自适应直方图均衡化增强对比度 (CLAHE)
        4. 双边滤波保边去噪
        返回增强后的 BGR 图像
        """
        # 第一步：高斯模糊 - 去除高频噪声
        blurred = cv2.GaussianBlur(image, (5, 5), 0)

        # 第二步：转 LAB 色彩空间，在 L 通道做 CLAHE 增强对比度
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel_eq = clahe.apply(l_channel)
        lab_eq = cv2.merge([l_channel_eq, a_channel, b_channel])
        enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

        # 第三步：双边滤波 - 保边去噪
        enhanced = cv2.bilateralFilter(enhanced, 7, 50, 50)

        return enhanced

    @staticmethod
    def preprocess_for_color(image: np.ndarray, color_name: str) -> np.ndarray:
        """
        针对特定颜色的预处理：
        - 黑色需要降低光照影响
        - 其他颜色使用标准流程
        """
        enhanced = ImagePreprocessor.preprocess(image)

        if color_name == "black":
            # 对黑色额外做轻微腐蚀以分离粘连区域
            kernel = np.ones((3, 3), np.uint8)
            enhanced = cv2.erode(enhanced, kernel, iterations=1)

        return enhanced


# ============================================================
# 颜色分割
# ============================================================

class ColorSegmenter:
    """颜色分割器"""

    @staticmethod
    def create_mask(image: np.ndarray, color_name: str) -> np.ndarray:
        """
        根据颜色名称生成二值掩膜
        使用形态学操作清理噪点和填充孔洞
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        color_info = COLOR_RANGES[color_name]
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for r in color_info["ranges"]:
            lower = np.array(r["lower"], dtype=np.uint8)
            upper = np.array(r["upper"], dtype=np.uint8)
            part_mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.bitwise_or(mask, part_mask)

        # 形态学操作：去除小噪点，填充小孔洞
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # 开运算：先腐蚀后膨胀，去除细小噪点
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        # 闭运算：先膨胀后腐蚀，填充内部小孔洞
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium, iterations=2)

        return mask


# ============================================================
# 轮廓检测与分析
# ============================================================

class ContourAnalyzer:
    """轮廓分析器 - findContours + 旋转矩阵 + 形状判断 + 中心计算"""

    MIN_CONTOUR_AREA = 200     # 最小轮廓面积（过滤噪声）
    CIRCULARITY_THRESHOLD = 0.80   # 圆形度阈值：大于此值判定为圆
    RECTANGULARITY_THRESHOLD = 0.80  # 矩形度阈值：大于此值判定为方形

    @staticmethod
    def find_contours(mask: np.ndarray) -> List[np.ndarray]:
        """
        使用 findContours 寻找所有外部轮廓
        返回有效轮廓列表（已过滤面积过小的）
        """
        contours, hierarchy = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,       # 只检测外轮廓
            cv2.CHAIN_APPROX_TC89_KCOS  # 使用 Teh-Chin 链近似，保留更多细节
        )

        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= ContourAnalyzer.MIN_CONTOUR_AREA:
                valid_contours.append(cnt)

        return valid_contours

    @staticmethod
    def get_rotated_rect(cnt: np.ndarray) -> Tuple[Tuple, np.ndarray]:
        """
        使用 minAreaRect 获取最小面积旋转矩形
        返回:
          rotated_rect: RotatedRect 对象
          box_points: 旋转矩形的 4 个顶点 (4,2) 顺序: 左下->左上->右上->右下（OpenCV 标准）
        """
        rotated_rect = cv2.minAreaRect(cnt)
        box_points = cv2.boxPoints(rotated_rect)  # shape (4,2), float32
        return rotated_rect, box_points

    @staticmethod
    def get_center_by_diagonal(box_points: np.ndarray) -> Tuple[float, float]:
        """
        通过对角线连线确定中心
        方法：连接旋转矩形的两条对角线，交点即为几何中心
        实际就是取 4 个顶点的平均值 = minAreaRect 的 center
        但这里明确演示对角线交点：
          对角线1: box[0] <-> box[2]
          对角线2: box[1] <-> box[3]
          交点 = ((p0+p2)/2 + (p1+p3)/2) / 2 = (p0+p1+p2+p3)/4
        """
        pts = box_points
        mid_diag1 = ((pts[0][0] + pts[2][0]) / 2.0, (pts[0][1] + pts[2][1]) / 2.0)
        mid_diag2 = ((pts[1][0] + pts[3][0]) / 2.0, (pts[1][1] + pts[3][1]) / 2.0)
        # 对角线的两条中点线交于中心
        center_x = (mid_diag1[0] + mid_diag2[0]) / 2.0
        center_y = (mid_diag1[1] + mid_diag2[1]) / 2.0
        return center_x, center_y

    @staticmethod
    def classify_shape(cnt: np.ndarray, rotated_rect: Tuple,
                       box_points: np.ndarray) -> Tuple[str, float, float, int, float, float]:
        """
        综合判断形状类型：
          - 圆形度 (circularity) = 4π * area / perimeter²，越接近1越圆
          - 矩形度 (rectangularity) = contour_area / (w*h of rotated_rect)
          - 多边形拟合顶点数

        返回: (shape, circularity, rectangularity, approx_points, area, perimeter)
        """
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        # 圆形度
        if perimeter > 0:
            circularity = (4.0 * math.pi * area) / (perimeter * perimeter)
        else:
            circularity = 0.0

        # 矩形度：轮廓面积 / 旋转矩形面积
        rw, rh = rotated_rect[1]  # width, height
        rect_area = rw * rh
        if rect_area > 0:
            rectangularity = area / rect_area
        else:
            rectangularity = 0.0

        # 多边形拟合（用于辅助判断）
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        approx_points = len(approx)

        # 综合判断
        shape = "unknown"
        if circularity >= ContourAnalyzer.CIRCULARITY_THRESHOLD:
            shape = "circle"
        elif rectangularity >= ContourAnalyzer.RECTANGULARITY_THRESHOLD:
            shape = "square"
        else:
            # 二次判断：圆形度优先，其次矩形度
            if circularity > rectangularity:
                shape = "circle"
            else:
                shape = "square"

        return shape, circularity, rectangularity, approx_points, area, perimeter

    @staticmethod
    def analyze(cnt: np.ndarray) -> dict:
        """
        对单条轮廓进行完整分析，返回分析结果字典
        """
        rotated_rect, box_points = ContourAnalyzer.get_rotated_rect(cnt)
        center_px = rotated_rect[0]  # (cx, cy) 来自 minAreaRect
        center_alt_px = ContourAnalyzer.get_center_by_diagonal(box_points)
        angle_deg = rotated_rect[2]  # OpenCV 返回的角度
        width_px, height_px = rotated_rect[1]

        shape, circularity, rectangularity, approx_points, area, perimeter = \
            ContourAnalyzer.classify_shape(cnt, rotated_rect, box_points)

        # 正外接矩形
        x, y, w, h = cv2.boundingRect(cnt)
        bbox = (x, y, w, h)

        return {
            "rotated_rect": rotated_rect,
            "box_points": box_points,
            "center_px": center_px,
            "center_alt_px": center_alt_px,
            "angle_deg": angle_deg,
            "width_px": width_px,
            "height_px": height_px,
            "shape": shape,
            "circularity": circularity,
            "rectangularity": rectangularity,
            "approx_points": approx_points,
            "area_px": area,
            "perimeter_px": perimeter,
            "bbox": bbox,
        }


# ============================================================
# 世界坐标解算
# ============================================================

class WorldCoordinateSolver:
    """
    世界坐标解算器
    使用简化的单目视觉模型：
      - 假设物体位于已知高度（如桌面 Z=0 平面）
      - 使用相机内参和畸变系数进行逆投影
      - 如果未标定，使用像素比例近似

    相机内参（示例值，实际使用中应从标定文件读取）：
      fx, fy: 焦距（像素单位）
      cx, cy: 主点坐标
    """

    # 默认相机内参（需要根据实际相机标定替换）
    # 这里使用典型 1080p 相机估计值
    DEFAULT_CAMERA_MATRIX = np.array([
        [1200.0, 0.0,    960.0],
        [0.0,    1200.0, 540.0],
        [0.0,    0.0,    1.0]
    ], dtype=np.float64)

    DEFAULT_DIST_COEFFS = np.zeros((5, 1), dtype=np.float64)  # 零畸变

    # 像素到毫米的比例因子（需要标定）
    PIXEL_TO_MM_SCALE = 0.264  # 典型值：1 pixel ≈ 0.264 mm（取决于工作距离）

    def __init__(self, camera_matrix=None, dist_coeffs=None, object_z_world=0.0):
        self.camera_matrix = camera_matrix if camera_matrix is not None else self.DEFAULT_CAMERA_MATRIX.copy()
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else self.DEFAULT_DIST_COEFFS.copy()
        self.object_z_world = object_z_world  # 假设物体在世界 Z=0 平面

    def pixel_to_world(self, px: float, py: float) -> Tuple[float, float, float]:
        """
        将像素坐标 (px, py) 转换为世界坐标 (X, Y, Z)
        使用逆透视投影（假设 Z_world 已知）

        公式推导：
          从像素坐标到归一化图像坐标：
            u_n = (px - cx) / fx
            v_n = (py - cy) / fy
          从归一化坐标到世界坐标（已知 Z_world）：
            X_world = u_n * Z_world
            Y_world = v_n * Z_world
            Z_world = object_z_world
        """
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        # 归一化图像坐标
        u_n = (px - cx) / fx
        v_n = (py - cy) / fy

        # 世界坐标
        X_world = u_n * self.object_z_world
        Y_world = v_n * self.object_z_world
        Z_world = self.object_z_world

        return X_world, Y_world, Z_world

    def get_world_scale(self, px_size: float) -> float:
        """
        估算像素尺寸对应的世界尺寸
        简化为平行投影近似
        """
        fx = self.camera_matrix[0, 0]
        return px_size * self.object_z_world / fx

    def estimate_world_position_simple(self, px: float, py: float,
                                        known_world_z: float = None) -> Tuple[float, float, float]:
        """
        简化版世界坐标估计（无需完整标定）
        使用 PIXEL_TO_MM_SCALE 直接缩放
        """
        if known_world_z is not None:
            z = known_world_z
        else:
            z = self.object_z_world

        # 假设图像中心为世界原点
        img_center_x = self.camera_matrix[0, 2]
        img_center_y = self.camera_matrix[1, 2]

        dx_mm = (px - img_center_x) * self.PIXEL_TO_MM_SCALE
        dy_mm = (py - img_center_y) * self.PIXEL_TO_MM_SCALE

        return dx_mm, dy_mm, z


# ============================================================
# 可视化
# ============================================================

class Visualizer:
    """可视化标注器"""

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.5
    FONT_THICKNESS = 1
    LINE_THICKNESS_CONTOUR = 2
    LINE_THICKNESS_BOX = 1
    LINE_THICKNESS_DIAGONAL = 1
    CIRCLE_RADIUS = 4

    @staticmethod
    def draw_all(image: np.ndarray, objects: List[DetectedObject],
                 show_contours: bool = True,
                 show_diagonals: bool = True,
                 show_angle: bool = True) -> np.ndarray:
        """
        在图像上绘制所有检测结果的标注
        """
        output = image.copy()

        for obj in objects:
            color_bgr = COLOR_RANGES[obj.color_name]["bgr"]

            # ---- 1. 绘制旋转矩形（边界框） ----
            box_pts = np.array(obj.box_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(output, [box_pts], isClosed=True,
                          color=color_bgr, thickness=Visualizer.LINE_THICKNESS_BOX)

            # ---- 2. 绘制对角线 ----
            if show_diagonals and len(obj.box_points) == 4:
                pts = obj.box_points
                # 对角线1: 顶点0 <-> 顶点2
                cv2.line(output,
                         (int(pts[0][0]), int(pts[0][1])),
                         (int(pts[2][0]), int(pts[2][1])),
                         color_bgr, Visualizer.LINE_THICKNESS_DIAGONAL, cv2.LINE_AA)
                # 对角线2: 顶点1 <-> 顶点3
                cv2.line(output,
                         (int(pts[1][0]), int(pts[1][1])),
                         (int(pts[3][0]), int(pts[3][1])),
                         color_bgr, Visualizer.LINE_THICKNESS_DIAGONAL, cv2.LINE_AA)

            # ---- 3. 绘制轮廓 ----
            if show_contours:
                # 从 box_points 重建近似轮廓（这里用旋转矩形边界代替）
                # 在实际使用中可以在 analyze 时保存轮廓
                cv2.drawContours(output, [box_pts], -1, color_bgr, Visualizer.LINE_THICKNESS_CONTOUR)

            # ---- 4. 绘制中心点 ----
            cx, cy = int(obj.center_px[0]), int(obj.center_px[1])
            cv2.circle(output, (cx, cy), Visualizer.CIRCLE_RADIUS,
                       (255, 255, 255), -1, cv2.LINE_AA)  # 白点
            cv2.circle(output, (cx, cy), Visualizer.CIRCLE_RADIUS + 1,
                       color_bgr, 1, cv2.LINE_AA)  # 有色圈

            # ---- 5. 绘制对角中心点（用于对比） ----
            cx_alt, cy_alt = int(obj.center_alt_px[0]), int(obj.center_alt_px[1])
            cv2.circle(output, (cx_alt, cy_alt), 3,
                       (0, 255, 255), -1, cv2.LINE_AA)  # 黄点标记对角中心

            # ---- 6. 标注文字 ----
            # 标签信息
            label_lines = [
                f"#{obj.id} {obj.color_name} ({obj.shape})",
                f"angle: {obj.angle_deg:.1f}",
                f"center: ({cx},{cy})",
                f"size: {obj.width_px:.0f}x{obj.height_px:.0f}",
                f"circ: {obj.circularity:.3f}",
                f"world: ({obj.world_xyz[0]:.1f},{obj.world_xyz[1]:.1f})mm",
            ]

            # 文字起始位置（在 bbox 上方）
            bx, by, _, _ = obj.bbox
            text_x = bx
            text_y = by - 5
            if text_y < 15:
                text_y = by + 15 * len(label_lines) + 5

            for i, line in enumerate(label_lines):
                y_offset = text_y - (len(label_lines) - 1 - i) * 15
                # 背景半透明框
                (tw, th), _ = cv2.getTextSize(line, Visualizer.FONT,
                                               Visualizer.FONT_SCALE, Visualizer.FONT_THICKNESS)
                cv2.rectangle(output,
                              (text_x - 2, y_offset - th - 2),
                              (text_x + tw + 2, y_offset + 2),
                              (0, 0, 0), -1)
                cv2.putText(output, line,
                            (text_x, y_offset),
                            Visualizer.FONT, Visualizer.FONT_SCALE,
                            color_bgr, Visualizer.FONT_THICKNESS, cv2.LINE_AA)

            # ---- 7. 绘制倾斜角弧线 ----
            if show_angle and obj.angle_deg != 0:
                Visualizer._draw_angle_arc(output, obj, color_bgr)

        return output

    @staticmethod
    def _draw_angle_arc(image: np.ndarray, obj: DetectedObject, color_bgr: Tuple[int, int, int]):
        """在中心绘制倾斜角方向线"""
        cx, cy = int(obj.center_px[0]), int(obj.center_px[1])

        # 角度转换为弧度（OpenCV 角度：水平向右为0，逆时针为正）
        angle_rad = math.radians(obj.angle_deg)
        length = max(obj.width_px, obj.height_px) * 0.6

        # 方向线端点
        end_x = int(cx + length * math.cos(angle_rad))
        end_y = int(cy + length * math.sin(angle_rad))

        cv2.arrowedLine(image, (cx, cy), (end_x, end_y),
                        color_bgr, 2, cv2.LINE_AA, tipLength=0.2)

    @staticmethod
    def draw_edge_overlay(image: np.ndarray, mask: np.ndarray,
                          color_bgr: Tuple[int, int, int]) -> np.ndarray:
        """
        在原始图像上叠加边缘描绘
        将 mask 的轮廓直接绘制为彩色边缘
        """
        edges = cv2.Canny(mask, 50, 150)
        colored_edges = np.zeros_like(image)
        colored_edges[edges > 0] = color_bgr
        overlay = cv2.addWeighted(image, 1.0, colored_edges, 0.8, 0)
        return overlay


# ============================================================
# 主检测流水线
# ============================================================

class ColorShapeDetector:
    """颜色形状检测器 - 主流水线"""

    def __init__(self, camera_matrix=None, dist_coeffs=None, object_z_world=500.0):
        """
        Args:
            object_z_world: 物体在世界坐标中的 Z 值（mm），默认 500mm（50cm）
        """
        self.preprocessor = ImagePreprocessor()
        self.segmenter = ColorSegmenter()
        self.analyzer = ContourAnalyzer()
        self.world_solver = WorldCoordinateSolver(
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            object_z_world=object_z_world
        )
        self.visualizer = Visualizer()
        self.detected_objects: List[DetectedObject] = []

    def detect(self, image: np.ndarray,
               target_colors: List[str] = None,
               enable_visualization: bool = True) -> Tuple[List[DetectedObject], np.ndarray]:
        """
        对输入图像执行完整的颜色形状检测

        Args:
            image: BGR 格式输入图像
            target_colors: 要检测的颜色列表，None 表示检测所有颜色
            enable_visualization: 是否生成可视化输出

        Returns:
            (detected_objects, visualization_image)
        """
        if target_colors is None:
            target_colors = list(COLOR_RANGES.keys())

        self.detected_objects = []
        h, w = image.shape[:2]
        output = image.copy()
        object_id = 0

        for color_name in target_colors:
            # ---- 步骤1：针对颜色的预处理 ----
            processed = self.preprocessor.preprocess_for_color(image, color_name)

            # ---- 步骤2：颜色分割生成掩膜 ----
            mask = self.segmenter.create_mask(processed, color_name)

            # ---- 步骤3：边缘描绘叠加 ----
            color_bgr = COLOR_RANGES[color_name]["bgr"]
            output = self.visualizer.draw_edge_overlay(output, mask, color_bgr)

            # ---- 步骤4：轮廓查找 ----
            contours = self.analyzer.find_contours(mask)

            # ---- 步骤5：对每个轮廓进行分析 ----
            for cnt in contours:
                analysis = self.analyzer.analyze(cnt)

                # ---- 步骤6：世界坐标解算 ----
                wx, wy, wz = self.world_solver.estimate_world_position_simple(
                    analysis["center_px"][0],
                    analysis["center_px"][1],
                )
                # 同时计算世界尺寸
                world_w = self.world_solver.get_world_scale(analysis["width_px"])
                world_h = self.world_solver.get_world_scale(analysis["height_px"])

                # ---- 步骤7：构建检测对象 ----
                obj = DetectedObject(
                    id=object_id,
                    color_name=color_name,
                    color_hsv_range=tuple(COLOR_RANGES[color_name]["ranges"][0]["lower"]),
                    shape=analysis["shape"],
                    center_px=analysis["center_px"],
                    center_alt_px=analysis["center_alt_px"],
                    angle_deg=analysis["angle_deg"],
                    width_px=analysis["width_px"],
                    height_px=analysis["height_px"],
                    area_px=analysis["area_px"],
                    perimeter_px=analysis["perimeter_px"],
                    circularity=analysis["circularity"],
                    rectangularity=analysis["rectangularity"],
                    contour_approx_points=analysis["approx_points"],
                    world_xyz=(wx, wy, wz),
                    box_points=[tuple(p) for p in analysis["box_points"]],
                    bbox=analysis["bbox"],
                )
                self.detected_objects.append(obj)
                object_id += 1

        # ---- 步骤8：可视化标注 ----
        if enable_visualization:
            output = self.visualizer.draw_all(
                output, self.detected_objects,
                show_contours=True,
                show_diagonals=True,
                show_angle=True,
            )

        return self.detected_objects, output

    def export_json(self, filepath: str):
        """导出检测结果为 JSON"""
        data = []
        def to_native(v):
            """递归转换 numpy 类型为 Python 原生类型"""
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating,)):
                return float(v)
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, dict):
                return {kk: to_native(vv) for kk, vv in v.items()}
            if isinstance(v, (list, tuple)):
                return [to_native(i) for i in v]
            return v

        for obj in self.detected_objects:
            d = asdict(obj)
            d = to_native(d)
            data.append(d)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[导出] JSON 数据已保存至: {filepath}")

    def export_csv(self, filepath: str):
        """导出检测结果为 CSV"""
        if not self.detected_objects:
            print("[导出] 无检测结果，跳过 CSV 导出")
            return

        fieldnames = [
            "id", "color_name", "shape", "center_x_px", "center_y_px",
            "angle_deg", "width_px", "height_px", "area_px", "perimeter_px",
            "circularity", "rectangularity", "contour_approx_points",
            "world_x_mm", "world_y_mm", "world_z_mm",
            "bbox_x", "bbox_y", "bbox_w", "bbox_h",
        ]

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for obj in self.detected_objects:
                writer.writerow({
                    "id": obj.id,
                    "color_name": obj.color_name,
                    "shape": obj.shape,
                    "center_x_px": f"{obj.center_px[0]:.2f}",
                    "center_y_px": f"{obj.center_px[1]:.2f}",
                    "angle_deg": f"{obj.angle_deg:.2f}",
                    "width_px": f"{obj.width_px:.2f}",
                    "height_px": f"{obj.height_px:.2f}",
                    "area_px": f"{obj.area_px:.2f}",
                    "perimeter_px": f"{obj.perimeter_px:.2f}",
                    "circularity": f"{obj.circularity:.4f}",
                    "rectangularity": f"{obj.rectangularity:.4f}",
                    "contour_approx_points": obj.contour_approx_points,
                    "world_x_mm": f"{obj.world_xyz[0]:.2f}",
                    "world_y_mm": f"{obj.world_xyz[1]:.2f}",
                    "world_z_mm": f"{obj.world_xyz[2]:.2f}",
                    "bbox_x": obj.bbox[0],
                    "bbox_y": obj.bbox[1],
                    "bbox_w": obj.bbox[2],
                    "bbox_h": obj.bbox[3],
                })
        print(f"[导出] CSV 数据已保存至: {filepath}")

    def print_summary(self):
        """打印检测摘要到控制台"""
        print("\n" + "=" * 70)
        print("                        检测结果摘要")
        print("=" * 70)
        print(f"{'ID':>3} | {'颜色':<8} | {'形状':<8} | {'中心(x,y)':<18} | "
              f"{'角度°':<8} | {'尺寸(w×h)':<16} | {'圆形度':<8} | {'世界坐标(mm)':<20}")
        print("-" * 70)
        for obj in self.detected_objects:
            print(f"{obj.id:>3} | {obj.color_name:<8} | {obj.shape:<8} | "
                  f"({obj.center_px[0]:.0f},{obj.center_px[1]:.0f}){'':>5} | "
                  f"{obj.angle_deg:>6.1f}° | "
                  f"{obj.width_px:.0f}×{obj.height_px:.0f}{'':>5} | "
                  f"{obj.circularity:>.3f}   | "
                  f"({obj.world_xyz[0]:.1f},{obj.world_xyz[1]:.1f},{obj.world_xyz[2]:.1f})")
        print("-" * 70)
        print(f"总计检测到 {len(self.detected_objects)} 个物体")
        print("=" * 70 + "\n")


# ============================================================
# 入口函数
# ============================================================

def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="颜色识别与形状检测系统 - OpenCV 实现"
    )
    parser.add_argument("input", nargs="?", help="输入图像路径（支持图片或摄像头索引 0/1...）")
    parser.add_argument("-o", "--output", default="output_result.jpg",
                        help="输出可视化图像路径 (默认: output_result.jpg)")
    parser.add_argument("--json", default="detection_result.json",
                        help="JSON 输出路径 (默认: detection_result.json)")
    parser.add_argument("--csv", default="detection_result.csv",
                        help="CSV 输出路径 (默认: detection_result.csv)")
    parser.add_argument("-c", "--colors", nargs="+",
                        default=["red", "yellow", "blue", "green", "black", "light_blue"],
                        help="要检测的颜色列表 (默认: red yellow blue green black light_blue)")
    parser.add_argument("-z", "--world-z", type=float, default=500.0,
                        help="假设的物体世界 Z 坐标 (mm) (默认: 500)")
    parser.add_argument("--camera", help="相机标定参数 JSON 文件路径")
    parser.add_argument("--no-show", action="store_true",
                        help="不显示图形窗口")
    parser.add_argument("--save-only", action="store_true",
                        help="仅保存结果，不显示窗口")
    parser.add_argument("--gray", action="store_true",
                        help="输入为灰度相机索引（如 0,1,2...）")
    args = parser.parse_args()

    # 加载相机参数（如果提供）
    camera_matrix = None
    dist_coeffs = None
    if args.camera:
        with open(args.camera, 'r') as f:
            calib = json.load(f)
            camera_matrix = np.array(calib.get("camera_matrix", WorldCoordinateSolver.DEFAULT_CAMERA_MATRIX))
            dist_coeffs = np.array(calib.get("dist_coeffs", WorldCoordinateSolver.DEFAULT_DIST_COEFFS))

    # 初始化检测器
    detector = ColorShapeDetector(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        object_z_world=args.world_z
    )

    # 加载图像或摄像头
    if args.input is None:
        print("错误：请提供输入图像路径或摄像头索引")
        print("用法: python color_shape_detector.py <image_path>")
        print("      python color_shape_detector.py 0   # 使用摄像头")
        parser.print_help()
        return

    # 尝试解析为摄像头索引
    try:
        src = int(args.input)
        use_camera = True
    except ValueError:
        src = args.input
        use_camera = False

    if use_camera:
        print(f"[摄像头] 开启摄像头索引 {src}，按 'q' 退出，按 's' 保存当前帧")
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"错误：无法打开摄像头 {src}")
            return

        window_name = "Color & Shape Detector - Live"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("摄像头读取失败")
                break

            objects, viz = detector.detect(frame, target_colors=args.colors)
            cv2.imshow(window_name, viz)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = cv2.getTickCount()
                save_path = f"capture_{timestamp}.jpg"
                cv2.imwrite(save_path, frame)
                print(f"[保存] 原始帧已保存至 {save_path}")
                cv2.imwrite("capture_result.jpg", viz)
                print("[保存] 检测结果已保存至 capture_result.jpg")

        cap.release()
        cv2.destroyAllWindows()
    else:
        print(f"[加载] 图像: {src}")
        image = cv2.imread(src)
        if image is None:
            print(f"错误：无法读取图像 {src}")
            return
        print(f"[图像] 尺寸: {image.shape[1]}×{image.shape[0]}")

        # 执行检测
        objects, viz = detector.detect(image, target_colors=args.colors)

        # 打印摘要
        detector.print_summary()

        # 保存可视化结果
        cv2.imwrite(args.output, viz)
        print(f"[保存] 可视化结果已保存至: {args.output}")

        # 导出数据
        detector.export_json(args.json)
        detector.export_csv(args.csv)

        # 显示
        if not args.no_show and not args.save_only:
            window_name = "Color & Shape Detector - Result"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, viz)
            print("[显示] 按任意键关闭窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    print("[完成] 检测完毕。")


if __name__ == "__main__":
    main()