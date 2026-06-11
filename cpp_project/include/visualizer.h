#ifndef VISUALIZER_H
#define VISUALIZER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// 前向声明
struct DetectedObject;

/**
 * 可视化标注器
 * 参照 TUP_Vision_2026 draw_ui.cpp 绘制风格
 */
class Visualizer {
public:
    Visualizer() = default;

    /**
     * 绘制所有检测结果的标注
     * @param image 输入图像 (原地修改)
     * @param objects 检测结果列表
     * @param show_diagonals 是否绘制对角线
     * @param show_angle 是否绘制角度箭头
     */
    void drawAll(cv::Mat &image,
                 const std::vector<DetectedObject> &objects,
                 bool show_diagonals = true,
                 bool show_angle = true);

    /**
     * 在图像上叠加边缘描绘
     * @param image 输入图像
     * @param mask 二值掩膜
     * @param color_bgr 边缘颜色
     * @return 叠加后的图像
     */
    static cv::Mat drawEdgeOverlay(const cv::Mat &image,
                                    const cv::Mat &mask,
                                    const cv::Scalar &color_bgr);

private:
    /**
     * 绘制旋转矩形边界框
     */
    void drawRotatedRect(cv::Mat &image,
                         const std::vector<cv::Point2f> &box_points,
                         const cv::Scalar &color);

    /**
     * 绘制对角线
     */
    void drawDiagonals(cv::Mat &image,
                       const std::vector<cv::Point2f> &box_points,
                       const cv::Scalar &color);

    /**
     * 绘制中心点
     */
    void drawCenter(cv::Mat &image,
                    const cv::Point2f &center,
                    const cv::Point2f &center_alt,
                    const cv::Scalar &color);

    /**
     * 绘制文字标签
     */
    void drawLabel(cv::Mat &image,
                   const DetectedObject &obj,
                   const cv::Scalar &color);

    /**
     * 绘制倾斜角度方向箭头
     */
    void drawAngleArrow(cv::Mat &image,
                        const DetectedObject &obj,
                        const cv::Scalar &color);

    // 绘制参数
    static constexpr int FONT = cv::FONT_HERSHEY_SIMPLEX;
    static constexpr double FONT_SCALE = 0.5;
    static constexpr int FONT_THICKNESS = 1;
    static constexpr int LINE_THICKNESS_BOX = 1;
    static constexpr int LINE_THICKNESS_DIAGONAL = 1;
    static constexpr int CIRCLE_RADIUS = 4;
};

#endif // VISUALIZER_H