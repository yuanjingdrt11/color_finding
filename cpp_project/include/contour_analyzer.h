#ifndef CONTOUR_ANALYZER_H
#define CONTOUR_ANALYZER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

/**
 * 轮廓分析结果
 */
struct ContourAnalysis {
    cv::RotatedRect rotated_rect;     // minAreaRect 结果
    std::vector<cv::Point2f> box_points; // 旋转矩形 4 顶点
    cv::Point2f center_px;            // 中心坐标
    cv::Point2f center_alt_px;        // 对角连线中心
    float angle_deg;                  // 倾斜角度
    float width_px;                   // 旋转矩形宽
    float height_px;                  // 旋转矩形高
    std::string shape;                // "circle" / "square"
    float circularity;                // 圆形度
    float rectangularity;             // 矩形度
    int approx_points;                // 多边形拟合顶点数
    float area_px;                    // 轮廓面积
    float perimeter_px;               // 轮廓周长
    cv::Rect bbox;                    // 正外接矩形
};

/**
 * 轮廓分析器
 * findContours + minAreaRect + 形状判断 + 中心计算
 * 参照 TUP_Vision_2026 装甲板中心计算方法
 */
class ContourAnalyzer {
public:
    ContourAnalyzer();

    /**
     * 寻找所有有效外部轮廓
     * @param mask 二值掩膜
     * @return 有效轮廓列表
     */
    std::vector<std::vector<cv::Point>> findContours(const cv::Mat &mask);

    /**
     * 对单条轮廓进行完整分析
     * @param cnt 轮廓点
     * @return 分析结果
     */
    ContourAnalysis analyze(const std::vector<cv::Point> &cnt);

    /**
     * 获取旋转矩形和对角中心
     */
    static void getRotatedRect(const std::vector<cv::Point> &cnt,
                               cv::RotatedRect &rotated_rect,
                               std::vector<cv::Point2f> &box_points);

    /**
     * 通过对角线连线确定中心
     * 连接旋转矩形的两条对角线，交点 = 几何中心
     */
    static cv::Point2f getCenterByDiagonal(const std::vector<cv::Point2f> &box_points);

    /**
     * 判断形状类型
     * 圆形度 = 4πA/P², 矩形度 = A/(w×h)
     */
    static std::string classifyShape(float circularity, float rectangularity);

    // 可调参数
    static constexpr double MIN_CONTOUR_AREA = 200.0;
    static constexpr double CIRCULARITY_THRESHOLD = 0.80;
    static constexpr double RECTANGULARITY_THRESHOLD = 0.80;
};

#endif // CONTOUR_ANALYZER_H