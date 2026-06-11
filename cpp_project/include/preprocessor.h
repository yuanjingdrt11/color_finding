#ifndef PREPROCESSOR_H
#define PREPROCESSOR_H

#include <opencv2/opencv.hpp>

/**
 * 图像预处理器
 * 参照 TUP_Vision_2026 中使用的预处理方法：
 *   - 高斯模糊去噪
 *   - CLAHE 对比度增强 (LAB 色彩空间 L 通道)
 *   - 双边滤波保边去噪
 *   - 针对黑色的特殊处理（轻微腐蚀）
 */
class Preprocessor {
public:
    Preprocessor() = default;

    /**
     * 标准预处理流程
     * 1. 高斯模糊 (5x5)
     * 2. CLAHE 对比度增强
     * 3. 双边滤波
     * @param src 输入 BGR 图像
     * @return 增强后的 BGR 图像
     */
    cv::Mat process(const cv::Mat &src);

    /**
     * 针对特定颜色的预处理
     * @param src 输入 BGR 图像
     * @param color_name 颜色名称
     * @return 增强后的图像
     */
    cv::Mat processForColor(const cv::Mat &src, const std::string &color_name);

private:
    cv::Mat m_gaussian_kernel;
};

#endif // PREPROCESSOR_H