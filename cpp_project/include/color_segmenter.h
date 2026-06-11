#ifndef COLOR_SEGMENTER_H
#define COLOR_SEGMENTER_H

#include <opencv2/opencv.hpp>
#include <map>
#include <vector>
#include <string>

/**
 * 颜色范围定义
 */
struct ColorRange {
    cv::Scalar lower;
    cv::Scalar upper;
};

/**
 * 颜色信息
 */
struct ColorInfo {
    std::vector<ColorRange> ranges;
    cv::Scalar bgr;  // 标注颜色
};

/**
 * 颜色分割器
 * 使用 HSV 色彩空间进行多区间分割
 * 配合形态学操作清理噪点和填充孔洞
 */
class ColorSegmenter {
public:
    ColorSegmenter();

    /**
     * 根据颜色名称创建二值掩膜
     * @param image 输入 BGR 图像
     * @param color_name 颜色名称
     * @return 二值掩膜 CV_8UC1
     */
    cv::Mat createMask(const cv::Mat &image, const std::string &color_name);

    /**
     * 获取颜色定义表
     */
    static const std::map<std::string, ColorInfo> &getColorRanges();

    /**
     * 获取某个颜色的 BGR 标注色
     */
    static cv::Scalar getBGR(const std::string &color_name);

private:
    static std::map<std::string, ColorInfo> initColorRanges();
    static std::map<std::string, ColorInfo> s_color_ranges;

    cv::Mat m_kernel_small;
    cv::Mat m_kernel_medium;
};

#endif // COLOR_SEGMENTER_H