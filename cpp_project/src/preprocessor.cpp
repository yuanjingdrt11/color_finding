#include "../include/preprocessor.h"

cv::Mat Preprocessor::process(const cv::Mat &src) {
    cv::Mat result;

    // 第一步：高斯模糊 - 去除高频噪声
    cv::GaussianBlur(src, result, cv::Size(5, 5), 0);

    // 第二步：转 LAB 色彩空间，在 L 通道做 CLAHE 增强对比度
    cv::Mat lab;
    cv::cvtColor(result, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> lab_channels;
    cv::split(lab, lab_channels);

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(lab_channels[0], lab_channels[0]);

    cv::Mat lab_eq;
    cv::merge(lab_channels, lab_eq);
    cv::cvtColor(lab_eq, result, cv::COLOR_Lab2BGR);

    // 第三步：双边滤波 - 保边去噪
    cv::bilateralFilter(result, result, 7, 50, 50);

    return result;
}

cv::Mat Preprocessor::processForColor(const cv::Mat &src, const std::string &color_name) {
    cv::Mat enhanced = process(src);

    // 黑色需要额外做轻微腐蚀以分离粘连区域
    if (color_name == "black") {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::erode(enhanced, enhanced, kernel, cv::Point(-1, -1), 1);
    }

    return enhanced;
}