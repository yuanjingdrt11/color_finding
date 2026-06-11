#include "../include/color_segmenter.h"

std::map<std::string, ColorInfo> ColorSegmenter::s_color_ranges = ColorSegmenter::initColorRanges();

std::map<std::string, ColorInfo> ColorSegmenter::initColorRanges() {
    std::map<std::string, ColorInfo> ranges;

    ranges["red"].ranges = {
        { cv::Scalar(0, 70, 50),   cv::Scalar(10, 255, 255) },
        { cv::Scalar(170, 70, 50), cv::Scalar(179, 255, 255) }
    };
    ranges["red"].bgr = cv::Scalar(0, 0, 255);

    ranges["yellow"].ranges = {
        { cv::Scalar(20, 70, 50),  cv::Scalar(35, 255, 255) }
    };
    ranges["yellow"].bgr = cv::Scalar(0, 255, 255);

    ranges["blue"].ranges = {
        { cv::Scalar(100, 70, 50), cv::Scalar(130, 255, 255) }
    };
    ranges["blue"].bgr = cv::Scalar(255, 0, 0);

    ranges["green"].ranges = {
        { cv::Scalar(36, 70, 50),  cv::Scalar(80, 255, 255) }
    };
    ranges["green"].bgr = cv::Scalar(0, 255, 0);

    ranges["black"].ranges = {
        { cv::Scalar(0, 0, 0),     cv::Scalar(179, 255, 50) }
    };
    ranges["black"].bgr = cv::Scalar(50, 50, 50);

    ranges["light_blue"].ranges = {
        { cv::Scalar(90, 40, 120), cv::Scalar(115, 200, 255) }
    };
    ranges["light_blue"].bgr = cv::Scalar(255, 255, 150);

    return ranges;
}

ColorSegmenter::ColorSegmenter() {
    m_kernel_small  = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    m_kernel_medium = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
}

cv::Mat ColorSegmenter::createMask(const cv::Mat &image, const std::string &color_name) {
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    auto it = s_color_ranges.find(color_name);
    if (it == s_color_ranges.end()) {
        return cv::Mat::zeros(image.size(), CV_8UC1);
    }

    const auto &info = it->second;
    cv::Mat mask = cv::Mat::zeros(hsv.size(), CV_8UC1);

    for (const auto &r : info.ranges) {
        cv::Mat part_mask;
        cv::inRange(hsv, r.lower, r.upper, part_mask);
        cv::bitwise_or(mask, part_mask, mask);
    }

    cv::morphologyEx(mask, mask, cv::MORPH_OPEN,  m_kernel_small,  cv::Point(-1, -1), 1);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, m_kernel_medium, cv::Point(-1, -1), 2);

    return mask.clone();  // 确保返回连续的 Mat
}

const std::map<std::string, ColorInfo> &ColorSegmenter::getColorRanges() {
    return s_color_ranges;
}

cv::Scalar ColorSegmenter::getBGR(const std::string &color_name) {
    auto it = s_color_ranges.find(color_name);
    if (it != s_color_ranges.end()) {
        return it->second.bgr;
    }
    return cv::Scalar(255, 255, 255);
}