#include "../include/contour_analyzer.h"
#include <cmath>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

ContourAnalyzer::ContourAnalyzer() = default;

std::vector<std::vector<cv::Point>> ContourAnalyzer::findContours(const cv::Mat &mask) {
    std::vector<std::vector<cv::Point>> contours;
    
    // 确保 mask 是连续的单通道 CV_8UC1
    cv::Mat safe_mask;
    if (!mask.isContinuous() || mask.type() != CV_8UC1) {
        safe_mask = mask.clone();
    } else {
        safe_mask = mask;
    }
    
    try {
        cv::findContours(safe_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    } catch (const cv::Exception &e) {
        std::cerr << "[WARN] findContours exception: " << e.what() << std::endl;
        return {};
    }

    // 过滤面积过小的轮廓
    std::vector<std::vector<cv::Point>> valid_contours;
    for (const auto &cnt : contours) {
        double area = cv::contourArea(cnt);
        if (area >= MIN_CONTOUR_AREA) {
            valid_contours.push_back(cnt);
        }
    }
    return valid_contours;
}

void ContourAnalyzer::getRotatedRect(const std::vector<cv::Point> &cnt,
                                     cv::RotatedRect &rotated_rect,
                                     std::vector<cv::Point2f> &box_points) {
    rotated_rect = cv::minAreaRect(cnt);
    box_points.clear();
    cv::boxPoints(rotated_rect, box_points);
}

cv::Point2f ContourAnalyzer::getCenterByDiagonal(const std::vector<cv::Point2f> &box_points) {
    if (box_points.size() < 4) return cv::Point2f(0, 0);
    cv::Point2f mid_diag1(
        (box_points[0].x + box_points[2].x) / 2.0f,
        (box_points[0].y + box_points[2].y) / 2.0f
    );
    cv::Point2f mid_diag2(
        (box_points[1].x + box_points[3].x) / 2.0f,
        (box_points[1].y + box_points[3].y) / 2.0f
    );
    return cv::Point2f(
        (mid_diag1.x + mid_diag2.x) / 2.0f,
        (mid_diag1.y + mid_diag2.y) / 2.0f
    );
}

std::string ContourAnalyzer::classifyShape(float circularity, float rectangularity) {
    if (circularity >= CIRCULARITY_THRESHOLD) {
        return "circle";
    } else if (rectangularity >= RECTANGULARITY_THRESHOLD) {
        return "square";
    } else {
        return (circularity > rectangularity) ? "circle" : "square";
    }
}

ContourAnalysis ContourAnalyzer::analyze(const std::vector<cv::Point> &cnt) {
    ContourAnalysis result;

    getRotatedRect(cnt, result.rotated_rect, result.box_points);

    result.center_px = result.rotated_rect.center;
    result.center_alt_px = getCenterByDiagonal(result.box_points);

    result.angle_deg = result.rotated_rect.angle;
    result.width_px  = result.rotated_rect.size.width;
    result.height_px = result.rotated_rect.size.height;

    result.area_px = static_cast<float>(cv::contourArea(cnt));
    result.perimeter_px = static_cast<float>(cv::arcLength(cnt, true));

    if (result.perimeter_px > 0) {
        result.circularity = static_cast<float>(
            (4.0 * M_PI * result.area_px) / 
            (result.perimeter_px * result.perimeter_px)
        );
    } else {
        result.circularity = 0.0f;
    }

    float rect_area = result.width_px * result.height_px;
    result.rectangularity = (rect_area > 0) ? result.area_px / rect_area : 0.0f;

    double epsilon = 0.02 * result.perimeter_px;
    std::vector<cv::Point> approx;
    cv::approxPolyDP(cnt, approx, epsilon, true);
    result.approx_points = static_cast<int>(approx.size());

    result.shape = classifyShape(result.circularity, result.rectangularity);
    result.bbox = cv::boundingRect(cnt);

    return result;
}