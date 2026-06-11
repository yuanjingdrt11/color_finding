#include "../include/visualizer.h"
#include "../include/color_shape_detector.h"
#include "../include/color_segmenter.h"

void Visualizer::drawAll(cv::Mat &image,
                          const std::vector<DetectedObject> &objects,
                          bool show_diagonals,
                          bool show_angle) {
    for (const auto &obj : objects) {
        cv::Scalar color_bgr = ColorSegmenter::getBGR(obj.color_name);
        drawRotatedRect(image, obj.box_points, color_bgr);
        if (show_diagonals && obj.box_points.size() == 4) {
            drawDiagonals(image, obj.box_points, color_bgr);
        }
        drawCenter(image, obj.center_px, obj.center_alt_px, color_bgr);
        drawLabel(image, obj, color_bgr);
        if (show_angle && std::abs(obj.angle_deg) > 0.01f) {
            drawAngleArrow(image, obj, color_bgr);
        }
    }
}

cv::Mat Visualizer::drawEdgeOverlay(const cv::Mat &image,
                                     const cv::Mat &mask,
                                     const cv::Scalar &color_bgr) {
    cv::Mat edges;
    cv::Canny(mask, edges, 50, 150);

    cv::Mat overlay_result = image.clone();
    for (int r = 0; r < image.rows; ++r) {
        for (int c = 0; c < image.cols; ++c) {
            if (edges.at<uchar>(r, c) > 0) {
                cv::Vec3b &pixel = overlay_result.at<cv::Vec3b>(r, c);
                pixel[0] = static_cast<uchar>(0.2f * pixel[0] + 0.8f * static_cast<float>(color_bgr[0]));
                pixel[1] = static_cast<uchar>(0.2f * pixel[1] + 0.8f * static_cast<float>(color_bgr[1]));
                pixel[2] = static_cast<uchar>(0.2f * pixel[2] + 0.8f * static_cast<float>(color_bgr[2]));
            }
        }
    }
    return overlay_result;
}

void Visualizer::drawRotatedRect(cv::Mat &image,
                                  const std::vector<cv::Point2f> &box_points,
                                  const cv::Scalar &color) {
    std::vector<cv::Point> pts;
    for (const auto &p : box_points) {
        pts.push_back(cv::Point(static_cast<int>(p.x), static_cast<int>(p.y)));
    }
    std::vector<std::vector<cv::Point>> contours = { pts };
    cv::polylines(image, contours, true, color, LINE_THICKNESS_BOX, cv::LINE_AA);
    cv::drawContours(image, contours, -1, color, 2, cv::LINE_AA);
}

void Visualizer::drawDiagonals(cv::Mat &image,
                                const std::vector<cv::Point2f> &box_points,
                                const cv::Scalar &color) {
    cv::line(image,
             cv::Point(static_cast<int>(box_points[0].x), static_cast<int>(box_points[0].y)),
             cv::Point(static_cast<int>(box_points[2].x), static_cast<int>(box_points[2].y)),
             color, LINE_THICKNESS_DIAGONAL, cv::LINE_AA);
    cv::line(image,
             cv::Point(static_cast<int>(box_points[1].x), static_cast<int>(box_points[1].y)),
             cv::Point(static_cast<int>(box_points[3].x), static_cast<int>(box_points[3].y)),
             color, LINE_THICKNESS_DIAGONAL, cv::LINE_AA);
}

void Visualizer::drawCenter(cv::Mat &image,
                             const cv::Point2f &center,
                             const cv::Point2f &center_alt,
                             const cv::Scalar &color) {
    cv::Point pt(static_cast<int>(center.x), static_cast<int>(center.y));
    cv::Point pt_alt(static_cast<int>(center_alt.x), static_cast<int>(center_alt.y));
    cv::circle(image, pt, CIRCLE_RADIUS, cv::Scalar(255, 255, 255), -1, cv::LINE_AA);
    cv::circle(image, pt, CIRCLE_RADIUS + 1, color, 1, cv::LINE_AA);
    cv::circle(image, pt_alt, 3, cv::Scalar(0, 255, 255), -1, cv::LINE_AA);
}

void Visualizer::drawLabel(cv::Mat &image,
                            const DetectedObject &obj,
                            const cv::Scalar &color) {
    char buffer[256];
    int cx = static_cast<int>(obj.center_px.x);
    int cy = static_cast<int>(obj.center_px.y);
    int bx = obj.bbox.x;
    int by = obj.bbox.y;

    std::vector<std::string> lines;
    snprintf(buffer, sizeof(buffer), "#%d %s (%s)", obj.id, obj.color_name.c_str(), obj.shape.c_str());
    lines.push_back(buffer);
    snprintf(buffer, sizeof(buffer), "angle: %.1f deg", obj.angle_deg);
    lines.push_back(buffer);
    snprintf(buffer, sizeof(buffer), "center: (%d,%d)", cx, cy);
    lines.push_back(buffer);
    snprintf(buffer, sizeof(buffer), "size: %.0fx%.0f px", obj.width_px, obj.height_px);
    lines.push_back(buffer);
    snprintf(buffer, sizeof(buffer), "circ: %.3f rect: %.3f", obj.circularity, obj.rectangularity);
    lines.push_back(buffer);
    snprintf(buffer, sizeof(buffer), "world: (%.1f,%.1f)mm", obj.world_xyz.x, obj.world_xyz.y);
    lines.push_back(buffer);

    int text_x = bx;
    int text_y = by - 5;
    if (text_y < 15) {
        text_y = by + static_cast<int>(15 * lines.size()) + 5;
    }

    for (size_t i = 0; i < lines.size(); ++i) {
        int y_offset = text_y - static_cast<int>((lines.size() - 1 - i) * 15);
        cv::Size text_size = cv::getTextSize(lines[i], FONT, FONT_SCALE, FONT_THICKNESS, nullptr);
        cv::rectangle(image,
                      cv::Point(text_x - 2, y_offset - text_size.height - 2),
                      cv::Point(text_x + text_size.width + 2, y_offset + 2),
                      cv::Scalar(0, 0, 0), -1);
        cv::putText(image, lines[i], cv::Point(text_x, y_offset),
                    FONT, FONT_SCALE, color, FONT_THICKNESS, cv::LINE_AA);
    }
}

void Visualizer::drawAngleArrow(cv::Mat &image,
                                 const DetectedObject &obj,
                                 const cv::Scalar &color) {
    cv::Point pt(static_cast<int>(obj.center_px.x), static_cast<int>(obj.center_px.y));
    double angle_rad = obj.angle_deg * CV_PI / 180.0;
    double length = std::max(obj.width_px, obj.height_px) * 0.6;
    cv::Point end_pt(
        static_cast<int>(pt.x + length * std::cos(angle_rad)),
        static_cast<int>(pt.y + length * std::sin(angle_rad)));
    cv::arrowedLine(image, pt, end_pt, color, 2, cv::LINE_AA, 0, 0.2);
}