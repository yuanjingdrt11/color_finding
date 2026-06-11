#ifndef COLOR_SHAPE_DETECTOR_H
#define COLOR_SHAPE_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <map>
#include "preprocessor.h"
#include "color_segmenter.h"
#include "contour_analyzer.h"
#include "world_coord_solver.h"
#include "visualizer.h"
#include "exporter.h"

/**
 * 检测结果结构体
 * 参照 TUP_Vision_2026 Armor 结构体设计
 */
struct DetectedObject {
    int id;
    std::string color_name;
    std::string shape;
    cv::Point2f center_px;
    cv::Point2f center_alt_px;
    float angle_deg;
    float width_px;
    float height_px;
    float area_px;
    float perimeter_px;
    float circularity;
    float rectangularity;
    int contour_approx_points;
    cv::Point3f world_xyz;
    std::vector<cv::Point2f> box_points;
    cv::Rect bbox;
    std::vector<cv::Point> contour;
};

/**
 * 颜色形状检测器主类
 * 参照 TUP_Vision_2026 Autoaim 类结构设计
 */
class ColorShapeDetector {
public:
    ColorShapeDetector(float object_z_world = 500.0f);
    ~ColorShapeDetector() = default;

    std::vector<DetectedObject> detect(
        cv::Mat &image,
        const std::vector<std::string> &target_colors = {},
        bool enable_visualization = true
    );

    cv::Mat getVisualization() const { return m_visualization; }
    bool exportJSON(const std::string &filepath);
    bool exportCSV(const std::string &filepath);
    void printSummary() const;
    static const std::map<std::string, ColorInfo> &getColorRanges();
    const std::vector<DetectedObject> &getObjects() const { return m_objects; }

private:
    static std::map<std::string, ColorInfo> initColorRanges();

    Preprocessor m_preprocessor;
    ColorSegmenter m_segmenter;
    ContourAnalyzer m_analyzer;
    WorldCoordSolver m_world_solver;
    Visualizer m_visualizer;
    JsonExporter m_json_exporter;
    CsvExporter m_csv_exporter;

    std::vector<DetectedObject> m_objects;
    cv::Mat m_visualization;
    float m_object_z_world;

    static std::map<std::string, ColorInfo> s_color_ranges;
};

#endif