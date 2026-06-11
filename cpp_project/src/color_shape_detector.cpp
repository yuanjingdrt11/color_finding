#include "../include/color_shape_detector.h"
#include <iostream>
#include <iomanip>

ColorShapeDetector::ColorShapeDetector(float object_z_world)
    : m_world_solver(object_z_world)
    , m_object_z_world(object_z_world)
{
}

std::vector<DetectedObject> ColorShapeDetector::detect(
    cv::Mat &image,
    const std::vector<std::string> &target_colors,
    bool enable_visualization)
{
    m_objects.clear();
    
    std::vector<std::string> colors_to_detect = target_colors;
    if (colors_to_detect.empty()) {
        const auto &ranges = ColorSegmenter::getColorRanges();
        for (const auto &pair : ranges) {
            colors_to_detect.push_back(pair.first);
        }
    }
    
    m_visualization = image.clone();
    int object_id = 0;
    
    for (const auto &color_name : colors_to_detect) {
        // 颜色分割：直接对原始图像操作
        cv::Mat mask = m_segmenter.createMask(image, color_name);
        
        // 查找轮廓
        auto contours = m_analyzer.findContours(mask);
        
        // 分析每个轮廓
        for (const auto &cnt : contours) {
            ContourAnalysis analysis = m_analyzer.analyze(cnt);
            cv::Point3f world = m_world_solver.estimateWorldSimple(
                analysis.center_px.x, analysis.center_px.y);
            
            DetectedObject obj;
            obj.id = object_id++;
            obj.color_name = color_name;
            obj.shape = analysis.shape;
            obj.center_px = analysis.center_px;
            obj.center_alt_px = analysis.center_alt_px;
            obj.angle_deg = analysis.angle_deg;
            obj.width_px = analysis.width_px;
            obj.height_px = analysis.height_px;
            obj.area_px = analysis.area_px;
            obj.perimeter_px = analysis.perimeter_px;
            obj.circularity = analysis.circularity;
            obj.rectangularity = analysis.rectangularity;
            obj.contour_approx_points = analysis.approx_points;
            obj.world_xyz = world;
            obj.box_points = analysis.box_points;
            obj.bbox = analysis.bbox;
            obj.contour = cnt;
            m_objects.push_back(obj);
        }
    }
    
    if (enable_visualization) {
        m_visualizer.drawAll(m_visualization, m_objects, true, true);
    }
    
    return m_objects;
}

bool ColorShapeDetector::exportJSON(const std::string &filepath) {
    return m_json_exporter.exportToFile(m_objects, filepath);
}

bool ColorShapeDetector::exportCSV(const std::string &filepath) {
    return m_csv_exporter.exportToFile(m_objects, filepath);
}

void ColorShapeDetector::printSummary() const {
    std::cout << "\n=================================================================\n";
    std::cout << "                      检测结果摘要\n";
    std::cout << "=================================================================\n";
    printf("%-5s %-10s %-8s %-16s %-8s %-14s %-8s %-20s\n",
           "ID","颜色","形状","中心(x,y)","角度","尺寸(wxh)","圆形度","世界坐标(mm)");
    std::cout << "-----------------------------------------------------------------\n";
    
    for (const auto &obj : m_objects) {
        printf("%-5d %-10s %-8s (%-5.0f,%-5.0f) %6.1f   %-5.0fx%-5.0f   %6.3f   (%-6.1f,%-6.1f,%.0f)\n",
               obj.id,
               obj.color_name.c_str(),
               obj.shape.c_str(),
               obj.center_px.x, obj.center_px.y,
               obj.angle_deg,
               obj.width_px, obj.height_px,
               obj.circularity,
               obj.world_xyz.x, obj.world_xyz.y, obj.world_xyz.z);
    }
    
    std::cout << "-----------------------------------------------------------------\n";
    std::cout << "总计检测到 " << m_objects.size() << " 个物体\n";
    std::cout << "=================================================================\n\n";
}