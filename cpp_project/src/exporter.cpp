#include "../include/exporter.h"
#include "../include/color_shape_detector.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <jsoncpp/json/json.h>

bool JsonExporter::exportToFile(const std::vector<DetectedObject> &objects,
                                 const std::string &filepath) {
    Json::Value root(Json::arrayValue);
    
    for (const auto &obj : objects) {
        Json::Value item;
        item["id"] = obj.id;
        item["color_name"] = obj.color_name;
        item["shape"] = obj.shape;
        
        Json::Value center;
        center.append(obj.center_px.x);
        center.append(obj.center_px.y);
        item["center_px"] = center;
        
        Json::Value center_alt;
        center_alt.append(obj.center_alt_px.x);
        center_alt.append(obj.center_alt_px.y);
        item["center_alt_px"] = center_alt;
        
        item["angle_deg"] = obj.angle_deg;
        item["width_px"] = obj.width_px;
        item["height_px"] = obj.height_px;
        item["area_px"] = obj.area_px;
        item["perimeter_px"] = obj.perimeter_px;
        item["circularity"] = obj.circularity;
        item["rectangularity"] = obj.rectangularity;
        item["contour_approx_points"] = obj.contour_approx_points;
        
        Json::Value world;
        world.append(obj.world_xyz.x);
        world.append(obj.world_xyz.y);
        world.append(obj.world_xyz.z);
        item["world_xyz"] = world;
        
        Json::Value box_pts(Json::arrayValue);
        for (const auto &bp : obj.box_points) {
            Json::Value pt;
            pt.append(bp.x);
            pt.append(bp.y);
            box_pts.append(pt);
        }
        item["box_points"] = box_pts;
        
        Json::Value bbox;
        bbox.append(obj.bbox.x);
        bbox.append(obj.bbox.y);
        bbox.append(obj.bbox.width);
        bbox.append(obj.bbox.height);
        item["bbox"] = bbox;
        
        root.append(item);
    }
    
    std::ofstream ofs(filepath);
    if (!ofs.is_open()) {
        return false;
    }
    
    Json::StyledWriter writer;
    ofs << writer.write(root);
    ofs.close();
    
    std::cout << "[导出] JSON 数据已保存至: " << filepath << std::endl;
    return true;
}

bool CsvExporter::exportToFile(const std::vector<DetectedObject> &objects,
                                const std::string &filepath) {
    if (objects.empty()) {
        std::cout << "[导出] 无检测结果，跳过 CSV 导出" << std::endl;
        return false;
    }
    
    std::ofstream ofs(filepath);
    if (!ofs.is_open()) {
        return false;
    }
    
    // CSV 表头
    ofs << "id,color_name,shape,center_x_px,center_y_px,"
        << "angle_deg,width_px,height_px,area_px,perimeter_px,"
        << "circularity,rectangularity,contour_approx_points,"
        << "world_x_mm,world_y_mm,world_z_mm,"
        << "bbox_x,bbox_y,bbox_w,bbox_h\n";
    
    for (const auto &obj : objects) {
        ofs << obj.id << ","
            << obj.color_name << ","
            << obj.shape << ","
            << std::fixed << std::setprecision(2)
            << obj.center_px.x << ","
            << obj.center_px.y << ","
            << obj.angle_deg << ","
            << obj.width_px << ","
            << obj.height_px << ","
            << obj.area_px << ","
            << obj.perimeter_px << ","
            << std::setprecision(4)
            << obj.circularity << ","
            << obj.rectangularity << ","
            << obj.contour_approx_points << ","
            << std::setprecision(2)
            << obj.world_xyz.x << ","
            << obj.world_xyz.y << ","
            << obj.world_xyz.z << ","
            << obj.bbox.x << ","
            << obj.bbox.y << ","
            << obj.bbox.width << ","
            << obj.bbox.height << "\n";
    }
    
    ofs.close();
    std::cout << "[导出] CSV 数据已保存至: " << filepath << std::endl;
    return true;
}