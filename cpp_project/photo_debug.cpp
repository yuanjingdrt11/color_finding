/**
 * photo_debug - 图像文件检测程序
 * 对输入图片进行颜色+形状识别，输出可视化结果和数据
 * 
 * 用法:
 *   ./photo_debug <图像路径> [-c 颜色列表] [-z Z值] [-o 输出路径]
 *   ./photo_debug test_image.png
 *   ./photo_debug test_image.png -c red blue -o result.jpg
 */
#include "include/color_shape_detector.h"
#include <iostream>

int main(int argc, char *argv[]) {
    std::string input_path;
    std::string output_path = "photo_result.jpg";
    std::string json_path = "photo_result.json";
    std::string csv_path = "photo_result.csv";
    std::vector<std::string> target_colors;
    float world_z = 500.0f;
    bool no_show = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            std::cout << "用法: photo_debug <图像路径> [选项]\n\n"
                      << "选项:\n"
                      << "  -o <path>     输出可视化图像 (默认: photo_result.jpg)\n"
                      << "  --json <path>  JSON 输出 (默认: photo_result.json)\n"
                      << "  --csv <path>   CSV 输出 (默认: photo_result.csv)\n"
                      << "  -c <colors>    检测颜色列表\n"
                      << "  -z <mm>        世界 Z 坐标 mm (默认: 500)\n"
                      << "  --no-show      不显示窗口\n"
                      << "示例:\n"
                      << "  photo_debug test_image.png\n"
                      << "  photo_debug test_image.png -c red blue green\n";
            return 0;
        } else if (arg == "-o" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--json" && i + 1 < argc) {
            json_path = argv[++i];
        } else if (arg == "--csv" && i + 1 < argc) {
            csv_path = argv[++i];
        } else if (arg == "-c") {
            while (i + 1 < argc && argv[i + 1][0] != '-')
                target_colors.push_back(argv[++i]);
        } else if (arg == "-z" && i + 1 < argc) {
            world_z = std::stof(argv[++i]);
        } else if (arg == "--no-show") {
            no_show = true;
        } else if (arg[0] != '-') {
            input_path = arg;
        }
    }

    if (input_path.empty()) {
        std::cerr << "错误: 请提供输入图像路径\n";
        std::cerr << "用法: photo_debug <图像路径>\n";
        return 1;
    }

    std::cout << "[加载] " << input_path << std::endl;
    cv::Mat image = cv::imread(input_path);
    if (image.empty()) {
        std::cerr << "错误: 无法读取图像 " << input_path << std::endl;
        return 1;
    }
    std::cout << "[图像] " << image.cols << "x" << image.rows << std::endl;

    ColorShapeDetector detector(world_z);
    auto objects = detector.detect(image, target_colors, false);
    cv::Mat viz = detector.getVisualization();

    // 绘制所有标注
    for (const auto &obj : objects) {
        cv::Scalar color = ColorSegmenter::getBGR(obj.color_name);
        
        // 旋转矩形
        if (obj.box_points.size() == 4) {
            std::vector<cv::Point> pts;
            for (const auto &p : obj.box_points)
                pts.push_back(cv::Point((int)p.x, (int)p.y));
            cv::polylines(viz, std::vector<std::vector<cv::Point>>{pts}, true, color, 1);
            // 对角线
            cv::line(viz, cv::Point((int)obj.box_points[0].x, (int)obj.box_points[0].y),
                     cv::Point((int)obj.box_points[2].x, (int)obj.box_points[2].y), color, 1);
            cv::line(viz, cv::Point((int)obj.box_points[1].x, (int)obj.box_points[1].y),
                     cv::Point((int)obj.box_points[3].x, (int)obj.box_points[3].y), color, 1);
        }
        
        // 中心点
        cv::circle(viz, cv::Point((int)obj.center_px.x, (int)obj.center_px.y),
                   4, cv::Scalar(255, 255, 255), -1);
        cv::circle(viz, cv::Point((int)obj.center_px.x, (int)obj.center_px.y),
                   5, color, 1);
        // 对角中心
        cv::circle(viz, cv::Point((int)obj.center_alt_px.x, (int)obj.center_alt_px.y),
                   3, cv::Scalar(0, 255, 255), -1);
        
        // 标签
        char label[128];
        snprintf(label, sizeof(label), "#%d %s %s", obj.id, obj.color_name.c_str(), obj.shape.c_str());
        cv::putText(viz, label, cv::Point(obj.bbox.x, obj.bbox.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
        snprintf(label, sizeof(label), "angle=%.1fdeg", obj.angle_deg);
        cv::putText(viz, label, cv::Point(obj.bbox.x, obj.bbox.y + obj.bbox.height + 12),
                    cv::FONT_HERSHEY_SIMPLEX, 0.35, color, 1);
    }

    // 保存结果
    cv::imwrite(output_path, viz);
    std::cout << "[保存] 可视化: " << output_path << std::endl;

    // 打印摘要
    printf("\n==============================================================\n");
    printf("                      检测结果摘要\n");
    printf("==============================================================\n");
    printf("%-5s %-10s %-8s %-14s %-8s %-12s %-8s %-20s\n",
           "ID","颜色","形状","中心(x,y)","角度","尺寸(wxh)","圆形度","世界坐标(mm)");
    printf("--------------------------------------------------------------\n");
    for (const auto &obj : objects) {
        printf("%-5d %-10s %-8s (%-5.0f,%-5.0f) %6.1f   %-4.0fx%-4.0f   %6.3f   (%-5.1f,%-5.1f,%.0f)\n",
               obj.id, obj.color_name.c_str(), obj.shape.c_str(),
               obj.center_px.x, obj.center_px.y,
               obj.angle_deg, obj.width_px, obj.height_px,
               obj.circularity,
               obj.world_xyz.x, obj.world_xyz.y, obj.world_xyz.z);
    }
    printf("--------------------------------------------------------------\n");
    printf("总计: %zu 个物体\n", objects.size());
    printf("==============================================================\n\n");

    // 导出数据
    JsonExporter::exportToFile(objects, json_path);
    CsvExporter::exportToFile(objects, csv_path);

    // 显示窗口
    if (!no_show) {
        cv::namedWindow("Photo Debug - Result", cv::WINDOW_NORMAL);
        cv::imshow("Photo Debug - Result", viz);
        std::cout << "[显示] 按任意键关闭..." << std::endl;
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    std::cout << "[完成]" << std::endl;
    return 0;
}