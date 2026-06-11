/**
 * 颜色识别与形状检测系统 - C++ OpenCV 实现
 * 参照 TUP_Vision_2026 项目架构设计
 * 
 * 功能：
 *   1. 识别红/黄/蓝/绿/黑/浅蓝六种颜色
 *   2. findContours + minAreaRect 寻找边界
 *   3. 圆形度 + 矩形度判断圆形与方形
 *   4. 对角连线法确定中心
 *   5. 计算倾斜角
 *   6. 世界坐标解算
 *   7. 可视化标注 + 边缘描绘
 *   8. JSON/CSV 数据输出
 */

#include "include/color_shape_detector.h"
#include <iostream>
#include <cstring>

void printUsage(const char *prog) {
    std::cout << "用法: " << prog << " <图像路径> [选项]\n"
              << "      " << prog << " 0   # 使用摄像头\n\n"
              << "选项:\n"
              << "  -o <path>      输出可视化图像路径 (默认: output_result.jpg)\n"
              << "  --json <path>   JSON 输出路径 (默认: detection_result.json)\n"
              << "  --csv <path>    CSV 输出路径 (默认: detection_result.csv)\n"
              << "  -c <colors>     要检测的颜色 (空格分隔, 默认全部)\n"
              << "  -z <mm>         物体世界 Z 坐标 (mm) (默认: 500)\n"
              << "  --no-show       不显示图形窗口\n"
              << "  --save-only     仅保存结果，不显示窗口\n"
              << "  -h, --help      显示帮助信息\n"
              << std::endl;
}

int main(int argc, char *argv[]) {
    // 解析命令行参数
    std::string input_path;
    std::string output_path = "output_result.jpg";
    std::string json_path = "detection_result.json";
    std::string csv_path = "detection_result.csv";
    std::vector<std::string> target_colors;
    float world_z = 500.0f;
    bool no_show = false;
    bool save_only = false;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "-o" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--json" && i + 1 < argc) {
            json_path = argv[++i];
        } else if (arg == "--csv" && i + 1 < argc) {
            csv_path = argv[++i];
        } else if (arg == "-c" && i + 1 < argc) {
            // 解析颜色列表
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                target_colors.push_back(argv[++i]);
            }
        } else if (arg == "-z" && i + 1 < argc) {
            world_z = std::stof(argv[++i]);
        } else if (arg == "--no-show") {
            no_show = true;
        } else if (arg == "--save-only") {
            save_only = true;
        } else if (i == 1 && arg[0] != '-') {
            input_path = arg;
        }
    }
    
    if (input_path.empty()) {
        std::cerr << "错误：请提供输入图像路径或摄像头索引\n";
        printUsage(argv[0]);
        return 1;
    }
    
    // 初始化检测器
    ColorShapeDetector detector(world_z);
    
    // 判断输入来源：摄像头或图像文件
    bool use_camera = false;
    int camera_index = 0;
    try {
        camera_index = std::stoi(input_path);
        use_camera = true;
    } catch (...) {
        use_camera = false;
    }
    
    if (use_camera) {
        std::cout << "[摄像头] 开启摄像头索引 " << camera_index 
                  << "，按 'q' 退出，按 's' 保存当前帧" << std::endl;
        
        cv::VideoCapture cap(camera_index);
        if (!cap.isOpened()) {
            std::cerr << "错误：无法打开摄像头 " << camera_index << std::endl;
            return 1;
        }
        
        cv::namedWindow("Color & Shape Detector - Live (C++)", cv::WINDOW_NORMAL);
        
        while (true) {
            cv::Mat frame;
            cap >> frame;
            if (frame.empty()) {
                std::cerr << "摄像头读取失败" << std::endl;
                break;
            }
            
            auto objects = detector.detect(frame, target_colors);
            cv::Mat viz = detector.getVisualization();
            cv::imshow("Color & Shape Detector - Live (C++)", viz);
            
            char key = static_cast<char>(cv::waitKey(1) & 0xFF);
            if (key == 'q') {
                break;
            } else if (key == 's') {
                std::string save_path = "capture_cpp.jpg";
                cv::imwrite(save_path, frame);
                std::cout << "[保存] 原始帧已保存至 " << save_path << std::endl;
                cv::imwrite("capture_result_cpp.jpg", viz);
                std::cout << "[保存] 检测结果已保存至 capture_result_cpp.jpg" << std::endl;
            }
        }
        
        cap.release();
        cv::destroyAllWindows();
    } else {
        std::cout << "[加载] 图像: " << input_path << std::endl;
        cv::Mat image = cv::imread(input_path);
        if (image.empty()) {
            std::cerr << "错误：无法读取图像 " << input_path << std::endl;
            return 1;
        }
        std::cout << "[图像] 尺寸: " << image.cols << "x" << image.rows << std::endl;
        
        // 执行检测
        auto objects = detector.detect(image, target_colors);
        
        // 打印摘要
        detector.printSummary();
        
        // 保存可视化结果
        cv::Mat viz = detector.getVisualization();
        cv::imwrite(output_path, viz);
        std::cout << "[保存] 可视化结果已保存至: " << output_path << std::endl;
        
        // 导出数据
        detector.exportJSON(json_path);
        detector.exportCSV(csv_path);
        
        // 显示
        if (!no_show && !save_only) {
            cv::namedWindow("Color & Shape Detector - Result (C++)", cv::WINDOW_NORMAL);
            cv::imshow("Color & Shape Detector - Result (C++)", viz);
            std::cout << "[显示] 按任意键关闭窗口..." << std::endl;
            cv::waitKey(0);
            cv::destroyAllWindows();
        }
    }
    
    std::cout << "[完成] 检测完毕。" << std::endl;
    return 0;
}