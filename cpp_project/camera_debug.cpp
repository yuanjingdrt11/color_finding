/**
 * camera_debug - 摄像头实时检测程序
 * 调用电脑摄像头进行颜色+形状实时识别
 * 
 * 按键:
 *   q / ESC = 退出
 *   s       = 保存当前帧
 */
#include "include/color_shape_detector.h"
#include <iostream>
#include <chrono>
#include <thread>

int main(int argc, char *argv[]) {
    int camera_index = 0;
    float world_z = 500.0f;
    std::vector<std::string> target_colors;
    
    // 解析参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            std::cout << "用法: camera_debug [摄像头索引] [-c 颜色1 颜色2...] [-z Z值]\n"
                      << "默认: 摄像头0, 全部颜色, Z=500mm\n"
                      << "按键: q=退出 s=保存帧\n";
            return 0;
        } else if (arg == "-c") {
            while (i + 1 < argc && argv[i + 1][0] != '-')
                target_colors.push_back(argv[++i]);
        } else if (arg == "-z" && i + 1 < argc) {
            world_z = std::stof(argv[++i]);
        } else if (arg[0] != '-') {
            camera_index = std::stoi(arg);
        }
    }

    std::cout << "[摄像头] 打开摄像头 " << camera_index << std::endl;
    cv::VideoCapture cap(camera_index);
    if (!cap.isOpened()) {
        std::cerr << "错误: 无法打开摄像头 " << camera_index << std::endl;
        return 1;
    }

    // 设置分辨率
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    ColorShapeDetector detector(world_z);
    cv::namedWindow("Camera Debug - Color & Shape Detector", cv::WINDOW_NORMAL);

    auto last_time = std::chrono::steady_clock::now();
    int frame_count = 0;
    double fps = 0;

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        // FPS 计算
        frame_count++;
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time).count();
        if (elapsed >= 1000) {
            fps = frame_count * 1000.0 / elapsed;
            frame_count = 0;
            last_time = now;
        }

        // 执行检测
        auto objects = detector.detect(frame, target_colors, false);
        cv::Mat viz = detector.getVisualization();

        // 绘制 FPS 信息
        char fps_text[64];
        snprintf(fps_text, sizeof(fps_text), "FPS: %.1f | Objects: %zu", fps, objects.size());
        cv::putText(viz, fps_text, cv::Point(10, 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        // 绘制每个检测框
        for (const auto &obj : objects) {
            cv::Scalar color = ColorSegmenter::getBGR(obj.color_name);
            // 旋转矩形
            if (obj.box_points.size() == 4) {
                std::vector<cv::Point> pts;
                for (auto &p : obj.box_points)
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
            // 标签
            char label[128];
            snprintf(label, sizeof(label), "%s(%s)", obj.color_name.c_str(), obj.shape.c_str());
            cv::putText(viz, label,
                       cv::Point(obj.bbox.x, obj.bbox.y - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.45, color, 1);
        }

        cv::imshow("Camera Debug - Color & Shape Detector", viz);

        char key = (char)cv::waitKey(1);
        if (key == 'q' || key == 27) break;
        if (key == 's') {
            cv::imwrite("camera_capture.jpg", frame);
            cv::imwrite("camera_result.jpg", viz);
            std::cout << "[保存] 已保存 camera_capture.jpg / camera_result.jpg" << std::endl;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}