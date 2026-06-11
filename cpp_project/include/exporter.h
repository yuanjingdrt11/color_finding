#ifndef EXPORTER_H
#define EXPORTER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

struct DetectedObject;

/**
 * JSON 导出器
 */
class JsonExporter {
public:
    /**
     * 导出检测结果到 JSON 文件
     */
    static bool exportToFile(const std::vector<DetectedObject> &objects,
                             const std::string &filepath);
};

/**
 * CSV 导出器
 */
class CsvExporter {
public:
    /**
     * 导出检测结果到 CSV 文件
     */
    static bool exportToFile(const std::vector<DetectedObject> &objects,
                             const std::string &filepath);
};

#endif // EXPORTER_H