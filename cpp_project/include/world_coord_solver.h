#ifndef WORLD_COORD_SOLVER_H
#define WORLD_COORD_SOLVER_H

#include <opencv2/opencv.hpp>

/**
 * 世界坐标解算器
 * 参照 TUP_Vision_2026 coordsolver 模块设计
 * 
 * 使用简化的单目视觉模型：
 *   - 假设物体位于已知高度 (Z_world)
 *   - 使用相机内参进行逆透视投影
 * 
 * 公式：
 *   u_n = (px - cx) / fx
 *   v_n = (py - cy) / fy
 *   X_world = u_n * Z_world
 *   Y_world = v_n * Z_world
 */
class WorldCoordSolver {
public:
    /**
     * @param object_z_world 物体在世界坐标系中的 Z 值 (mm)
     */
    WorldCoordSolver(float object_z_world = 500.0f);

    /**
     * 设置相机内参矩阵
     */
    void setCameraMatrix(const cv::Mat &K);

    /**
     * 设置像素到毫米的比例因子
     */
    void setPixelToMMScale(float scale);

    /**
     * 将像素坐标转换为世界坐标
     * @param px 像素 x
     * @param py 像素 y
     * @return 世界坐标 (X, Y, Z) in mm
     */
    cv::Point3f pixelToWorld(float px, float py) const;

    /**
     * 简化版世界坐标估计
     * 使用像素到毫米比例因子
     */
    cv::Point3f estimateWorldSimple(float px, float py) const;

    /**
     * 获取像素尺寸对应的世界尺寸
     */
    float getWorldScale(float px_size) const;

    /**
     * 获取当前物体世界 Z 值
     */
    float getObjectZ() const { return m_object_z_world; }

    /**
     * 设置物体世界 Z 值
     */
    void setObjectZ(float z) { m_object_z_world = z; }

private:
    cv::Mat m_camera_matrix;    // 3x3 相机内参矩阵
    cv::Mat m_dist_coeffs;      // 畸变系数
    float m_object_z_world;     // 物体世界 Z 值 (mm)
    float m_pixel_to_mm_scale;  // 像素到毫米比例因子

    // 默认相机内参 (典型 1080p 相机估计值)
    static cv::Mat defaultCameraMatrix();
};

#endif // WORLD_COORD_SOLVER_H