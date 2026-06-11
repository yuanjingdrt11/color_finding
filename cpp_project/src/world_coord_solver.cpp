#include "../include/world_coord_solver.h"

cv::Mat WorldCoordSolver::defaultCameraMatrix() {
    // 典型 1080p 相机估计值
    cv::Mat K = (cv::Mat_<double>(3, 3) <<
        1200.0, 0.0,    960.0,
        0.0,    1200.0, 540.0,
        0.0,    0.0,    1.0
    );
    return K;
}

WorldCoordSolver::WorldCoordSolver(float object_z_world)
    : m_camera_matrix(defaultCameraMatrix())
    , m_dist_coeffs(cv::Mat::zeros(5, 1, CV_64F))
    , m_object_z_world(object_z_world)
    , m_pixel_to_mm_scale(0.264f)  // 典型: 1px ≈ 0.264mm
{
}

void WorldCoordSolver::setCameraMatrix(const cv::Mat &K) {
    K.copyTo(m_camera_matrix);
}

void WorldCoordSolver::setPixelToMMScale(float scale) {
    m_pixel_to_mm_scale = scale;
}

cv::Point3f WorldCoordSolver::pixelToWorld(float px, float py) const {
    double fx = m_camera_matrix.at<double>(0, 0);
    double fy = m_camera_matrix.at<double>(1, 1);
    double cx = m_camera_matrix.at<double>(0, 2);
    double cy = m_camera_matrix.at<double>(1, 2);

    // 归一化图像坐标
    double u_n = (px - cx) / fx;
    double v_n = (py - cy) / fy;

    // 世界坐标
    float X = static_cast<float>(u_n * m_object_z_world);
    float Y = static_cast<float>(v_n * m_object_z_world);
    float Z = m_object_z_world;

    return cv::Point3f(X, Y, Z);
}

cv::Point3f WorldCoordSolver::estimateWorldSimple(float px, float py) const {
    double cx = m_camera_matrix.at<double>(0, 2);
    double cy = m_camera_matrix.at<double>(1, 2);

    float dx_mm = static_cast<float>((px - cx) * m_pixel_to_mm_scale);
    float dy_mm = static_cast<float>((py - cy) * m_pixel_to_mm_scale);

    return cv::Point3f(dx_mm, dy_mm, m_object_z_world);
}

float WorldCoordSolver::getWorldScale(float px_size) const {
    double fx = m_camera_matrix.at<double>(0, 0);
    return static_cast<float>(px_size * m_object_z_world / fx);
}