cmake_minimum_required(VERSION 3.15)
project(colorfind CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -march=native")

find_package(OpenCV REQUIRED)
find_package(jsoncpp REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${JSONCPP_INCLUDE_DIRS})

# 核心库源文件（被两个目标共享）
set(CORE_SOURCES
    src/color_shape_detector.cpp
    src/preprocessor.cpp
    src/color_segmenter.cpp
    src/contour_analyzer.cpp
    src/world_coord_solver.cpp
    src/visualizer.cpp
    src/exporter.cpp
)

# ========================================
# camera_debug - 摄像头实时检测程序
# ========================================
add_executable(camera_debug 
    camera_debug.cpp
    ${CORE_SOURCES}
)

target_link_libraries(camera_debug
    ${OpenCV_LIBRARIES}
    jsoncpp
)

# ========================================
# photo_debug - 图像文件检测程序
# ========================================
add_executable(photo_debug 
    photo_debug.cpp
    ${CORE_SOURCES}
)

target_link_libraries(photo_debug
    ${OpenCV_LIBRARIES}
    jsoncpp
)

# 复制测试图像生成器到 build 目录
configure_file(
    ${CMAKE_SOURCE_DIR}/../generate_test_image.py
    ${CMAKE_BINARY_DIR}/generate_test_image.py
    COPYONLY
)

# 打印构建信息
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "Build targets: camera_debug, photo_debug")