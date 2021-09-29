#pragma once
#include <opencv2/core/types.hpp> // cv::Rect
namespace EyeTracker::Camera {
    constexpr float FPS = 60;
    constexpr int RESOLUTION_X = 1280; // px
    constexpr int RESOLUTION_Y = 1024; // px
    constexpr float SENSOR_X = 6.144; // mm
    constexpr float SENSOR_Y = 4.915; // mm
    constexpr float PIXEL_PITCH = 0.0048; // mm
    const cv::Rect ROI(200, 150, 850, 650);
    // Camera model: "UI-3140CP-M-GL Rev.2 (AB00613)"
}
