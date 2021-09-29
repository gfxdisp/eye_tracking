#pragma once
#include <opencv2/opencv.hpp> // cv::Point2f
#include <cstdint> // uint8_t
#include <limits> // std::numeric_limits
namespace EyeTracker::ImageProcessing {
    struct PointWithRating {
        cv::Point2f point = {-1, -1};
        float rating = std::numeric_limits<float>::infinity();
        inline bool operator<(const PointWithRating& other) const {
            return rating < other.rating;
        }
    };

    void correct(const cv::cuda::GpuMat& image, cv::cuda::GpuMat& result, float alpha=1, float beta=0, float gamma=0.5);
    std::vector<PointWithRating> findCircles(const cv::cuda::GpuMat& frame, uint8_t thresh, float min_radius, float max_radius, float max_rating = 0.05);
}
