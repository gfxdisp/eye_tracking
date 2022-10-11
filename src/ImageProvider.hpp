#ifndef IMAGE_PROVIDER_H
#define IMAGE_PROVIDER_H

#include "EyeEstimator.hpp"
#include "FeatureDetector.hpp"

#include <opencv2/opencv.hpp>

namespace et {
class ImageProvider {
public:
    virtual void initialize() = 0;
    virtual cv::Mat grabImage(int camera_id) = 0;
    virtual void close() = 0;
    virtual std::vector<int> getCameraIds() = 0;

protected:
    cv::Mat image_{};
};
} // namespace et

#endif