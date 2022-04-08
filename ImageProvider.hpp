#ifndef IMAGE_PROVIDER_H
#define IMAGE_PROVIDER_H

#include <opencv2/opencv.hpp>

namespace et {
class ImageProvider {
public:
    virtual void initialize() = 0;
    virtual cv::Mat grabImage() = 0;
    virtual cv::Size2i getImageResolution() = 0;
    virtual cv::Size2i getDeviceResolution() = 0;
    virtual cv::Point2d getOffset() = 0;
    virtual void close() = 0;
    virtual double getPixelPitch();
    [[maybe_unused]] virtual void setExposure(double exposure);
    virtual void setGamma(float gamma);
    virtual void setFramerate(double framerate);

protected:
    cv::Size2i image_resolution_{};
    cv::Point2d offset_{450, 0};
    cv::Mat image_{};
    static constexpr double pixel_pitch_{0.0048};
};
}// namespace et

#endif