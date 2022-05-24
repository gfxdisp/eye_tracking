#ifndef IMAGE_PROVIDER_H
#define IMAGE_PROVIDER_H

#include <opencv2/opencv.hpp>

namespace et {
class ImageProvider {
public:
    virtual void initialize() = 0;
    virtual cv::Mat grabImage() = 0;
    virtual void close() = 0;
    [[maybe_unused]] virtual void setExposure(double exposure);
    virtual void setGamma(float gamma);
    virtual void setFramerate(double framerate);

protected:
    cv::Mat image_{};
};
}// namespace et

#endif