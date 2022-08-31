#ifndef IMAGE_PROVIDER_H
#define IMAGE_PROVIDER_H

#include <opencv2/opencv.hpp>

namespace et {
class ImageProvider {
public:
    virtual void initialize(bool separate_exposures);
    virtual cv::Mat grabImage(int camera_id);
    virtual cv::Mat grabPupilImage(int camera_id) = 0;
    virtual cv::Mat grabGlintImage(int camera_id) = 0;
    virtual void close();
    virtual bool isSeparateExposures();

protected:
    cv::Mat image_{};
    bool separate_exposures_{false};
};
} // namespace et

#endif