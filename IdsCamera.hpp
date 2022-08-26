/// Class containing all information about the IDS Camera used for eye-tracking.
/// It serves as an interface for grabbing images and setting camera parameters.

#ifndef IDS_CAMERA_H
#define IDS_CAMERA_H

#include "ImageProvider.hpp"

#include <opencv2/opencv.hpp>
#include <ueye.h>

#include <mutex>
#include <string>
#include <thread>

namespace et {
class IdsCamera : public ImageProvider {
public:
    void initialize(bool separate_exposures) override;
    cv::Mat grabPupilImage() override;
    cv::Mat grabGlintImage() override;
    void close() override;
    void setExposure(double exposure) override;
    void setGamma(float gamma) override;
    void setFramerate(double framerate) override;

private:
    void initializeCamera();
    void initializeImage();
    void imageGatheringTwoExposuresThread();
    void imageGatheringOneExposureThread();

    static constexpr int IMAGE_IN_QUEUE_COUNT = 10;

    cv::Mat pupil_image_queue_[IMAGE_IN_QUEUE_COUNT]{};
    cv::Mat glint_image_queue_[IMAGE_IN_QUEUE_COUNT]{};
    int image_index_{-1};

    bool thread_running_{true};

    bool separate_exposures_{false};

    std::thread image_gatherer_{};

    int n_cameras_{};
    PUEYE_CAMERA_LIST camera_list_{};
    uint32_t camera_handle_{};
    SENSORINFO sensor_info_{};
    char *image_handle_{};
    int image_id_{};
    double framerate_{100};

    cv::Mat temp_image_{};
};
}// namespace et

#endif