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
    void initialize() override;
    cv::Mat grabImage() override;
    void close() override;
    void setExposure(double exposure) override;
    void setGamma(float gamma) override;
    void setFramerate(double framerate) override;

private:
    void initializeCamera();
    void initializeImage();
    void imageGatheringThread();

    static constexpr int IMAGE_IN_QUEUE_COUNT = 10;

    cv::Mat image_queue_[IMAGE_IN_QUEUE_COUNT]{};
    int image_index_{-1};

    bool thread_running_{true};

    std::thread image_gatherer_{};

    int n_cameras_{};
    PUEYE_CAMERA_LIST camera_list_{};
    uint32_t camera_handle_{};
    SENSORINFO sensor_info_{};
    char *image_handle_{};
    int image_id_{};

    cv::Mat temp_image_{};
};
}// namespace et

#endif