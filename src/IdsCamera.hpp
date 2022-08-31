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
    cv::Mat grabPupilImage(int camera_id) override;
    cv::Mat grabGlintImage(int camera_id) override;
    void close() override;
    void setExposure(double exposure, int camera_id);
    void setGamma(float gamma, int camera_id);
    void setFramerate(double framerate, int camera_id);

private:
    void initializeCamera();
    void initializeImage();
    void imageGatheringTwoExposuresThread();
    void imageGatheringOneExposureThread();

    static constexpr int IMAGE_IN_QUEUE_COUNT = 10;

    cv::Mat pupil_image_queues_[IMAGE_IN_QUEUE_COUNT][2]{};
    cv::Mat glint_image_queues_[IMAGE_IN_QUEUE_COUNT][2]{};
    int image_index_{-1};

    bool thread_running_{true};

    std::thread image_gatherer_{};

    int used_camera_count_{};
    uint32_t camera_handles_[2]{};
    char *image_handles_[2]{};
    int image_ids_[2]{};
    double framerate_{100};
};
}// namespace et

#endif