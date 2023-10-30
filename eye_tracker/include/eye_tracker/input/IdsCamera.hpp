#ifndef HDRMFS_EYE_TRACKER_IDS_CAMERA_HPP
#define HDRMFS_EYE_TRACKER_IDS_CAMERA_HPP

#include "eye_tracker/input/ImageProvider.hpp"

#include <opencv2/opencv.hpp>
#include <ueye.h>

#include <mutex>
#include <string>
#include <thread>

namespace et {
/**
 * Contains all information about the IDS Camera and gatheres images from
 * the camera feed.
 */
class IdsCamera : public ImageProvider {
public:
    /**
     * Initializes camera and starts a separate thread for image gathering.
     */
    IdsCamera(int camera_id);
    /**
     * Returns the newest image obtained by the camera.
     * @param camera_id An id of the camera for which the value is returned.
     * @return A pair of images: one for detecting pupil and one for detecting glints.
     */
    EyeImage grabImage() override;
    /**
     * Shuts down image gathering thread, closes the camera and frees images.
     */
    void close() override;

    /**
     * Sets the exposure of the selected camera.
     * @param exposure Exposure to set in milliseconds.
     * @param camera_id An id of the updated camera.
     */
    void setExposure(double exposure);
    /**
     * Sets the gamma parameter of the selected camera.
     * @param gamma Set exponent of gamma correction
     * @param camera_id An id of the updated camera.
     */
    void setGamma(float gamma);
    /**
     * Sets the framerate of the selected camera.
     * @param framerate Set number of frames per second.
     * @param camera_id An id of the updated camera.
     */
    void setFramerate(double framerate);

private:
    /**
     * Rapidly captures the image from the camera and saves them to the buffer.
     */
    void imageGatheringThread();

    // Size of the buffer to which the captured images are saved.
    static constexpr int IMAGE_IN_QUEUE_COUNT = 10;

    // Array of images in the buffer to which they are captured. One per eye.
    cv::Mat image_queue_[IMAGE_IN_QUEUE_COUNT]{};

    // Index of the most recent image in the buffer.
    int image_index_{-1};

    // Signifies whether the capturing thread is running or not.
    bool thread_running_{true};

    // A thread used to gather images from the camera.
    std::thread image_gatherer_{};

    // Handles to IDS cameras. One per eye.
    uint32_t camera_handle_{};
    // Memory arrays to which the images are directly captured. One per eye.
    char *image_handle_{};
    // Ids of the memory to which the images are directly captured. One per eye.
    int image_id_{};
    // Framerate of the camera.
    double framerate_{100};

    cv::Mat fake_image_{};

    bool fake_camera_{};

    int image_counter_{0};
};
}// namespace et

#endif //HDRMFS_EYE_TRACKER_IDS_CAMERA_HPP