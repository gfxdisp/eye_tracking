#ifndef HDRMFS_EYE_TRACKER_IDS_CAMERA_HPP
#define HDRMFS_EYE_TRACKER_IDS_CAMERA_HPP

#include "ImageProvider.hpp"

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
    void initialize() override;
    /**
     * Returns the newest image obtained by the camera.
     * @param camera_id An id of the camera for which the value is returned.
     * @return A pair of images: one for detecting pupil and one for detecting glints.
     */
    ImageToProcess grabImage(int camera_id) override;
    /**
     * Shuts down image gathering thread, closes the camera and frees images.
     */
    void close() override;

    /**
     * Sets the exposure of the selected camera.
     * @param exposure Exposure to set in milliseconds.
     * @param camera_id An id of the updated camera.
     */
    void setExposure(double exposure, int camera_id);
    /**
     * Sets the gamma parameter of the selected camera.
     * @param gamma Set exponent of gamma correction
     * @param camera_id An id of the updated camera.
     */
    void setGamma(float gamma, int camera_id);
    /**
     * Sets the framerate of the selected camera.
     * @param framerate Set number of frames per second.
     * @param camera_id An id of the updated camera.
     */
    void setFramerate(double framerate, int camera_id);

private:
    /**
     * Finds all cameras connected to the PC, selects those which correspond to
     * the left and right eye, and sets all of their parameters according to the
     * Settings. It also allocated the memory for all images in the buffer.
     */
    void initializeCamera();
    /**
     * Rapidly captures the image from the camera and saves them to the buffer.
     */
    void imageGatheringThread();

    // Size of the buffer to which the captured images are saved.
    static constexpr int IMAGE_IN_QUEUE_COUNT = 10;

    // Array of images in the buffer to which they are captured. One per eye.
    cv::Mat image_queues_[IMAGE_IN_QUEUE_COUNT][2]{};

    // Index of the most recent image in the buffer.
    int image_index_{-1};

    // Signifies whether the capturing thread is running or not.
    bool thread_running_{true};

    // A thread used to gather images from the camera.
    std::thread image_gatherer_{};

    // Handles to IDS cameras. One per eye.
    uint32_t camera_handles_[2]{};
    // Memory arrays to which the images are directly captured. One per eye.
    char *image_handles_[2]{};
    // Ids of the memory to which the images are directly captured. One per eye.
    int image_ids_[2]{};
    // Framerate of the camera.
    double framerate_{100};
    // Ids of the eye-tracking cameras connected to the PC.
    std::vector<int> camera_ids_{};
};
}// namespace et

#endif //HDRMFS_EYE_TRACKER_IDS_CAMERA_HPP