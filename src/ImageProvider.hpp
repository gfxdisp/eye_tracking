#ifndef HDRMFS_EYE_TRACKER_IMAGE_PROVIDER_HPP
#define HDRMFS_EYE_TRACKER_IMAGE_PROVIDER_HPP

#include "EyeEstimator.hpp"
#include "FeatureDetector.hpp"

#include <opencv2/opencv.hpp>

namespace et {
/**
 * Abstract class for image gathering.
 */
class ImageProvider {
public:
    /**
     * Initializes image gathering.
     */
    virtual void initialize() = 0;
    /**
     * Grabs a new image from the camera.
     * @param camera_id An id of the camera for which the value is returned
     * @return Camera image.
     */
    virtual cv::Mat grabImage(int camera_id) = 0;
    /**
     * Closes any open videos or camera feeds.
     */
    virtual void close() = 0;

protected:
    /**
     * Most recent image.
     */
    cv::Mat image_{};
};
} // namespace et

#endif //HDRMFS_EYE_TRACKER_IMAGE_PROVIDER_HPP