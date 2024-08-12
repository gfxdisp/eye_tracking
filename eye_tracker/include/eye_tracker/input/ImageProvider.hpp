#ifndef HDRMFS_EYE_TRACKER_IMAGE_PROVIDER_HPP
#define HDRMFS_EYE_TRACKER_IMAGE_PROVIDER_HPP

#include "eye_tracker/Settings.hpp"

#include <opencv2/opencv.hpp>

namespace et {
    struct EyeImage {
        cv::Mat frame;
        int frame_num{0};
    };

    /**
     * Abstract class for image gathering.
     */
    class ImageProvider {
    public:
        virtual ~ImageProvider() = default;

        ImageProvider() = default;

        /**
         * Grabs a new image from the camera.
         * @param camera_id An id of the camera for which the value is returned
         * @return A pair of images: one for detecting pupil and one for detecting glints.
         */
        virtual EyeImage grabImage() = 0;

        /**
         * Closes any open videos or camera feeds.
         */
        virtual void close() = 0;

    protected:
        /**
         * Most recent image.
         */
        cv::Mat frame_{};
        CameraParams* camera_params_{};
        FeaturesParams* user_params_{};
    };
} // namespace et

#endif //HDRMFS_EYE_TRACKER_IMAGE_PROVIDER_HPP
