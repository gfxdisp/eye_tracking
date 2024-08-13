#ifndef HDRMFS_EYE_TRACKER_IMAGE_PROVIDER_HPP
#define HDRMFS_EYE_TRACKER_IMAGE_PROVIDER_HPP

#include <eye_tracker/Settings.hpp>

#include <opencv2/opencv.hpp>

namespace et {
    struct EyeImage {
        cv::Mat frame;
        int frame_num{0};
    };

    class ImageProvider {
    public:
        virtual ~ImageProvider() = default;

        ImageProvider() = default;

        virtual EyeImage grabImage() = 0;

        virtual void close() = 0;

    protected:
        cv::Mat frame_{};
        CameraParams* camera_params_{};
        FeaturesParams* user_params_{};
    };
} // namespace et

#endif //HDRMFS_EYE_TRACKER_IMAGE_PROVIDER_HPP
