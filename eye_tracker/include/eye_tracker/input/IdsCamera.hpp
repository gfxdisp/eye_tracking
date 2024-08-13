#ifndef HDRMFS_EYE_TRACKER_IDS_CAMERA_HPP
#define HDRMFS_EYE_TRACKER_IDS_CAMERA_HPP

#include <eye_tracker/input/ImageProvider.hpp>

#include <opencv2/opencv.hpp>

#include <thread>

namespace et {
    class IdsCamera : public ImageProvider {
    public:
        explicit IdsCamera(int camera_id);

        EyeImage grabImage() override;

        void close() override;

        void setExposure(double exposure) const;

        void setGamma(double gamma) const;

        void setFramerate(double framerate);

    private:
        void imageGatheringThread();

        static constexpr int IMAGE_IN_QUEUE_COUNT = 10;

        cv::Mat image_queue_[IMAGE_IN_QUEUE_COUNT]{};

        int image_index_{};

        bool thread_running_{true};

        std::thread image_gatherer_{};

        uint32_t camera_handle_{};

        char* image_handle_{};

        int image_id_{};

        double framerate_{};

        cv::Mat fake_image_{};

        bool fake_camera_{};

        int image_counter_{};
    };
} // namespace et

#endif //HDRMFS_EYE_TRACKER_IDS_CAMERA_HPP
