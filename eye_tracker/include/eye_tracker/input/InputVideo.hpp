#ifndef HDRMFS_EYE_TRACKER_INPUT_VIDEO_HPP
#define HDRMFS_EYE_TRACKER_INPUT_VIDEO_HPP

#include <eye_tracker/input/ImageProvider.hpp>

#include <opencv2/opencv.hpp>

namespace et {
    class InputVideo : public ImageProvider {
    public:
        explicit InputVideo(const std::string& input_video_path, bool loop = false);

        EyeImage grabImage() override;

        void close() override;

    private:
        std::string input_video_path_{};

        cv::VideoCapture video_capture_{};

        int frame_num_{};
        bool loop_{false};
    };
} // namespace et

#endif //HDRMFS_EYE_TRACKER_INPUT_VIDEO_HPP
