#include "eye_tracker/input/InputVideo.hpp"

#include <filesystem>
#include <iostream>

namespace et {
    InputVideo::InputVideo(const std::string& input_video_path, bool loop) : loop_{loop} {
        input_video_path_ = input_video_path;
        assert(input_video_path_.length() > 0);

        video_capture_.open(input_video_path_);
        assert(video_capture_.isOpened());
    }

    EyeImage InputVideo::grabImage() {
        const bool successful{video_capture_.read(frame_)};
        if (!successful) {
            if (loop_) {
                video_capture_.set(cv::CAP_PROP_POS_FRAMES, 0);
                video_capture_.read(frame_);
            } else {
                return {};
            }
        }

        cv::cvtColor(frame_, frame_, cv::COLOR_BGR2GRAY);
        return {frame_, frame_num_++};
    }

    void InputVideo::close() {
        video_capture_.release();
    }
} // namespace et
