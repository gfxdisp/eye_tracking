#include "InputVideo.hpp"

#include <filesystem>
#include <iostream>

namespace et {
InputVideo::InputVideo(const std::string &input_video_path) {

    bool two_eyes{true};
    bool one_eye{true};

    for (int i = 0; i < 2; i++) {
        input_video_path_[i] =
            input_video_path + "_" + std::to_string(i) + ".mp4";
        two_eyes &= std::filesystem::exists(input_video_path_[i]);
    }

    one_eye &= std::filesystem::exists(input_video_path + ".mp4");

    if (two_eyes) {
        for (int i = 0; i < 2; i++) {
            input_video_path_[i] =
                input_video_path + "_" + std::to_string(i) + ".mp4";
        }
    } else if (one_eye) {
        input_video_path_[0] = input_video_path + ".mp4";
        input_video_path_[1] = input_video_path_[0];
    } else {
        input_video_path_[0] = input_video_path;
        input_video_path_[1] = input_video_path_[0];
    }
}

void InputVideo::initialize() {
    for (int i = 0; i < 2; i++) {
        assert(input_video_path_[i].length() > 0);

        video_capture_[i].open(input_video_path_[i]);
        assert(video_capture_[i].isOpened());
    }
}

ImageToProcess InputVideo::grabImage(int camera_id) {
    bool successful{video_capture_[camera_id].read(pupil_image_)};
    if (!successful) {
        video_capture_[camera_id].set(cv::CAP_PROP_POS_FRAMES, 0);
        video_capture_[camera_id].read(pupil_image_);
    }

    cv::cvtColor(pupil_image_, pupil_image_, cv::COLOR_BGR2GRAY);
    glints_image_ = pupil_image_;
    return {pupil_image_, glints_image_};
}

void InputVideo::close() {
    for (auto & i : video_capture_) {
        i.release();
    }
}
} // namespace et