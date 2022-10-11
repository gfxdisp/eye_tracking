#include "InputVideo.hpp"

#include <iostream>
#include <filesystem>
#include <utility>

namespace et {
InputVideo::InputVideo(const std::string &input_video_path) {

    bool two_eyes{true};
    bool one_eye{true};

    for (int i = 0; i < 2; i++) {
        input_video_path_[i] =
            input_video_path + "_" + std::to_string(i) + ".mp4";
        two_eyes &=
            std::filesystem::exists(input_video_path_[i]);
    }

    one_eye &=
        std::filesystem::exists(input_video_path + ".mp4");

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

cv::Mat InputVideo::grabImage(int camera_id) {
    bool successful{video_capture_[camera_id].read(image_)};
    if (!successful) {
        video_capture_[camera_id].set(cv::CAP_PROP_POS_FRAMES, 0);
        successful = video_capture_[camera_id].read(image_);
    }

    assert(successful);
    cv::cvtColor(image_, image_, cv::COLOR_BGR2GRAY);
    return image_;
}

std::vector<int> InputVideo::getCameraIds() {
    return {0, 1};
}

void InputVideo::close() {
    for (int i = 0; i < 2; i++) {
        video_capture_[i].release();
    }
}
} // namespace et