#include "InputVideo.hpp"

#include <iostream>
#include <filesystem>
#include <utility>

namespace et {
InputVideo::InputVideo(const std::string &input_video_path) {

    bool two_eyes_two_exposures{true};
    bool two_eyes_one_exposure{true};
    bool one_eye_two_exposures{true};
    bool one_eye_one_exposure{true};

    for (int i = 0; i < 2; i++) {
        input_pupil_video_path_[i] =
            input_video_path + "_" + std::to_string(i) + "_pupil.mp4";
        input_glint_video_path_[i] =
            input_video_path + "_" + std::to_string(i) + "_glint.mp4";
        two_eyes_two_exposures &=
            std::filesystem::exists(input_pupil_video_path_[i]);
        two_eyes_two_exposures &=
            std::filesystem::exists(input_glint_video_path_[i]);
    }
    for (int i = 0; i < 2; i++) {
        input_pupil_video_path_[i] =
            input_video_path + "_" + std::to_string(i) + ".mp4";
        two_eyes_one_exposure &=
            std::filesystem::exists(input_pupil_video_path_[i]);
    }

    one_eye_two_exposures &=
        std::filesystem::exists(input_video_path + "_pupil.mp4");
    one_eye_two_exposures &=
        std::filesystem::exists(input_video_path + "_glint.mp4");

    one_eye_one_exposure &= std::filesystem::exists(input_video_path + ".mp4");

    if (two_eyes_two_exposures) {
        for (int i = 0; i < 2; i++) {
            input_pupil_video_path_[i] =
                input_video_path + "_" + std::to_string(i) + "_pupil.mp4";
            input_glint_video_path_[i] =
                input_video_path + "_" + std::to_string(i) + "_glint.mp4";
        }
    } else if (two_eyes_one_exposure) {
        for (int i = 0; i < 2; i++) {
            input_pupil_video_path_[i] =
                input_video_path + "_" + std::to_string(i) + ".mp4";
        }
    } else if (one_eye_two_exposures) {
        input_pupil_video_path_[0] = input_video_path + "_pupil.mp4";
        input_pupil_video_path_[1] = input_pupil_video_path_[0];
        input_glint_video_path_[0] = input_video_path + "_glint.mp4";
        input_glint_video_path_[1] = input_glint_video_path_[0];
    } else if (one_eye_one_exposure) {
        input_pupil_video_path_[0] = input_video_path + ".mp4";
        input_pupil_video_path_[1] = input_pupil_video_path_[0];
    } else {
        input_pupil_video_path_[0] = input_video_path;
        input_pupil_video_path_[1] = input_pupil_video_path_[0];
    }
}

void InputVideo::initialize(bool separate_exposures) {
    for (int i = 0; i < 2; i++) {
        assert(input_pupil_video_path_[i].length() > 0);

        pupil_video_[i].open(input_pupil_video_path_[i]);
        assert(pupil_video_[i].isOpened());

        separate_exposures_ = separate_exposures;

        if (separate_exposures_) {
            glint_video_[i].open(input_glint_video_path_[i]);
            assert(glint_video_[i].isOpened());
        }
    }
}

cv::Mat InputVideo::grabPupilImage(int camera_id) {
    bool successful{pupil_video_[camera_id].read(image_)};
    if (!successful) {
        pupil_video_[camera_id].set(cv::CAP_PROP_POS_FRAMES, 0);
        successful = pupil_video_[camera_id].read(image_);
    }

    assert(successful);
    cv::cvtColor(image_, image_, cv::COLOR_BGR2GRAY);
    return image_;
}

cv::Mat InputVideo::grabGlintImage(int camera_id) {
    if (separate_exposures_) {
        bool successful{glint_video_[camera_id].read(image_)};
        if (!successful) {
            glint_video_[camera_id].set(cv::CAP_PROP_POS_FRAMES, 0);
            successful = glint_video_[camera_id].read(image_);
        }
        assert(successful);
        cv::cvtColor(image_, image_, cv::COLOR_BGR2GRAY);
    }
    return image_;
}

std::vector<int> InputVideo::getCameraIds() {
    return {0, 1};
}

void InputVideo::close() {
    for (int i = 0; i < 2; i++) {
        pupil_video_[i].release();
        if (separate_exposures_) {
            glint_video_[i].release();
        }
    }
}
} // namespace et