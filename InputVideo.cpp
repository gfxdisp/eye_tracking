#include "InputVideo.hpp"

#include <utility>

namespace et {
InputVideo::InputVideo(std::string input_video_path)
    : input_pupil_video_path_(std::move(input_video_path)) {
    input_glint_video_path_ = input_pupil_video_path_;
}

InputVideo::InputVideo(std::string input_pupil_video_path,
                       std::string glint_pupil_video_path)
    : input_pupil_video_path_(std::move(input_pupil_video_path)),
      input_glint_video_path_(std::move(glint_pupil_video_path)) {
}

void InputVideo::initialize(bool separate_exposures) {
    assert(input_pupil_video_path_.length() > 0);
    assert(input_glint_video_path_.length() > 0);

    pupil_video_.open(input_pupil_video_path_);
    assert(pupil_video_.isOpened());

    if (input_pupil_video_path_ != input_glint_video_path_) {
        glint_video_.open(input_glint_video_path_);
        assert(glint_video_.isOpened());
    }
}

cv::Mat InputVideo::grabPupilImage() {
    bool successful{pupil_video_.read(image_)};
    if (!successful) {
        pupil_video_.set(cv::CAP_PROP_POS_FRAMES, 0);
        successful = pupil_video_.read(image_);
    }

    assert(successful);
    cv::cvtColor(image_, image_, cv::COLOR_BGR2GRAY);
    return image_;
}

cv::Mat InputVideo::grabGlintImage() {
    if (input_pupil_video_path_ != input_glint_video_path_) {
        bool successful{glint_video_.read(image_)};
        if (!successful) {
            glint_video_.set(cv::CAP_PROP_POS_FRAMES, 0);
            successful = glint_video_.read(image_);
        }
        assert(successful);
        cv::cvtColor(image_, image_, cv::COLOR_BGR2GRAY);
    }
    return image_;
}

void InputVideo::close() {
    pupil_video_.release();
    if (input_pupil_video_path_ != input_glint_video_path_) {
        glint_video_.release();
    }
}
} // namespace et