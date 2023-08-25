#include "InputImages.hpp"

#include <iostream>

namespace et {
InputImages::InputImages(const std::string &images_folder_path) {
    cv::glob(images_folder_path + "/images/*_lights_off.jpg", pupil_filenames_, false);
    cv::glob(images_folder_path + "/images/*_lights_on.jpg", glints_filenames_, false);
}

ImageToProcess InputImages::grabImage(int camera_id) {
    if (image_count_ >= pupil_filenames_.size() || image_count_ >= glints_filenames_.size()) {
        // Once all images have been processed, empty image is returned.
        return {};
//        image_count_ = 0;
    }

    pupil_image_ = cv::imread(pupil_filenames_[image_count_]);
    cv::cvtColor(pupil_image_, pupil_image_, cv::COLOR_BGR2GRAY);
    glints_image_ = cv::imread(glints_filenames_[image_count_]);
    cv::cvtColor(glints_image_, glints_image_, cv::COLOR_BGR2GRAY);
    image_count_++;
    return {pupil_image_, glints_image_};
}

void InputImages::initialize() {
}

void InputImages::close() {
}

} // namespace et