#include "InputImages.hpp"

namespace et {
InputImages::InputImages(const std::string &images_folder_path) {
    cv::glob(images_folder_path + "/*.png", filenames_, false);
}

cv::Mat InputImages::grabImage(int camera_id) {
    if (image_count_ >= filenames_.size()) {
        // Once all images have been processed, empty image is returned.
        return {};
    }

    image_ = cv::imread(filenames_[image_count_]);
    cv::cvtColor(image_, image_, cv::COLOR_BGR2GRAY);
    image_count_++;
    return image_;
}

void InputImages::initialize() {
}

void InputImages::close() {
}

} // namespace et