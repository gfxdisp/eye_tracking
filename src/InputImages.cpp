#include "InputImages.hpp"

namespace et {
InputImages::InputImages(std::string &images_folder_path) {
    cv::glob(images_folder_path + "/*.png", filenames_, false);
}

cv::Mat InputImages::grabPupilImage(int camera_id) {
    if (image_count_ >= filenames_.size()) {
        return {};
    }

    image_ = cv::imread(filenames_[image_count_]);
    cv::cvtColor(image_, image_, cv::COLOR_BGR2GRAY);
    image_count_++;
    return image_;
}

cv::Mat InputImages::grabGlintImage(int camera_id) {
    return image_;
}
} // namespace et