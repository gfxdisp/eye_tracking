#include "InputImages.hpp"

namespace et {
InputImages::InputImages(std::string &images_folder_path) {
    cv::glob(images_folder_path + "/*.png", filenames_, false);
}

cv::Mat InputImages::grabImage() {
    if (image_count_ >= filenames_.size()) {
        return {};
    }

    cv::Mat image = cv::imread(filenames_[image_count_]);
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    image_count_++;
    return image;
}

} // namespace et