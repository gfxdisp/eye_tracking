#ifndef INPUT_IMAGES_H
#define INPUT_IMAGES_H

#include "ImageProvider.hpp"

#include <string>
#include <vector>

namespace et {

class InputImages : public ImageProvider {
public:
    explicit InputImages(std::string &images_folder_path);
    cv::Mat grabPupilImage(int camera_id) override;
    cv::Mat grabGlintImage(int camera_id) override;

private:
    std::vector<std::string> filenames_{};
    int image_count_{0};
};

} // namespace et

#endif //INPUT_IMAGES_H
