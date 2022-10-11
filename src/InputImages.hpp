#ifndef INPUT_IMAGES_H
#define INPUT_IMAGES_H

#include "ImageProvider.hpp"

#include <string>
#include <vector>

namespace et {

class InputImages : public ImageProvider {
public:
    explicit InputImages(std::string &images_folder_path);
    void initialize() override;
    void close() override;
    cv::Mat grabImage(int camera_id) override;
    std::vector<int> getCameraIds() override;

private:
    std::vector<std::string> filenames_{};
    int image_count_{0};
};

} // namespace et

#endif //INPUT_IMAGES_H
