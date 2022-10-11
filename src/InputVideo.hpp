#ifndef INPUT_VIDEO_H
#define INPUT_VIDEO_H

#include "ImageProvider.hpp"

#include <opencv2/opencv.hpp>

namespace et {
class InputVideo : public ImageProvider {
public:
    explicit InputVideo(const std::string& input_video_path);
    void initialize() override;
    cv::Mat grabImage(int camera_id) override;
    void close() override;
    std::vector<int> getCameraIds() override;

private:
    std::string input_video_path_[2]{};
    cv::VideoCapture video_capture_[2]{};
};
}// namespace et

#endif