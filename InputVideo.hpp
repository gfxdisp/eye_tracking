#ifndef INPUT_VIDEO_H
#define INPUT_VIDEO_H

#include "ImageProvider.hpp"

#include <opencv2/opencv.hpp>

namespace et {
class InputVideo : public ImageProvider {
public:
    explicit InputVideo(std::string &input_video_path);
    void initialize() override;
    cv::Mat grabImage() override;
    cv::Size2i getResolution() override;
    cv::Point2d getOffset() override;
    void close() override;

private:
    std::string input_video_path_{};
    cv::VideoCapture video_{};
};
}// namespace et

#endif