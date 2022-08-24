#ifndef INPUT_VIDEO_H
#define INPUT_VIDEO_H

#include "ImageProvider.hpp"

#include <opencv2/opencv.hpp>

namespace et {
class InputVideo : public ImageProvider {
public:
    explicit InputVideo(std::string input_video_path);
    InputVideo(std::string input_pupil_video_path, std::string input_glint_video_path);
    void initialize() override;
    cv::Mat grabPupilImage() override;
    cv::Mat grabGlintImage() override;
    void close() override;

private:
    std::string input_pupil_video_path_{};
    std::string input_glint_video_path_{};
    cv::VideoCapture pupil_video_{};
    cv::VideoCapture glint_video_{};
};
}// namespace et

#endif