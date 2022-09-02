#ifndef INPUT_VIDEO_H
#define INPUT_VIDEO_H

#include "ImageProvider.hpp"

#include <opencv2/opencv.hpp>

namespace et {
class InputVideo : public ImageProvider {
public:
    explicit InputVideo(const std::string& input_video_path);
    void initialize(bool separate_exposures) override;
    cv::Mat grabPupilImage(int camera_id) override;
    cv::Mat grabGlintImage(int camera_id) override;
    void close() override;
    std::vector<int> getCameraIds() override;

private:
    std::string input_pupil_video_path_[2]{};
    std::string input_glint_video_path_[2]{};
    cv::VideoCapture pupil_video_[2]{};
    cv::VideoCapture glint_video_[2]{};
};
}// namespace et

#endif