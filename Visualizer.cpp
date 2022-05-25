#include "Visualizer.hpp"
#include "Settings.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>

using namespace std::chrono_literals;

namespace et {

int Visualizer::pupil_threshold_tracker_{};
int Visualizer::glint_threshold_tracker_{};

Visualizer::Visualizer(FeatureDetector *feature_detector,
                       EyeTracker *eye_tracker)
    : feature_detector_(feature_detector), eye_tracker_(eye_tracker) {
    last_frame_time_ = std::chrono::steady_clock::now();
    fps_text_ << std::fixed << std::setprecision(2);

    namedWindow(SLIDER_WINDOW_NAME.begin(), cv::WINDOW_AUTOSIZE);

    FeaturesParams *user_params = Settings::parameters.user_params;

    pupil_threshold_tracker_ = user_params->pupil_threshold;
    cv::createTrackbar(PUPIL_THRESHOLD_NAME.begin(), SLIDER_WINDOW_NAME.begin(),
                       &pupil_threshold_tracker_, PUPIL_THRESHOLD_MAX,
                       Visualizer::onPupilThresholdUpdate);

    glint_threshold_tracker_ = user_params->glint_threshold;
    cv::createTrackbar(GLINT_THRESHOLD_NAME.begin(), SLIDER_WINDOW_NAME.begin(),
                       &glint_threshold_tracker_, GLINT_THRESHOLD_MAX,
                       Visualizer::onGlintThresholdUpdate);
}

void Visualizer::drawUi(cv::Mat image) {
    cv::cvtColor(image, image_, cv::COLOR_GRAY2BGR);
    auto led_positions = feature_detector_->getGlints();
    for (const auto &led : (*led_positions)) {
        cv::circle(image_, led, 5, cv::Scalar(0x00, 0x00, 0xFF), 2);
    }
    cv::circle(image_, feature_detector_->getPupil(),
               feature_detector_->getPupilRadius(),
               cv::Scalar(0xFF, 0x00, 0x00), 2);

    cv::circle(image_, eye_tracker_->getCorneaCurvaturePixelPosition(), 2,
               cv::Scalar(0x00, 0xFF, 0x00), 5);

    cv::putText(image_, fps_text_.str(), cv::Point2i(100, 100),
                cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(0x00, 0x00, 0xFF), 3);
}

void Visualizer::show() {
    cv::imshow(WINDOW_NAME.begin(), image_);
}

void Visualizer::calculateFramerate() {
    if (++frame_index_ == FRAMES_FOR_FPS_MEASUREMENT) {
        const std::chrono::duration<float> frame_time =
            std::chrono::steady_clock::now() - last_frame_time_;
        fps_text_.str("");// Clear contents of fps_text
        fps_text_ << 1s / (frame_time / FRAMES_FOR_FPS_MEASUREMENT);
        frame_index_ = 0;
        last_frame_time_ = std::chrono::steady_clock::now();
    }
}

void Visualizer::printFramerateInterval() {
    static auto start{std::chrono::steady_clock::now()};
    auto current{std::chrono::steady_clock::now()};
    if (current - start > 1s) {
        start = current;
        std::clog << "Frames per second: " << fps_text_.str() << "\n";
    }
}

void Visualizer::onPupilThresholdUpdate(int, void *) {
    Settings::parameters.user_params->pupil_threshold =
        pupil_threshold_tracker_;
}

void Visualizer::onGlintThresholdUpdate(int, void *) {
    Settings::parameters.user_params->glint_threshold =
        glint_threshold_tracker_;
}
}// namespace et