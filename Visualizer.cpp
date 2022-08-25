#include "Visualizer.hpp"
#include "Settings.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>

using namespace std::chrono_literals;

namespace et {

int Visualizer::pupil_threshold_tracker_{};
int Visualizer::glint_threshold_tracker_{};
int Visualizer::pupil_exposure_tracker_{};
int Visualizer::glint_exposure_tracker_{};

Visualizer::Visualizer(FeatureDetector *feature_detector,
                       EyeTracker *eye_tracker)
    : feature_detector_(feature_detector), eye_tracker_(eye_tracker) {
    last_frame_time_ = std::chrono::steady_clock::now();
    fps_text_ << std::fixed << std::setprecision(2);

    namedWindow(SLIDER_WINDOW_NAME.begin(), cv::WINDOW_AUTOSIZE);

    FeaturesParams *user_params = Settings::parameters.user_params;
    CameraParams *camera_params = &Settings::parameters.camera_params;

    pupil_threshold_tracker_ = user_params->pupil_threshold;
    cv::createTrackbar(PUPIL_THRESHOLD_NAME.begin(), SLIDER_WINDOW_NAME.begin(),
                       &pupil_threshold_tracker_, PUPIL_THRESHOLD_MAX,
                       Visualizer::onPupilThresholdUpdate);

    glint_threshold_tracker_ = user_params->glint_threshold;
    cv::createTrackbar(GLINT_THRESHOLD_NAME.begin(), SLIDER_WINDOW_NAME.begin(),
                       &glint_threshold_tracker_, GLINT_THRESHOLD_MAX,
                       Visualizer::onGlintThresholdUpdate);

    pupil_exposure_tracker_ = (int)round(100 * camera_params->pupil_exposure);
    cv::createTrackbar(PUPIL_EXPOSURE_NAME.begin(), SLIDER_WINDOW_NAME.begin(),
                       &pupil_exposure_tracker_, PUPIL_EXPOSURE_MAX - PUPIL_EXPOSURE_MIN,
                       Visualizer::onPupilExposureUpdate);
    cv::setTrackbarMin(PUPIL_EXPOSURE_NAME.begin(), SLIDER_WINDOW_NAME.begin(), PUPIL_EXPOSURE_MIN);
    cv::setTrackbarMax(PUPIL_EXPOSURE_NAME.begin(), SLIDER_WINDOW_NAME.begin(), PUPIL_EXPOSURE_MAX);
    cv::setTrackbarPos(PUPIL_EXPOSURE_NAME.begin(), SLIDER_WINDOW_NAME.begin(), pupil_exposure_tracker_);

    glint_exposure_tracker_ = (int)round(100 * camera_params->glint_exposure);
    cv::createTrackbar(GLINT_EXPOSURE_NAME.begin(), SLIDER_WINDOW_NAME.begin(),
                       &glint_exposure_tracker_, GLINT_EXPOSURE_MAX - GLINT_EXPOSURE_MIN,
                       Visualizer::onGlintExposureUpdate);
    cv::setTrackbarMin(GLINT_EXPOSURE_NAME.begin(), SLIDER_WINDOW_NAME.begin(), GLINT_EXPOSURE_MIN);
    cv::setTrackbarMax(GLINT_EXPOSURE_NAME.begin(), SLIDER_WINDOW_NAME.begin(), GLINT_EXPOSURE_MAX);
    cv::setTrackbarPos(GLINT_EXPOSURE_NAME.begin(), SLIDER_WINDOW_NAME.begin(), glint_exposure_tracker_);
}

void Visualizer::drawUi(const cv::Mat& image) {
    cv::cvtColor(image, image_, cv::COLOR_GRAY2BGR);
    auto led_positions = feature_detector_->getGlints();
    for (const auto &led : (*led_positions)) {
        cv::circle(image_, led, 5, cv::Scalar(0x00, 0x00, 0xFF), 2);
    }
    cv::circle(image_, feature_detector_->getPupil(),
               feature_detector_->getPupilRadius(),
               cv::Scalar(0xFF, 0x00, 0x00), 2);

    cv::circle(image_, eye_tracker_->getEyeCentrePixelPosition(), 2,
               cv::Scalar(0x00, 0xFF, 0x00), 5);

    cv::ellipse(image_, feature_detector_->getEllipse(),
                cv::Scalar(0x00, 0x00, 0xFF), 2);

    cv::putText(image_, fps_text_.str(), cv::Point2i(100, 100),
                cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(0x00, 0x00, 0xFF), 3);
}

void Visualizer::show() {
    cv::imshow(WINDOW_NAME.begin(), image_);
}

bool Visualizer::isWindowOpen() {
	return cv::getWindowProperty(WINDOW_NAME.begin(), 0) >= 0;
}

void Visualizer::calculateFramerate() {
    if (++frame_index_ == FRAMES_FOR_FPS_MEASUREMENT) {
        const std::chrono::duration<float> frame_time =
            std::chrono::steady_clock::now() - last_frame_time_;
        fps_text_.str("");// Clear contents of fps_text
        fps_text_ << 1s / (frame_time / FRAMES_FOR_FPS_MEASUREMENT);
        frame_index_ = 0;
        last_frame_time_ = std::chrono::steady_clock::now();
        total_frames_++;
        total_framerate_ += 1s / (frame_time / FRAMES_FOR_FPS_MEASUREMENT);
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

cv::Mat Visualizer::getUiImage() {
	cv::Mat bgr_image;
    cv::cvtColor(image_, bgr_image, cv::COLOR_BGR2GRAY);
	return bgr_image;
}

void Visualizer::onPupilThresholdUpdate(int, void *) {
    Settings::parameters.user_params->pupil_threshold =
        pupil_threshold_tracker_;
}

void Visualizer::onGlintThresholdUpdate(int, void *) {
    Settings::parameters.user_params->glint_threshold =
        glint_threshold_tracker_;
}

float Visualizer::getAvgFramerate() {
    return total_framerate_ / total_frames_;
}

void Visualizer::onPupilExposureUpdate(int, void *) {
    Settings::parameters.camera_params.pupil_exposure = (double)pupil_exposure_tracker_ / 100.0f;
}

void Visualizer::onGlintExposureUpdate(int, void *) {
    Settings::parameters.camera_params.glint_exposure = (double)glint_exposure_tracker_ / 100.0f;
}
}// namespace et