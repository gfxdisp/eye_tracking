#include "Visualizer.hpp"
#include "Settings.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>

using namespace std::chrono_literals;

namespace et {

int Visualizer::pupil_threshold_tracker_[]{};
int Visualizer::glint_threshold_tracker_[]{};
int Visualizer::pupil_exposure_tracker_[]{};
int Visualizer::glint_exposure_tracker_[]{};

Visualizer::Visualizer(FeatureDetector *feature_detector,
                       EyeTracker *eye_tracker)
    : feature_detector_(feature_detector), eye_tracker_(eye_tracker) {
    last_frame_time_ = std::chrono::steady_clock::now();
    fps_text_ << std::fixed << std::setprecision(2);

    typedef void (*TrackerPointer)(int, void *);

    TrackerPointer left_trackers[] = {Visualizer::onPupilLeftThresholdUpdate,
                                      Visualizer::onGlintLeftThresholdUpdate,
                                      Visualizer::onPupilLeftExposureUpdate,
                                      Visualizer::onGlintLeftExposureUpdate};
    TrackerPointer right_trackers[] = {Visualizer::onPupilRightThresholdUpdate,
                                       Visualizer::onGlintRightThresholdUpdate,
                                       Visualizer::onPupilRightExposureUpdate,
                                       Visualizer::onGlintRightExposureUpdate};

    TrackerPointer *all_trackers[] = {left_trackers, right_trackers};

    FeaturesParams *user_params = Settings::parameters.user_params;
    for (int i = 0; i < 2; i++) {
        TrackerPointer *trackbar_pointers = all_trackers[i];
        CameraParams *camera_params = &Settings::parameters.camera_params[i];
        namedWindow(SLIDER_WINDOW_NAME[i].begin(), cv::WINDOW_AUTOSIZE);

        pupil_threshold_tracker_[i] = user_params->pupil_threshold[i];
        cv::createTrackbar(PUPIL_THRESHOLD_NAME.begin(),
                           SLIDER_WINDOW_NAME[i].begin(), nullptr,
                           PUPIL_THRESHOLD_MAX, trackbar_pointers[0]);
        cv::setTrackbarPos(PUPIL_THRESHOLD_NAME.begin(),
                           SLIDER_WINDOW_NAME[i].begin(),
                           user_params->pupil_threshold[i]);

        glint_threshold_tracker_[i] = user_params->glint_threshold[i];
        cv::createTrackbar(GLINT_THRESHOLD_NAME.begin(),
                           SLIDER_WINDOW_NAME[i].begin(), nullptr,
                           GLINT_THRESHOLD_MAX, trackbar_pointers[1]);
        cv::setTrackbarPos(GLINT_THRESHOLD_NAME.begin(),
                           SLIDER_WINDOW_NAME[i].begin(),
                           user_params->glint_threshold[i]);

        pupil_exposure_tracker_[i] =
            (int) round(100 * camera_params->pupil_exposure);
        cv::createTrackbar(
            PUPIL_EXPOSURE_NAME.begin(), SLIDER_WINDOW_NAME[i].begin(), nullptr,
            PUPIL_EXPOSURE_MAX - PUPIL_EXPOSURE_MIN, trackbar_pointers[2]);
        cv::setTrackbarMin(PUPIL_EXPOSURE_NAME.begin(),
                           SLIDER_WINDOW_NAME[i].begin(), PUPIL_EXPOSURE_MIN);
        cv::setTrackbarMax(PUPIL_EXPOSURE_NAME.begin(),
                           SLIDER_WINDOW_NAME[i].begin(), PUPIL_EXPOSURE_MAX);
        cv::setTrackbarPos(PUPIL_EXPOSURE_NAME.begin(),
                           SLIDER_WINDOW_NAME[i].begin(),
                           pupil_exposure_tracker_[i]);

        glint_exposure_tracker_[i] =
            (int) round(100 * camera_params->glint_exposure);
        cv::createTrackbar(
            GLINT_EXPOSURE_NAME.begin(), SLIDER_WINDOW_NAME[i].begin(), nullptr,
            GLINT_EXPOSURE_MAX - GLINT_EXPOSURE_MIN, trackbar_pointers[3]);
        cv::setTrackbarMin(GLINT_EXPOSURE_NAME.begin(),
                           SLIDER_WINDOW_NAME[i].begin(), GLINT_EXPOSURE_MIN);
        cv::setTrackbarMax(GLINT_EXPOSURE_NAME.begin(),
                           SLIDER_WINDOW_NAME[i].begin(), GLINT_EXPOSURE_MAX);
        cv::setTrackbarPos(GLINT_EXPOSURE_NAME.begin(),
                           SLIDER_WINDOW_NAME[i].begin(),
                           glint_exposure_tracker_[i]);
    }
}

void Visualizer::drawUi(const cv::Mat &image, int camera_id) {
    cv::cvtColor(image, image_[camera_id], cv::COLOR_GRAY2BGR);
    auto led_positions = feature_detector_->getGlints(camera_id);
    for (const auto &led : (*led_positions)) {
        cv::circle(image_[camera_id], led, 5, cv::Scalar(0x00, 0x00, 0xFF), 2);
    }
    cv::circle(image_[camera_id], feature_detector_->getPupil(camera_id),
               feature_detector_->getPupilRadius(camera_id),
               cv::Scalar(0xFF, 0xFF, 0x00), 2);

    cv::circle(
        image_[camera_id],
        Settings::parameters.detection_params.pupil_search_centre[camera_id],
        Settings::parameters.detection_params.pupil_search_radius[camera_id],
        cv::Scalar(0xFF, 0xFF, 0x00), 2);

    cv::circle(image_[camera_id],
               eye_tracker_->getEyeCentrePixelPosition(camera_id), 2,
               cv::Scalar(0x00, 0xFF, 0x00), 5);

    cv::ellipse(image_[camera_id], feature_detector_->getEllipse(camera_id),
                cv::Scalar(0xFF, 0xFF, 0x00), 2);
    cv::putText(image_[camera_id], fps_text_.str(), cv::Point2i(100, 100),
                cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(0x00, 0x00, 0xFF), 3);
}

void Visualizer::show() {
    for (int i = 0; i < 2; i++) {
        if (!image_[i].empty()) {
            cv::imshow(WINDOW_NAME[i].begin(), image_[i]);
        }
    }
}

bool Visualizer::isWindowOpen() {
    for (int i = 0; i < 2; i++) {
        if (!image_[i].empty()
            && cv::getWindowProperty(WINDOW_NAME[i].begin(),
                                     cv::WND_PROP_AUTOSIZE)
                < 0) {
            return false;
        }
    }
    return true;
}

void Visualizer::calculateFramerate() {
    if (++frame_index_ == FRAMES_FOR_FPS_MEASUREMENT) {
        const std::chrono::duration<float> frame_time =
            std::chrono::steady_clock::now() - last_frame_time_;
        fps_text_.str(""); // Clear contents of fps_text
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

cv::Mat Visualizer::getUiImage(int camera_id) {
    cv::Mat bgr_image;
    cv::cvtColor(image_[camera_id], bgr_image, cv::COLOR_BGR2GRAY);
    return bgr_image;
}

float Visualizer::getAvgFramerate() {
    return total_framerate_ / total_frames_;
}

void Visualizer::onPupilLeftThresholdUpdate(int value, void *) {
    Settings::parameters.user_params->pupil_threshold[0] = value;
}

void Visualizer::onGlintLeftThresholdUpdate(int value, void *) {
    Settings::parameters.user_params->glint_threshold[0] = value;
}

void Visualizer::onPupilLeftExposureUpdate(int value, void *) {
    Settings::parameters.camera_params[0].pupil_exposure =
        (double) value / 100.0f;
}

void Visualizer::onGlintLeftExposureUpdate(int value, void *) {
    Settings::parameters.camera_params[0].glint_exposure =
        (double) value / 100.0f;
}

void Visualizer::onPupilRightThresholdUpdate(int value, void *) {
    Settings::parameters.user_params->pupil_threshold[1] = value;
}

void Visualizer::onGlintRightThresholdUpdate(int value, void *) {
    Settings::parameters.user_params->glint_threshold[1] = value;
}

void Visualizer::onPupilRightExposureUpdate(int value, void *) {
    Settings::parameters.camera_params[1].pupil_exposure =
        (double) value / 100.0f;
}

void Visualizer::onGlintRightExposureUpdate(int value, void *) {
    Settings::parameters.camera_params[1].glint_exposure =
        (double) value / 100.0f;
}

} // namespace et