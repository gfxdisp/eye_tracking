#include "eye_tracker/Visualizer.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>

using namespace std::chrono_literals;

namespace et {
    Visualizer::Visualizer(const int camera_id, const bool headless) : headless_(headless) {
        last_frame_time_ = std::chrono::steady_clock::now();
        fps_text_ << std::fixed << std::setprecision(2);

        full_output_window_name_ = SIDE_NAMES[camera_id].begin();
        full_output_window_name_ += WINDOW_NAME.begin();

        typedef void (* TrackerPointer)(int, void*);

        const TrackerPointer trackers[] = {&Visualizer::onPupilThresholdUpdate, &Visualizer::onGlintThresholdUpdate, &Visualizer::onExposureUpdate};

        pupil_threshold_ = &Settings::parameters.user_params[camera_id]->pupil_threshold;
        glint_threshold_ = &Settings::parameters.user_params[camera_id]->glint_threshold;
        user_params_ = Settings::parameters.user_params[camera_id];

        if (!headless) {
            namedWindow(full_output_window_name_, cv::WINDOW_AUTOSIZE);

            cv::createTrackbar(PUPIL_THRESHOLD_NAME.begin(), full_output_window_name_, nullptr, PUPIL_THRESHOLD_MAX, trackers[0], this);
            cv::setTrackbarPos(PUPIL_THRESHOLD_NAME.begin(), full_output_window_name_, *pupil_threshold_);

            cv::createTrackbar(GLINT_THRESHOLD_NAME.begin(), full_output_window_name_, nullptr, GLINT_THRESHOLD_MAX, trackers[1], this);
            cv::setTrackbarPos(GLINT_THRESHOLD_NAME.begin(), full_output_window_name_, *glint_threshold_);

            cv::createTrackbar(EXPOSURE_NAME.begin(), full_output_window_name_, nullptr, EXPOSURE_MAX - EXPOSURE_MIN, trackers[2], this);
            cv::setTrackbarMin(EXPOSURE_NAME.begin(), full_output_window_name_, EXPOSURE_MIN);
            cv::setTrackbarMax(EXPOSURE_NAME.begin(), full_output_window_name_, EXPOSURE_MAX);
            cv::setTrackbarPos(EXPOSURE_NAME.begin(), full_output_window_name_, static_cast<int>(round(100.0 * user_params_->exposure)));
        }

        framerate_timer_ = std::chrono::steady_clock::now();
        cornea_history_index_ = 0;
        gaze_point_index_ = 0;
        cornea_history_full_ = false;
        gaze_point_history_full_ = false;
    }

    void Visualizer::prepareImage(const cv::Mat& image) {
        cv::cvtColor(image, image_, cv::COLOR_GRAY2BGR);
    }

    void Visualizer::show() const {
        if (!image_.empty() && !headless_) {
            cv::imshow(full_output_window_name_, image_);
        }
    }

    bool Visualizer::isWindowOpen() const {
        if (!image_.empty() && cv::getWindowProperty(full_output_window_name_, cv::WND_PROP_AUTOSIZE) < 0) {
            return false;
        }
        return true;
    }

    void Visualizer::calculateFramerate() {
        if (++frame_index_ == FRAMES_FOR_FPS_MEASUREMENT) {
            const std::chrono::duration<double> frame_time = std::chrono::steady_clock::now() - last_frame_time_;
            fps_text_.str(""); // Clear contents of fps_text
            fps_text_ << 1s / (frame_time / FRAMES_FOR_FPS_MEASUREMENT);
            frame_index_ = 0;
            last_frame_time_ = std::chrono::steady_clock::now();
            total_frames_++;
            total_framerate_ += 1s / (frame_time / FRAMES_FOR_FPS_MEASUREMENT);
        }
    }

    void Visualizer::printFramerateInterval() {
        const auto current{std::chrono::steady_clock::now()};
        if (current - framerate_timer_ > 1s) {
            framerate_timer_ = current;
            std::clog << "[" << full_output_window_name_ << "] Frames per second: " << fps_text_.str() << "\n";
        }
    }

    cv::Mat Visualizer::getUiImage() const {
        cv::Mat image = image_.clone();
        return image;
    }

    void Visualizer::onPupilThresholdUpdate(const int value, void* ptr) {
        const auto* visualizer = static_cast<Visualizer*>(ptr);
        visualizer->onPupilThresholdUpdate(value);
    }

    void Visualizer::onPupilThresholdUpdate(const int value) const {
        *pupil_threshold_ = value;
        Settings::saveSettings();
    }

    void Visualizer::onGlintThresholdUpdate(const int value, void* ptr) {
        const auto* visualizer = static_cast<Visualizer*>(ptr);
        visualizer->onGlintThresholdUpdate(value);
    }

    void Visualizer::onGlintThresholdUpdate(const int value) const {
        *glint_threshold_ = value;
        Settings::saveSettings();
    }

    void Visualizer::onExposureUpdate(const int value, void* ptr) {
        const auto* visualizer = static_cast<Visualizer*>(ptr);
        visualizer->onExposureUpdate(value);
    }

    void Visualizer::onExposureUpdate(const int value) const {
        // Scales exposure to millimeters
        user_params_->exposure = static_cast<double>(value) / 100.0;
    }

    void Visualizer::drawPupil(const cv::Point2d pupil, const int radius, const double diameter_mm) {
        cv::circle(image_, pupil, radius, cv::Scalar(0xFF, 0xFF, 0x00), 2);
        const std::string text = cv::format("%.1f mm", diameter_mm);
        cv::putText(image_, text, cv::Point2i(pupil.x - radius, pupil.y - 10 - radius), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0xFF, 0xFF, 0x00), 2, cv::LINE_AA);
    }

    void Visualizer::drawGlints(const std::vector<cv::Point2d>& glints, std::vector<bool>& glints_validity) {
        for (int i = 0; i < glints.size(); i++) {
            if (glints_validity.at(i)) {
                cv::circle(image_, glints[i], 5, cv::Scalar(0x00, 0xFF, 0x00), 2);
            } else {
                cv::line(image_, cv::Point2d(glints[i].x - 3.5, glints[i].y - 3.5), cv::Point2d(glints[i].x + 3.5, glints[i].y + 3.5), cv::Scalar(0x00, 0x00, 0xFF), 2);
                cv::line(image_, cv::Point2d(glints[i].x + 3.5, glints[i].y - 3.5), cv::Point2d(glints[i].x - 3.5, glints[i].y + 3.5), cv::Scalar(0x00, 0x00, 0xFF), 2);
            }
        }
    }

    void Visualizer::drawBoundingCircle(const cv::Point2d centre, const int radius) {
        cv::circle(image_, centre, radius, cv::Scalar(0xFF, 0xFF, 0x00), 1);
    }

    void Visualizer::drawEyeCentre(const cv::Point2d eye_centre) {
        cv::circle(image_, eye_centre, 2, cv::Scalar(0x00, 0xFF, 0xFF), 5);
    }

    void Visualizer::drawCorneaCentre(const cv::Point2d cornea_centre) {
        cv::circle(image_, cornea_centre, 2, cv::Scalar(0x00, 0x80, 0x00), 5);

        previous_cornea_centres_[cornea_history_index_] = cornea_centre;
        cornea_history_index_++;
        if (cornea_history_index_ >= CORNEA_HISTORY_SIZE) {
            cornea_history_index_ = 0;
            cornea_history_full_ = true;
        }
    }

    void Visualizer::drawGaze(const cv::Point2d normalized_gaze_point) {
        cv::putText(image_, "Gaze position", cv::Point2i(game_window_offset, image_.rows - game_window_offset - game_window_size - 25), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0xFF, 0xFF, 0xFF), 2, cv::LINE_AA);
        cv::rectangle(image_, cv::Rect(game_window_offset, image_.rows - game_window_offset - game_window_size, game_window_size, game_window_size), cv::Scalar(0xFF, 0xFF, 0xFF), cv::LINE_4);

        // Draw gaze point in the rectangle
        if (normalized_gaze_point.x >= 0 && normalized_gaze_point.x <= 1 && normalized_gaze_point.y >= 0 && normalized_gaze_point.y <= 1) {
            cv::circle(image_, cv::Point2d(game_window_offset + game_window_size * normalized_gaze_point.x, image_.rows - game_window_offset - game_window_size * normalized_gaze_point.y), 3, cv::Scalar(0x00, 0x00, 0xFF), 3);

            previous_gaze_points_[gaze_point_index_] = normalized_gaze_point;
            gaze_point_index_++;
            if (gaze_point_index_ >= GAZE_HISTORY_SIZE) {
                gaze_point_index_ = 0;
                gaze_point_history_full_ = true;
            }
        }
    }

    void Visualizer::drawMarker(const cv::Point2d normalized_marker) {
        if (normalized_marker.x >= 0 && normalized_marker.x <= 1 && normalized_marker.y >= 0 && normalized_marker.y <= 1) {
            cv::line(image_, cv::Point2d(game_window_offset + game_window_size * normalized_marker.x - 10, image_.rows - game_window_offset - game_window_size * normalized_marker.y), cv::Point2d(game_window_offset + game_window_size * normalized_marker.x + 10, image_.rows - game_window_offset - game_window_size * normalized_marker.y), cv::Scalar(0x00, 0x00, 0xFF), 2);
            cv::line(image_, cv::Point2d(game_window_offset + game_window_size * normalized_marker.x, image_.rows - game_window_offset - game_window_size * normalized_marker.y - 10), cv::Point2d(game_window_offset + game_window_size * normalized_marker.x, image_.rows - game_window_offset - game_window_size * normalized_marker.y + 10), cv::Scalar(0x00, 0x00, 0xFF), 2);
        }
    }

    void Visualizer::drawGazeTrace() {
        int start = gaze_point_history_full_ ? (gaze_point_index_ + 1) % GAZE_HISTORY_SIZE : 0;
        for (int i = start; (i + 1) % GAZE_HISTORY_SIZE != gaze_point_index_; i = (i + 1) % GAZE_HISTORY_SIZE) {
            auto full_pos_start = cv::Point2d(game_window_offset + game_window_size * previous_gaze_points_[i].x, image_.rows - game_window_offset - game_window_size * previous_gaze_points_[i].y);
            auto full_pos_end = cv::Point2d(game_window_offset + game_window_size * previous_gaze_points_[(i + 1) % GAZE_HISTORY_SIZE].x, image_.rows - game_window_offset - game_window_size * previous_gaze_points_[(i + 1) % GAZE_HISTORY_SIZE].y);
            cv::line(image_, full_pos_start, full_pos_end, cv::Scalar(0xFF, 0x00, 0x00), 2);
        }
    }

    void Visualizer::drawGlintEllipse(const cv::RotatedRect& ellipse) {
        cv::ellipse(image_, ellipse, cv::Scalar(0xFF, 0xFF, 0x00), 1);
    }

    void Visualizer::drawCorneaTrace() {
        int start = cornea_history_full_ ? (cornea_history_index_ + 1) % CORNEA_HISTORY_SIZE : 0;
        for (int i = start; (i + 1) % CORNEA_HISTORY_SIZE != cornea_history_index_; i = (i + 1) % CORNEA_HISTORY_SIZE) {
            cv::line(image_, previous_cornea_centres_[i], previous_cornea_centres_[(i + 1) % CORNEA_HISTORY_SIZE], cv::Scalar(0x00, 0x80, 0x00), 2);
        }
    }

    void Visualizer::drawFps() {
        cv::putText(image_, fps_text_.str(), cv::Point2i(100, 100), cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(0x00, 0x00, 0xFF), 3);
    }
} // namespace et
