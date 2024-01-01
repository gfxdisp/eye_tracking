#include "eye_tracker/Visualizer.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>

using namespace std::chrono_literals;

namespace et
{

    Visualizer::Visualizer(int camera_id, bool headless) : headless_(headless)
    {
        last_frame_time_ = std::chrono::steady_clock::now();
        fps_text_ << std::fixed << std::setprecision(2);

        full_output_window_name_ = SIDE_NAMES[camera_id].begin();
        full_output_window_name_ += WINDOW_NAME.begin();

        typedef void (*TrackerPointer)(int, void *);

        TrackerPointer trackers[] = {&Visualizer::onPupilThresholdUpdate, &Visualizer::onGlintThresholdUpdate,
                                     &Visualizer::onExposureUpdate};

        pupil_threshold_ = &Settings::parameters.user_params[camera_id]->pupil_threshold;
        glint_threshold_ = &Settings::parameters.user_params[camera_id]->glint_threshold;
        user_params_ = Settings::parameters.user_params[camera_id];

        if (!headless)
        {
            namedWindow(full_output_window_name_, cv::WINDOW_AUTOSIZE);

            cv::createTrackbar(PUPIL_THRESHOLD_NAME.begin(), full_output_window_name_, nullptr, PUPIL_THRESHOLD_MAX,
                               trackers[0], this);
            cv::setTrackbarPos(PUPIL_THRESHOLD_NAME.begin(), full_output_window_name_, *pupil_threshold_);

            cv::createTrackbar(GLINT_THRESHOLD_NAME.begin(), full_output_window_name_, nullptr, GLINT_THRESHOLD_MAX,
                               trackers[1], this);
            cv::setTrackbarPos(GLINT_THRESHOLD_NAME.begin(), full_output_window_name_, *glint_threshold_);

            cv::createTrackbar(EXPOSURE_NAME.begin(), full_output_window_name_, nullptr, EXPOSURE_MAX - EXPOSURE_MIN,
                               trackers[2], this);
            cv::setTrackbarMin(EXPOSURE_NAME.begin(), full_output_window_name_, EXPOSURE_MIN);
            cv::setTrackbarMax(EXPOSURE_NAME.begin(), full_output_window_name_, EXPOSURE_MAX);
            cv::setTrackbarPos(EXPOSURE_NAME.begin(), full_output_window_name_,
                               (int) round(100.0 * user_params_->exposure));
        }

        framerate_timer_ = std::chrono::steady_clock::now();
    }

    void Visualizer::prepareImage(const cv::Mat &image)
    {
        cv::cvtColor(image, image_, cv::COLOR_GRAY2BGR);
    }

    void Visualizer::show()
    {
        if (!image_.empty() && !headless_)
        {
            cv::imshow(full_output_window_name_, image_);
        }
    }

    bool Visualizer::isWindowOpen()
    {
        if (!image_.empty() && cv::getWindowProperty(full_output_window_name_, cv::WND_PROP_AUTOSIZE) < 0)
        {
            return false;
        }
        return true;
    }

    void Visualizer::calculateFramerate()
    {
        if (++frame_index_ == FRAMES_FOR_FPS_MEASUREMENT)
        {
            const std::chrono::duration<double> frame_time = std::chrono::steady_clock::now() - last_frame_time_;
            fps_text_.str(""); // Clear contents of fps_text
            fps_text_ << 1s / (frame_time / FRAMES_FOR_FPS_MEASUREMENT);
            frame_index_ = 0;
            last_frame_time_ = std::chrono::steady_clock::now();
            total_frames_++;
            total_framerate_ += 1s / (frame_time / FRAMES_FOR_FPS_MEASUREMENT);
        }
    }

    void Visualizer::printFramerateInterval()
    {
        auto current{std::chrono::steady_clock::now()};
        if (current - framerate_timer_ > 1s)
        {
            framerate_timer_ = current;
            std::clog << "[" << full_output_window_name_ << "] Frames per second: " << fps_text_.str() << "\n";
        }
    }

    cv::Mat Visualizer::getUiImage()
    {
        cv::Mat image = image_.clone();
        return image;
    }

    double Visualizer::getAvgFramerate()
    {
        return total_framerate_ / (double) total_frames_;
    }

    void Visualizer::onPupilThresholdUpdate(int value, void *ptr)
    {
        auto *visualizer = (Visualizer *) ptr;
        visualizer->onPupilThresholdUpdate(value);
    }

    void Visualizer::onPupilThresholdUpdate(int value)
    {
        *pupil_threshold_ = value;
        Settings::saveSettings();
    }

    void Visualizer::onGlintThresholdUpdate(int value, void *ptr)
    {
        auto *visualizer = (Visualizer *) ptr;
        visualizer->onGlintThresholdUpdate(value);
    }

    void Visualizer::onGlintThresholdUpdate(int value)
    {
        *glint_threshold_ = value;
        Settings::saveSettings();
    }

    void Visualizer::onExposureUpdate(int value, void *ptr)
    {
        auto *visualizer = (Visualizer *) ptr;
        visualizer->onExposureUpdate(value);
    }

    void Visualizer::onExposureUpdate(int value)
    {
        // Scales exposure to millimeters
        user_params_->exposure = (double) value / 100.0;
    }

    void Visualizer::drawPupil(cv::Point2d pupil, int radius)
    {
        cv::circle(image_, pupil, radius, cv::Scalar(0xFF, 0xFF, 0x00), 1);
    }

    void Visualizer::drawGlints(std::vector<cv::Point2d> *glints)
    {
        for (const auto &glint: (*glints))
        {
            cv::circle(image_, glint, 5, cv::Scalar(0x00, 0x00, 0xFF), 1);
        }
    }

    void Visualizer::drawBoundingCircle(cv::Point2d centre, int radius)
    {
        cv::circle(image_, centre, radius, cv::Scalar(0xFF, 0xFF, 0x00), 1);
    }

    void Visualizer::drawEyeCentre(cv::Point2d eye_centre)
    {
        cv::circle(image_, eye_centre, 2, cv::Scalar(0x00, 0x80, 0x00), 5);
    }

    void Visualizer::drawCorneaCentre(cv::Point2d cornea_centre)
    {
        cv::circle(image_, cornea_centre, 2, cv::Scalar(0x00, 0xFF, 0xFF), 5);
    }

    void Visualizer::drawGlintEllipse(cv::RotatedRect ellipse)
    {
        cv::ellipse(image_, ellipse, cv::Scalar(0xFF, 0xFF, 0x00), 1);
    }

    void Visualizer::drawFps()
    {
        cv::putText(image_, fps_text_.str(), cv::Point2i(100, 100), cv::FONT_HERSHEY_SIMPLEX, 3,
                    cv::Scalar(0x00, 0x00, 0xFF), 3);
    }

} // namespace et