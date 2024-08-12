#include "eye_tracker/frameworks/Framework.hpp"
#include "eye_tracker/Utils.hpp"
#include "eye_tracker/input/InputVideo.hpp"
#include "eye_tracker/image/FeatureAnalyser.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;

namespace et {
    std::mutex Framework::mutex{};

    Framework::Framework(int camera_id, bool headless) {
        camera_id_ = camera_id;
        if (headless) {
            visualization_type_ = VisualizationType::DISABLED;
        } else {
            visualization_type_ = VisualizationType::CAMERA_IMAGE;
        }
        visualizer_ = std::make_shared<Visualizer>(camera_id, headless);
        meta_model_ = std::make_shared<MetaModel>(camera_id);
    }

    bool Framework::analyzeNextFrame() {
        const auto now = std::chrono::system_clock::now();
        analyzed_frame_ = image_provider_->grabImage();
        if (analyzed_frame_.frame.empty()) {
            return false;
        }

        feature_detector_->preprocessImage(analyzed_frame_);
        features_found_ = feature_detector_->findPupil();
        features_found_ &= feature_detector_->findEllipsePoints();

        cv::Point2d pupil_centre_position_ics;
        feature_detector_->getPupilUndistorted(pupil_centre_position_ics);

        double pupil_radius_ics;
        feature_detector_->getPupilRadiusUndistorted(pupil_radius_ics);

        cv::Point3d cornea_centre_position_wcs{};
        if (features_found_) {
            std::vector<cv::Point2d> glint_positions_ics;
            feature_detector_->getGlints(glint_positions_ics);

            std::vector<bool> glints_valid;
            feature_detector_->getGlintsValidity(glints_valid);

            EyeInfo eye_info = {.pupil_centre_position_ics = pupil_centre_position_ics, .pupil_radius_ics = pupil_radius_ics, .glint_positions_ics = std::move(glint_positions_ics), .glints_valid = std::move(glints_valid)};
            eye_estimator_->estimateEyePositions(eye_info, !calibration_running_);
            eye_estimator_->getCorneaCentrePositionWCS(cornea_centre_position_wcs);
        }

        mutex.lock();
        if (output_video_.isOpened()) {
            output_video_.write(analyzed_frame_.frame);
            output_video_frame_counter_++;
        }
        mutex.unlock();

        if (calibration_running_) {
            CalibrationInput sample{};
            sample.detected = features_found_;
            sample.timestamp = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(now - calibration_start_time_).count()) / 1000.0;
            if (features_found_) {
                eye_estimator_->getEyeCentrePositionWCS(sample.eye_position);
                sample.cornea_position = cornea_centre_position_wcs;
                cv::Vec3d gaze_direction;
                eye_estimator_->getGazeDirection(gaze_direction);
                Utils::vectorToAngles(gaze_direction, sample.angles);
            }
            calibration_input_.push_back(sample);
        }

        return true;
    }

    void Framework::startRecording(const std::string& name, const bool record_ui) {
        mutex.lock();
        if (!output_video_.isOpened()) {
            if (!std::filesystem::is_directory("results")) {
                std::filesystem::create_directory("results");
            }
            std::string video, video_ui;
            if (name.empty()) {
                const auto current_time = Utils::getCurrentTimeText();
                video = "results/" + current_time;
                video_ui = "results/" + current_time;
                output_video_name_ = "results/" + current_time;
            } else {
                video = "results/" + name;
                video_ui = "results/" + name;
                output_video_name_ = "results/" + name;
            }

            video += "_" + std::to_string(camera_id_) + ".mp4";
            video_ui += "_" + std::to_string(camera_id_) + "_ui.mp4";
            output_video_name_ += "_" + std::to_string(camera_id_);

            std::clog << "Saving video to " << video << "\n";
            output_video_.open(video, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30, et::Settings::parameters.camera_params[camera_id_].region_of_interest, false);
            if (record_ui) {
                std::clog << "Saving UI video to " << video_ui << "\n";
                output_video_ui_.open(video_ui, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30, et::Settings::parameters.camera_params[camera_id_].region_of_interest, true);
            }
            output_video_frame_counter_ = 0;
        }
        mutex.unlock();
    }

    void Framework::stopRecording() {
        mutex.lock();
        if (output_video_.isOpened()) {
            std::clog << "Finished video recording.\n";
            output_video_.release();
        }
        if (output_video_ui_.isOpened()) {
            std::clog << "Finished video UI recording.\n";

            output_video_ui_.release();
        }
        mutex.unlock();
    }

    void Framework::captureCameraImage() const {
        if (!std::filesystem::is_directory("images")) {
            std::filesystem::create_directory("images");
        }
        const std::string filename{"images/" + Utils::getCurrentTimeText() + "_" + std::to_string(camera_id_) + ".png"};
        cv::imwrite(filename, analyzed_frame_.frame);
    }

    void Framework::updateUi() {
        static cv::Mat ui_image;
        visualizer_->calculateFramerate();
        switch (visualization_type_) {
            case VisualizationType::CAMERA_IMAGE:
                ui_image = analyzed_frame_.frame;
                break;
            case VisualizationType::THRESHOLD_PUPIL:
                feature_detector_->getThresholdedPupilImage(ui_image);
                break;
            case VisualizationType::THRESHOLD_GLINTS:
                feature_detector_->getThresholdedGlintsImage(ui_image);
                break;
            default:
                break;
        }

        if (visualization_type_ != VisualizationType::DISABLED) {
            visualizer_->prepareImage(ui_image);
            double pupil_diameter_mm{};
            cv::Point2d cornea_centre_position_ics{};
            cv::Point2d pupil_distorted{};
            std::vector<bool> glint_validity{};
            std::vector<cv::Point2d> glints_distorted;
            double pupil_radius_ics{};

            eye_estimator_->getPupilDiameter(pupil_diameter_mm);
            eye_estimator_->getCorneaCentrePositionICS(cornea_centre_position_ics, false);
            feature_detector_->getGlintsValidity(glint_validity);
            feature_detector_->getPupilDistorted(pupil_distorted);
            feature_detector_->getPupilRadiusDistorted(pupil_radius_ics);
            feature_detector_->getDistortedGlints(glints_distorted);

            visualizer_->drawCorneaCentre(feature_detector_->distort(cornea_centre_position_ics));
            visualizer_->drawCorneaTrace();
            visualizer_->drawGlints(glints_distorted, glint_validity);
            visualizer_->drawGazeTrace();
            visualizer_->drawGaze(eye_estimator_->getNormalizedGazePoint());
            visualizer_->drawMarker(getMarkerPosition());
            visualizer_->drawPupil(pupil_distorted, static_cast<int>(pupil_radius_ics), pupil_diameter_mm);

            visualizer_->drawFps();
            visualizer_->show();
        } else {
            visualizer_->printFramerateInterval();
        }

        mutex.lock();
        if (output_video_ui_.isOpened()) {
            output_video_ui_.write(visualizer_->getUiImage());
        }
        mutex.unlock();
    }

    void Framework::disableImageUpdate() {
        visualization_type_ = VisualizationType::DISABLED;
    }

    void Framework::switchToCameraImage() {
        visualization_type_ = VisualizationType::CAMERA_IMAGE;
    }

    void Framework::switchToPupilThreshImage() {
        visualization_type_ = VisualizationType::THRESHOLD_PUPIL;
    }

    void Framework::switchToGlintThreshImage() {
        visualization_type_ = VisualizationType::THRESHOLD_GLINTS;
    }

    bool Framework::shouldAppClose() const {
        if (!visualizer_->isWindowOpen()) {
            return true;
        }
        return false;
    }

    void Framework::startCalibration(std::string const& name) {
        std::cout << "Starting calibration" << std::endl;
        calibration_input_.clear();
        calibration_start_time_ = std::chrono::system_clock::now();
        calibration_running_ = true;
        startRecording(name, false);
    }

    void Framework::stopCalibration(const CalibrationOutput& calibration_output) {
        std::cout << "Stopping calibration" << std::endl;
        stopRecording();
        calibration_running_ = false;
        meta_model_->findMetaModelOnline(calibration_input_, calibration_output);
        eye_estimator_->updateFineTuning();
    }

    void Framework::stopEyeVideoRecording() {
        if (eye_video_.isOpened()) {
            std::clog << "Finished eye recording.\n";
            eye_video_.release();
        }

        if (eye_data_.is_open()) {
            std::clog << "Finished data recording.\n";
            eye_data_.close();
        }
    }

    void Framework::getEyeDataPackage(EyeDataToSend& eye_data_package) const {
        eye_estimator_->getCorneaCentrePositionWCS(eye_data_package.cornea_centre);
        eye_estimator_->getEyeCentrePositionWCS(eye_data_package.eye_centre);
        feature_detector_->getPupilUndistorted(eye_data_package.pupil);
        eye_estimator_->getGazeDirection(eye_data_package.gaze_direction);
        eye_estimator_->getPupilDiameter(eye_data_package.pupil_diameter);
        eye_data_package.frame_num = output_video_frame_counter_;
    }

    Framework::~Framework() {
        stopRecording();
        stopEyeVideoRecording();

        image_provider_->close();
    }

    cv::Point2d Framework::getMarkerPosition() {
        return {0, 0};
    }
} // namespace et
