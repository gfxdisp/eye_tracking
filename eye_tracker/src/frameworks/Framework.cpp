#include "eye_tracker/frameworks/Framework.hpp"
#include "eye_tracker/Utils.hpp"
#include "eye_tracker/input/InputVideo.hpp"
#include "eye_tracker/input/InputImages.hpp"
#include "eye_tracker/image/BlenderDiscreteFeatureAnalyser.hpp"
#include "eye_tracker/image/CameraFeatureAnalyser.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;

namespace et {
    std::mutex Framework::mutex{};

    Framework::Framework(int camera_id, bool headless) {
        camera_id_ = camera_id;
        // Initial shown image is raw camera feed, unless headless.
        if (headless) {
            visualization_type_ = VisualizationType::DISABLED;
        } else {
            visualization_type_ = VisualizationType::CAMERA_IMAGE;
        }
        initialized_ = true;
        visualizer_ = std::make_shared<Visualizer>(camera_id, headless);
        meta_model_ = std::make_shared<MetaModel>(camera_id);
    }

    bool Framework::analyzeNextFrame() {
        if (!initialized_) {
            return false;
        }

        analyzed_frame_ = image_provider_->grabImage();
        const auto now = std::chrono::system_clock::now();
        if (analyzed_frame_.pupil.empty() || analyzed_frame_.glints.empty()) {
            return false;
        }

        feature_detector_->preprocessImage(analyzed_frame_);
        features_found_ = feature_detector_->findPupil();
        cv::Point2d pupil = feature_detector_->getPupilUndistorted();
        features_found_ &= feature_detector_->findEllipsePoints();
        auto glints = feature_detector_->getGlints();
        auto glints_validity = feature_detector_->getGlintsValidity();
        cv::RotatedRect ellipse = feature_detector_->getEllipseUndistorted();
        int pupil_radius = feature_detector_->getPupilRadiusUndistorted();
        cv::Point3d cornea_centre{};
        if (features_found_) {
            EyeInfo eye_info = {
                    .pupil = pupil,
                    .pupil_radius = (double) pupil_radius,
                    .glints = *glints,
                    .glints_validity = *glints_validity,
                    .ellipse = ellipse
            };
            eye_estimator_->findEye(eye_info, !online_calibration_running_);
            eye_estimator_->getCorneaCurvaturePosition(cornea_centre);
            previous_cornea_centres_[cornea_history_index_] = eye_estimator_->getCorneaCurvaturePixelPosition(online_calibration_running_);
            cornea_history_index_++;
            if (cornea_history_index_ >= CORNEA_HISTORY_SIZE) {
                cornea_history_index_ = 0;
                cornea_history_full_ = true;
            }
            auto gaze_point = eye_estimator_->getNormalizedGazePoint();
            if (gaze_point.x >= 0 && gaze_point.x <= 1 && gaze_point.y >= 0 && gaze_point.y <= 1) {
                previous_gaze_points_[gaze_point_index_] = gaze_point;
                gaze_point_index_++;
                if (gaze_point_index_ >= GAZE_HISTORY_SIZE) {
                    gaze_point_index_ = 0;
                    gaze_point_history_full_ = true;
                }
            }
        }


        mutex.lock();
        if (output_video_.isOpened()) {
            output_video_.write(analyzed_frame_.glints);
            output_video_frame_counter_++;
        }
        mutex.unlock();

        feature_detector_->updateGazeBuffer();
        frame_counter_++;

        if (online_calibration_running_) {
            CalibrationInput sample{};
            sample.detected = features_found_;
            sample.timestamp = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(now - calibration_start_time_).count()) / 1000.0;
            if (features_found_) {
                eye_estimator_->getEyeCentrePosition(sample.eye_position);
                sample.cornea_position = cornea_centre;
                cv::Vec3d gaze_direction;
                eye_estimator_->getGazeDirection(gaze_direction);
                Utils::vectorToAngles(gaze_direction, sample.angles);
                sample.pcr_distance = pupil - (cv::Point2d) ellipse.center;
            }
            calibration_input_.push_back(sample);
        }

        return true;
    }

    void Framework::startRecording(const std::string& name) {
        mutex.lock();
        if (!output_video_.isOpened()) {
            if (!std::filesystem::is_directory("results")) {
                std::filesystem::create_directory("results");
            }
            std::string video, video_ui;
            if (name.empty()) {
                auto current_time = Utils::getCurrentTimeText();
                video = "results/" + current_time + "_" + std::to_string(camera_id_) + ".mp4";
                video_ui = "results/" + current_time + "_" + std::to_string(camera_id_) + "_ui.mp4";
                output_video_name_ = "results/" + current_time + "_" + std::to_string(camera_id_);
            } else {
                video = "results/" + name + "_" + std::to_string(camera_id_) + ".mp4";
                video_ui = "results/" + name + "_" + std::to_string(camera_id_) + "_ui.mp4";
                output_video_name_ = "results/" + name + "_" + std::to_string(camera_id_);
            }
            std::clog << "Saving video to " << video << "\n";
            output_video_.open(video, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30,
                               et::Settings::parameters.camera_params[camera_id_].region_of_interest, false);
            output_video_ui_.open(video_ui, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30,
                                  et::Settings::parameters.camera_params[camera_id_].region_of_interest, true);
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

    void Framework::captureCameraImage() {
        if (!std::filesystem::is_directory("images")) {
            std::filesystem::create_directory("images");
        }
        std::string filename{"images/" + Utils::getCurrentTimeText() + "_" + std::to_string(camera_id_) + ".png"};
        imwrite(filename, analyzed_frame_.glints);
    }

    void Framework::updateUi() {
        visualizer_->calculateFramerate();
        switch (visualization_type_) {
            case VisualizationType::CAMERA_IMAGE:
                visualizer_->prepareImage(analyzed_frame_.glints);
                break;
            case VisualizationType::THRESHOLD_PUPIL:
                visualizer_->prepareImage(feature_detector_->getThresholdedPupilImage());
                break;
            case VisualizationType::THRESHOLD_GLINTS:
                visualizer_->prepareImage(feature_detector_->getThresholdedGlintsImage());
                break;
            case VisualizationType::DISABLED:
                visualizer_->printFramerateInterval();
            default:
                break;
        }
        if (visualization_type_ != VisualizationType::DISABLED) {
            double pupil_diameter{};
            eye_estimator_->getPupilDiameter(pupil_diameter);
//            visualizer_->drawBoundingCircle(Settings::parameters.detection_params[camera_id_].pupil_search_centre,
//                                            Settings::parameters.detection_params[camera_id_].pupil_search_radius);
//           visualizer_->drawEyeCentre(feature_detector_->distort(eye_estimator_->getEyeCentrePixelPosition(online_calibration_running_)));
            visualizer_->drawCorneaCentre(
                    feature_detector_->distort(eye_estimator_->getCorneaCurvaturePixelPosition(online_calibration_running_)));
            visualizer_->drawCorneaTrace(previous_cornea_centres_, cornea_history_full_ ? (cornea_history_index_ + 1) % CORNEA_HISTORY_SIZE : 0, cornea_history_index_, CORNEA_HISTORY_SIZE);

//            visualizer_->drawGlintEllipse(feature_detector_->getEllipseDistorted());
            visualizer_->drawGlints(feature_detector_->getDistortedGlints(), feature_detector_->getGlintsValidity());

            visualizer_->drawGazeTrace(previous_gaze_points_, gaze_point_history_full_ ? (gaze_point_index_ + 1) % GAZE_HISTORY_SIZE : 0, gaze_point_index_, GAZE_HISTORY_SIZE);
            visualizer_->drawGaze(eye_estimator_->getNormalizedGazePoint());
            visualizer_->drawMarker(getMarkerPosition());


            visualizer_->drawPupil(feature_detector_->getPupilDistorted(),
                                   feature_detector_->getPupilRadiusDistorted(), pupil_diameter);

//            visualizer_->drawFps();
            visualizer_->show();
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

    bool Framework::shouldAppClose() {
        if (!visualizer_->isWindowOpen()) {
            return true;
        }
        return false;
    }

    double Framework::getAvgFramerate() {
        return visualizer_->getAvgFramerate();
    }

    void Framework::startOnlineCalibration() {
        std::cout << "Starting online calibration" << std::endl;
        calibration_input_.clear();
        calibration_start_time_ = std::chrono::system_clock::now();
        online_calibration_running_ = true;
        startRecording();
    }

    void Framework::stopOnlineCalibration(const CalibrationOutput& calibration_output, bool calibrate_from_scratch) {
        std::cout << "Stopping online calibration" << std::endl;
        stopRecording();
        online_calibration_running_ = false;
        meta_model_->findMetaModel(calibration_input_, calibration_output, calibrate_from_scratch, output_video_name_);
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

    void Framework::addEyeVideoData(const cv::Point3d& marker_position) {
        if (calibration_running_) {
            auto current_time = std::chrono::system_clock::now();
            auto glints = feature_detector_->getGlints();
            auto glints_validity = feature_detector_->getGlintsValidity();

            CalibrationSample sample;
            sample.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - calibration_start_time_).count() / 1000.0;
            sample.marker_position = marker_position;
            std::copy(glints->begin(), glints->end(), std::back_inserter(sample.glints));
            std::copy(glints_validity->begin(), glints_validity->end(), std::back_inserter(sample.glints_validity));
            sample.detected = features_found_;
            sample.glint_ellipse = feature_detector_->getEllipseUndistorted();
            sample.pupil_position = feature_detector_->getPupilUndistorted();
            sample.eye_position = calibration_eye_position_;
            if (!calibration_data_.empty()) {
                auto last_sample = calibration_data_.back();
                if (last_sample.marker_position != sample.marker_position) {
                    sample.marker_id = last_sample.marker_id + 1;
                    sample.marker_time = 0;
                } else {
                    sample.marker_id = last_sample.marker_id;
                    sample.marker_time = last_sample.marker_time + last_sample.timestamp - sample.timestamp;
                }
            } else {
                sample.marker_id = 0;
                sample.marker_time = 0;
            }
            calibration_data_.push_back(sample);
            calib_video_.write(analyzed_frame_.glints);
        }
    }

    void Framework::dumpCalibrationData(const std::string& video_path) {
        if (calibration_data_.empty()) {
            return;
        }

        std::string timestamp = Utils::getCurrentTimeText();

        std::string csv_path = fs::path(video_path) / (timestamp + std::to_string(camera_id_) + ".csv");
        std::string mp4_path = fs::path(video_path) / (timestamp + std::to_string(camera_id_) + ".mp4");

        std::ofstream file;
        file.open(csv_path);

        file << "eye_x, eye_y, eye_z, marker_id, marker_x, marker_y, marker_z, pupil_x, pupil_y, glint_x, glint_y, glint_width, glint_height, detected, timestamp, marker_time\n";

        for (auto sample: calibration_data_) {
            file << sample.eye_position.x << ", " << sample.eye_position.y << ", " << sample.eye_position.z << ", " <<
                 sample.marker_id << ", " << sample.marker_position.x << ", " << sample.marker_position.y << ", " << sample.marker_position.z << ", " <<
                 sample.pupil_position.x << ", " << sample.pupil_position.y << ", " <<
                 sample.glint_ellipse.center.x << ", " << sample.glint_ellipse.center.y << ", " << sample.glint_ellipse.size.width << ", " << sample.glint_ellipse.size.height << ", " <<
                 sample.detected << ", " << sample.timestamp << ", " << sample.marker_time << "\n";
        }
        file.close();
        if (fs::exists("calib_temp.mp4")) {
            fs::copy_file("calib_temp.mp4", mp4_path, fs::copy_options::overwrite_existing);
        }
    }

    void Framework::getEyeDataPackage(EyeDataToSend& eye_data_package) {
        eye_estimator_->getCorneaCurvaturePosition(eye_data_package.cornea_centre);
        eye_estimator_->getEyeCentrePosition(eye_data_package.eye_centre);
        eye_estimator_->getGazeDirection(eye_data_package.gaze_direction);
        eye_estimator_->getPupilDiameter(eye_data_package.pupil_diameter);
        feature_detector_->getPupilUndistorted(eye_data_package.pupil);
        feature_detector_->getPupilGlintVector(eye_data_package.pupil_glint_vector);
        feature_detector_->getEllipseUndistorted(eye_data_package.ellipse);
        eye_data_package.frame_num = output_video_frame_counter_;
    }

    Framework::~Framework() {
        stopRecording();
        stopEyeVideoRecording();

        image_provider_->close();
    }

    bool Framework::wereFeaturesFound() {
        return features_found_;
    }

    cv::Point2d Framework::getMarkerPosition() {
        return {0, 0};
    }
} // namespace et
