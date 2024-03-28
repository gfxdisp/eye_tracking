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

namespace et
{
    std::mutex Framework::mutex{};

    Framework::Framework(int camera_id, bool headless)
    {
        camera_id_ = camera_id;
        // Initial shown image is raw camera feed, unless headless.
        if (headless)
        {
            visualization_type_ = VisualizationType::DISABLED;
        }
        else
        {
            visualization_type_ = VisualizationType::CAMERA_IMAGE;
        }
        initialized_ = true;
        visualizer_ = std::make_shared<Visualizer>(camera_id, headless);
        meta_model_ = std::make_shared<MetaModel>(camera_id);
    }

    bool Framework::analyzeNextFrame()
    {
        if (!initialized_)
        {
            return false;
        }

        mutex.lock();

        analyzed_frame_ = image_provider_->grabImage();
        if (analyzed_frame_.pupil.empty() || analyzed_frame_.glints.empty())
        {
            return false;
        }

        feature_detector_->preprocessImage(analyzed_frame_);
        features_found_ = feature_detector_->findPupil();
        cv::Point2d pupil = feature_detector_->getPupilUndistorted();
        features_found_ &= feature_detector_->findEllipsePoints();
        auto glints = feature_detector_->getGlints();
        cv::RotatedRect ellipse = feature_detector_->getEllipseUndistorted();
        int pupil_radius = feature_detector_->getPupilRadiusUndistorted();
        if (features_found_)
        {
            EyeInfo eye_info = {
                    .pupil = pupil, .pupil_radius = (double) pupil_radius, .glints = *glints, .ellipse = ellipse
            };
            eye_estimator_->findEye(eye_info);
            cv::Point3d cornea_centre{};
            eye_estimator_->getCorneaCurvaturePosition(cornea_centre);
        }

        if (output_video_.isOpened())
        {
            output_video_.write(analyzed_frame_.glints);
        }

        feature_detector_->updateGazeBuffer();
        frame_counter_++;

        mutex.unlock();

        return true;
    }

    void Framework::switchVideoRecordingState()
    {
        if (output_video_.isOpened())
        {
            stopVideoRecording();
        }
        else
        {
            if (!std::filesystem::is_directory("videos"))
            {
                std::filesystem::create_directory("videos");
            }
            auto current_time = Utils::getCurrentTimeText();
            std::string filename = "videos/" + current_time + "_" + std::to_string(camera_id_) + ".mp4";
            std::clog << "Saving video to " << filename << "\n";
            output_video_.open(filename, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30,
                               et::Settings::parameters.camera_params[camera_id_].region_of_interest, false);
            filename = "videos/" + current_time + "_" + std::to_string(camera_id_) + "_ui.mp4";
            output_video_ui_.open(filename, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30,
                                  et::Settings::parameters.camera_params[camera_id_].region_of_interest, true);
        }
    }

    void Framework::stopVideoRecording()
    {
        if (output_video_.isOpened())
        {
            std::clog << "Finished video recording.\n";
            output_video_.release();
        }
        if (output_video_ui_.isOpened())
        {
            std::clog << "Finished video UI recording.\n";
            output_video_ui_.release();
        }
    }

    void Framework::captureCameraImage()
    {
        if (!std::filesystem::is_directory("images"))
        {
            std::filesystem::create_directory("images");
        }
        std::string filename{"images/" + Utils::getCurrentTimeText() + "_" + std::to_string(camera_id_) + ".png"};
        imwrite(filename, analyzed_frame_.glints);
    }

    void Framework::updateUi()
    {
        visualizer_->calculateFramerate();
        switch (visualization_type_)
        {
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
        if (visualization_type_ != VisualizationType::DISABLED)
        {
            visualizer_->drawBoundingCircle(Settings::parameters.detection_params[camera_id_].pupil_search_centre,
                                            Settings::parameters.detection_params[camera_id_].pupil_search_radius);
            visualizer_->drawPupil(feature_detector_->getPupilDistorted(),
                                   feature_detector_->getPupilRadiusDistorted());
            visualizer_->drawEyeCentre(feature_detector_->distort(eye_estimator_->getEyeCentrePixelPosition()));
            visualizer_->drawCorneaCentre(
                    feature_detector_->distort(eye_estimator_->getCorneaCurvaturePixelPosition()));

            visualizer_->drawGlintEllipse(feature_detector_->getEllipseDistorted());
            visualizer_->drawGlints(feature_detector_->getDistortedGlints());

            visualizer_->drawFps();
            visualizer_->show();
        }

        if (output_video_ui_.isOpened())
        {
            output_video_ui_.write(visualizer_->getUiImage());
        }
    }

    void Framework::disableImageUpdate()
    {
        visualization_type_ = VisualizationType::DISABLED;
    }

    void Framework::switchToCameraImage()
    {
        visualization_type_ = VisualizationType::CAMERA_IMAGE;
    }

    void Framework::switchToPupilThreshImage()
    {
        visualization_type_ = VisualizationType::THRESHOLD_PUPIL;
    }

    void Framework::switchToGlintThreshImage()
    {
        visualization_type_ = VisualizationType::THRESHOLD_GLINTS;
    }

    bool Framework::shouldAppClose()
    {
        if (!visualizer_->isWindowOpen())
        {
            return true;
        }
        return false;
    }

    double Framework::getAvgFramerate()
    {
        return visualizer_->getAvgFramerate();
    }

    void Framework::startCalibration(const cv::Point3d& eye_position)
    {
        calibration_data_.clear();
        calibration_eye_position_ = eye_position;
        calibration_running_ = true;
        const auto now = std::chrono::system_clock::now();
        calibration_start_time_ = std::chrono::system_clock::to_time_t(now);

        std::string filename = "calib_temp.mp4";
        calib_video_.open(filename, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30,
                          et::Settings::parameters.camera_params[camera_id_].region_of_interest, false);
    }

    void Framework::loadOldCalibrationData(const std::string& path)
    {
        std::string csv_path = path + ".csv";
        std::string mp4_path = path + "_" + std::to_string(camera_id_) + ".mp4";
        auto csv_file = Utils::readFloatRowsCsv(csv_path);
        calibration_eye_position_ = cv::Point3d(csv_file[0][1 + 3 * camera_id_], csv_file[0][2 + 3 * camera_id_],
                                                csv_file[0][3 + 3 * camera_id_]);

        auto image_provider = std::make_shared<InputVideo>(mp4_path);
        auto feature_analyser = std::make_shared<CameraFeatureAnalyser>(camera_id_);
        int counter = 0;
        double time_per_marker = 3;
        int samples_per_marker = 0;
        int markers_count = 0;
        cv::Point2d previous_pupil;
        while (true)
        {
            auto analyzed_frame_ = image_provider->grabImage();
            if (analyzed_frame_.pupil.empty() || analyzed_frame_.glints.empty())
            {
                break;
            }
            feature_analyser->preprocessImage(analyzed_frame_);
            bool features_found = feature_analyser->findPupil() && feature_analyser->findEllipsePoints();

            cv::Point2d pupil = feature_analyser->getPupilUndistorted();
            auto glints = feature_analyser->getGlints();
            cv::RotatedRect ellipse = feature_analyser->getEllipseUndistorted();

            previous_pupil = pupil;

            CalibrationSample sample;
            sample.detected = features_found;
            sample.eye_position = calibration_eye_position_;
            sample.pupil_position = pupil;
            sample.glint_ellipse = ellipse;
            sample.top_left_glint = glints->at(0);
            sample.bottom_right_glint = glints->at(1);
            sample.marker_position = cv::Point3d(csv_file[counter][7], csv_file[counter][8], csv_file[counter][9]);
            sample.marker_id = markers_count;
            calibration_data_.push_back(sample);

            samples_per_marker++;
            counter++;
            if (counter >= csv_file.size() || counter > 0 && (csv_file[counter][7] != csv_file[counter - 1][7] || csv_file[counter][8] != csv_file[counter - 1][8] || csv_file[counter][9] != csv_file[counter - 1][9]))
            {
                for (int i = 0; i < samples_per_marker; i++)
                {
                    calibration_data_[counter - samples_per_marker + i].timestamp = i * time_per_marker / samples_per_marker + markers_count * time_per_marker;
                    calibration_data_[counter - samples_per_marker + i].marker_time = i * time_per_marker / samples_per_marker;
                }
                samples_per_marker = 0;
                markers_count++;
            }
        }
        meta_model_->findMetaModel(calibration_data_);
    }

    void Framework::stopCalibration()
    {
        calibration_running_ = false;
        calib_video_.release();
        meta_model_->findMetaModel(calibration_data_);
    }

    void Framework::stopEyeVideoRecording()
    {
        if (eye_video_.isOpened())
        {
            std::clog << "Finished eye recording.\n";
            eye_video_.release();
        }

        if (eye_data_.is_open())
        {
            std::clog << "Finished data recording.\n";
            eye_data_.close();
        }
    }

    void Framework::addEyeVideoData(const cv::Point3d& marker_position)
    {
        if (calibration_running_)
        {
            const auto now = std::chrono::system_clock::now();
            auto current_time = std::chrono::system_clock::to_time_t(now);
            auto glints = feature_detector_->getGlints();

            CalibrationSample sample;
            sample.timestamp = std::difftime(current_time, calibration_start_time_);
            sample.marker_position = marker_position;
            sample.top_left_glint = glints->at(0);
            sample.bottom_right_glint = glints->at(1);
            sample.detected = features_found_;
            sample.glint_ellipse = feature_detector_->getEllipseUndistorted();
            sample.pupil_position = feature_detector_->getPupilUndistorted();
            sample.eye_position = calibration_eye_position_;
            if (!calibration_data_.empty())
            {
                auto last_sample = calibration_data_.back();
                if (last_sample.marker_position != sample.marker_position)
                {
                    sample.marker_id = last_sample.marker_id + 1;
                    sample.marker_time = 0;
                }
                else
                {
                    sample.marker_id = last_sample.marker_id;
                    sample.marker_time = last_sample.marker_time + last_sample.timestamp - sample.timestamp;
                }
            }
            else
            {
                sample.marker_id = 0;
                sample.marker_time = 0;
            }
            calibration_data_.push_back(sample);
            calib_video_.write(analyzed_frame_.glints);
        }
    }

    void Framework::dumpCalibrationData(const std::string& video_path)
    {
        if (calibration_data_.empty())
        {
            return;
        }

        std::string timestamp = Utils::getCurrentTimeText();

        std::string csv_path = fs::path(video_path) / (timestamp + std::to_string(camera_id_) + ".csv");
        std::string mp4_path = fs::path(video_path) / (timestamp + std::to_string(camera_id_) + ".mp4");

        std::ofstream file;
        file.open(csv_path);

        file << "eye_x, eye_y, eye_z, marker_id, marker_x, marker_y, marker_z, pupil_x, pupil_y, glint_x, glint_y, glint_width, glint_height, detected, timestamp, marker_time\n";

        for (auto sample: calibration_data_)
        {
            file << sample.eye_position.x << ", " << sample.eye_position.y << ", " << sample.eye_position.z << ", " <<
                 sample.marker_id << ", " << sample.marker_position.x << ", " << sample.marker_position.y << ", " << sample.marker_position.z << ", " <<
                 sample.pupil_position.x << ", " << sample.pupil_position.y << ", " <<
                 sample.glint_ellipse.center.x << ", " << sample.glint_ellipse.center.y << ", " << sample.glint_ellipse.size.width << ", " << sample.glint_ellipse.size.height << ", " <<
                 sample.detected << ", " << sample.timestamp << ", " << sample.marker_time << "\n";
        }
        file.close();
        if (fs::exists("calib_temp.mp4"))
        {
            fs::copy_file("calib_temp.mp4", mp4_path, fs::copy_options::overwrite_existing);
        }
    }

    void Framework::getEyeDataPackage(EyeDataToSend& eye_data_package)
    {
        eye_estimator_->getCorneaCurvaturePosition(eye_data_package.cornea_centre);
        eye_estimator_->getEyeCentrePosition(eye_data_package.eye_centre);
        eye_estimator_->getGazeDirection(eye_data_package.gaze_direction);
        eye_estimator_->getPupilDiameter(eye_data_package.pupil_diameter);
        feature_detector_->getPupilUndistorted(eye_data_package.pupil);
        feature_detector_->getPupilGlintVector(eye_data_package.pupil_glint_vector);
        feature_detector_->getEllipseUndistorted(eye_data_package.ellipse);
        feature_detector_->getFrameNum(eye_data_package.frame_num);
    }

    Framework::~Framework()
    {
        stopVideoRecording();
        stopEyeVideoRecording();

        image_provider_->close();
    }

    bool Framework::wereFeaturesFound()
    {
        return features_found_;
    }
} // namespace et
