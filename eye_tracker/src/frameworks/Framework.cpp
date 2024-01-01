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
            EyeInfo eye_info = {.pupil = pupil, .pupil_radius = (double) pupil_radius, .glints = *glints, .ellipse = ellipse};
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

    std::string Framework::startEyeVideoRecording(const std::string &folder_name)
    {
        fs::path output_path = folder_name;
        std::string timestamp = Utils::getCurrentTimeText() + "_" + std::to_string(camera_id_);
        std::filesystem::create_directories(output_path / timestamp);

        if (!eye_video_.isOpened())
        {
            std::string filename = output_path / timestamp / "video.mp4";
            eye_video_.open(filename, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30,
                            et::Settings::parameters.camera_params[camera_id_].region_of_interest, false);
        }

        if (!eye_data_.is_open())
        {
            eye_data_.open(output_path / timestamp / "calib_data.csv");
        }
        eye_frame_counter_ = 0;
        return output_path / timestamp;
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

    void Framework::addEyeVideoData(const EyeDataToReceive &eye_data)
    {
        if (eye_data_.is_open())
        {
            eye_data_ << eye_frame_counter_;
            for (int i = 0; i < 4; i++)
            {
                eye_data_ << "," << eye_data.front_corners[i].x << "," << eye_data.front_corners[i].y << ","
                          << eye_data.front_corners[i].z;
            }
            for (int i = 0; i < 4; i++)
            {
                eye_data_ << "," << eye_data.back_corners[i].x << "," << eye_data.back_corners[i].y << ","
                          << eye_data.back_corners[i].z;
            }
            eye_data_ << "," << eye_data.marker_position.x << "," << eye_data.marker_position.y << ","
                      << eye_data.marker_position.z << "," << eye_data.timer << "\n";
        }
        if (eye_video_.isOpened())
        {
            eye_video_.write(analyzed_frame_.glints);
            eye_frame_counter_++;
        }
    }

    cv::Point3d Framework::setMetaModel(const std::string &input_path, const std::string &user_id)
    {
        fs::path path = input_path;
        std::clog << "Input path: " << input_path << std::endl;
        std::clog << "User id: " << user_id << std::endl;

        std::shared_ptr<ImageProvider> image_provider{};
        std::shared_ptr<FeatureAnalyser> feature_analyser{};
        if (fs::is_directory(path / "images"))
        {
            image_provider = std::make_shared<InputImages>(path);
            feature_analyser = std::make_shared<BlenderDiscreteFeatureAnalyser>(camera_id_);
        }
        else
        {
            image_provider = std::make_shared<InputVideo>(path / "video.mp4");
            feature_analyser = std::make_shared<CameraFeatureAnalyser>(camera_id_);
        }

        auto csv_path = path / "calib_data.csv";

        std::clog << "CSV path: " << csv_path.string() << std::endl;
        return meta_model_->findMetaModel(image_provider, feature_analyser, csv_path.string(), user_id);
    }

    void Framework::getEyeDataPackage(EyeDataToSend &eye_data_package)
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