#include "EyeTracker.hpp"
#include "Utils.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;

namespace et {
void EyeTracker::initialize(ImageProvider *image_provider,
                            const std::string &settings_path,
                            const bool enabled_cameras[],
                            const bool ellipse_fitting[],
                            const bool enabled_kalman[],
                            const bool enabled_template_matching[]) {
    image_provider_ = image_provider;
    image_provider_->initialize();
    settings_path_ = settings_path;

    for (int i = 0; i < 2; i++) {
        if (enabled_cameras[i]) {
            camera_ids_.push_back(i);
        }
        feature_detectors_[i].initialize(settings_path_, enabled_kalman[i],
                                         enabled_template_matching[i], i);
        eye_estimators_[i].initialize(settings_path_, enabled_kalman[i], i);
        ellipse_fitting_[i] = ellipse_fitting[i];
    }
    for (auto &i : camera_ids_) {
        visualizer_[i].initialize(i);
    }
    // Initial shown image is raw camera feed.
    visualization_type_ = VisualizationType::CAMERA_IMAGE;
    initialized_ = true;
}

bool EyeTracker::analyzeNextFrame() {
    if (!initialized_) {
        return false;
    }
    for (auto &i : camera_ids_) {
        analyzed_frame_[i] = image_provider_->grabImage(i);
        if (analyzed_frame_[i].empty()) {
            return false;
        }

        feature_detectors_[i].preprocessImage(analyzed_frame_[i]);
        bool features_found = feature_detectors_[i].findPupil();
        cv::Point2f pupil = feature_detectors_[i].getPupil();
        features_found &= feature_detectors_[i].findEllipse();
        int pupil_radius = feature_detectors_[i].getPupilRadius();
        if (ellipse_fitting_[i]) {
            // Polynomial-based approach: ellipse fitting does not look for
            // individual glints but for the pattern they form.
            cv::RotatedRect ellipse = feature_detectors_[i].getEllipse();
            if (features_found) {
                eye_estimators_[i].getEyeFromPolynomial(pupil, ellipse);
                cv::Vec3d cornea_centre{};
                eye_estimators_[i].getCorneaCurvaturePosition(cornea_centre);
                eye_estimators_[i].calculatePupilDiameter(pupil, pupil_radius,
                                                          cornea_centre);
            }

        } else {
            // Model-based approach: position of every glint needs to found.
            auto glints = feature_detectors_[i].getGlints();
            if (features_found) {
                eye_estimators_[i].getEyeFromModel(pupil, glints);
                cv::Vec3d cornea_centre{};
                eye_estimators_[i].getCorneaCurvaturePosition(cornea_centre);
                eye_estimators_[i].calculatePupilDiameter(pupil, pupil_radius,
                                                          cornea_centre);
            }
        }
        if (output_video_[i].isOpened()) {
            output_video_[i].write(analyzed_frame_[i]);
        }
        feature_detectors_[i].updateGazeBuffer();
    }
    frame_counter_++;
    return true;
}

void EyeTracker::getPupil(cv::Point2f &pupil, int camera_id) {
    feature_detectors_[camera_id].getPupil(pupil);
}

void EyeTracker::getPupilFiltered(cv::Point2f &pupil, int camera_id) {
    feature_detectors_[camera_id].getPupilFiltered(pupil);
}

void EyeTracker::getPupilDiameter(float &pupil_diameter, int camera_id) {
    eye_estimators_[camera_id].getPupilDiameter(pupil_diameter);
}

void EyeTracker::getGazeDirection(cv::Vec3f &gaze_direction, int camera_id) {
    eye_estimators_[camera_id].getGazeDirection(gaze_direction);
}

void EyeTracker::setGazeBufferSize(uint8_t value) {
    for (auto &feature_detector : feature_detectors_) {
        feature_detector.setGazeBufferSize(value);
    }
}

void EyeTracker::getPupilGlintVector(cv::Vec2f &pupil_glint_vector,
                                     int camera_id) {
    feature_detectors_[camera_id].getPupilGlintVector(pupil_glint_vector);
}

void EyeTracker::getPupilGlintVectorFiltered(cv::Vec2f &pupil_glint_vector,
                                             int camera_id) {
    feature_detectors_[camera_id].getPupilGlintVectorFiltered(
        pupil_glint_vector);
}

void EyeTracker::getEyeCentrePosition(cv::Vec3d &eye_centre, int camera_id) {
    eye_estimators_[camera_id].getEyeCentrePosition(eye_centre);
}

void EyeTracker::getCorneaCentrePosition(cv::Vec3d &cornea_centre,
                                         int camera_id) {
    eye_estimators_[camera_id].getCorneaCurvaturePosition(cornea_centre);
}

void EyeTracker::logDetectedFeatures(std::ostream &output, int camera_id) {
    cv::Point2f pupil = feature_detectors_[camera_id].getPupil();
    output << frame_counter_ - 1 << "," << pupil.x << "," << pupil.y;

    if (ellipse_fitting_[camera_id]) {
        cv::RotatedRect ellipse = feature_detectors_[camera_id].getEllipse();
        output << "," << ellipse.center.x << "," << ellipse.center.y << ",";
        output << ellipse.size.width << "," << ellipse.size.height << ",";
        output << ellipse.angle;
    } else {
        auto glints = feature_detectors_[camera_id].getGlints();
        output << "," << glints->size() << ",";
        for (auto &glint : *glints) {
            output << glint.x << ";" << glint.y << ";";
        }
    }
    output << "\n";
}

void EyeTracker::logEyePosition(std::ostream &output, int camera_id) {

    cv::Vec3d cornea_centre{}, eye_centre{};
    eye_estimators_[camera_id].getCorneaCurvaturePosition(cornea_centre);
    eye_estimators_[camera_id].getEyeCentrePosition(eye_centre);
    output << frame_counter_ - 1 << "," << cornea_centre[0] << ","
           << cornea_centre[1] << "," << cornea_centre[2] << ","
           << eye_centre[0] << "," << eye_centre[1] << "," << eye_centre[2]
           << "\n";
}

void EyeTracker::switchVideoRecordingState() {
    if (!camera_ids_.empty() && output_video_[camera_ids_[0]].isOpened()) {
        stopVideoRecording();
    } else {
        if (!std::filesystem::is_directory("videos")) {
            std::filesystem::create_directory("videos");
        }
        for (auto &i : camera_ids_) {
            auto current_time = Utils::getCurrentTimeText();
            std::string filename =
                "videos/" + current_time + "_" + std::to_string(i) + ".mp4";
            std::clog << "Saving video to " << filename << "\n";
            output_video_[i].open(
                filename, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30,
                et::Settings::parameters.camera_params[i].region_of_interest,
                false);
            filename =
                "videos/" + current_time + "_" + std::to_string(i) + "_ui.mp4";
            output_video_ui_[i].open(
                filename, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30,
                et::Settings::parameters.camera_params[i].region_of_interest,
                true);
        }
    }
}

void EyeTracker::stopVideoRecording() {
    std::clog << "Finished video recording.\n";
    for (auto &i : camera_ids_) {
        if (output_video_[i].isOpened()) {
            output_video_[i].release();
        }
        if (output_video_ui_[i].isOpened()) {
            output_video_ui_[i].release();
        }
    }
}

void EyeTracker::captureCameraImage() {
    if (!std::filesystem::is_directory("images")) {
        std::filesystem::create_directory("images");
    }
    for (auto &i : camera_ids_) {
        std::string filename{"images/" + Utils::getCurrentTimeText() + "_"
                             + std::to_string(i) + ".png"};
        imwrite(filename, analyzed_frame_[i]);
    }
}

void EyeTracker::updateUi() {
    Visualizer::calculateFramerate();
    if (visualization_type_ == VisualizationType::DISABLED) {
        et::Visualizer::printFramerateInterval();
    }
    for (auto &i : camera_ids_) {
        switch (visualization_type_) {
        case VisualizationType::CAMERA_IMAGE:
            visualizer_[i].prepareImage(analyzed_frame_[i]);
            break;
        case VisualizationType::THRESHOLD_PUPIL:
            visualizer_[i].prepareImage(
                feature_detectors_[i].getThresholdedPupilImage());
            break;
        case VisualizationType::THRESHOLD_GLINTS:
            visualizer_[i].prepareImage(
                feature_detectors_[i].getThresholdedGlintsImage());
            break;
        default:
            break;
        }
        if (visualization_type_ != VisualizationType::DISABLED) {
            visualizer_[i].drawBoundingCircle(
                Settings::parameters.detection_params[i].pupil_search_centre,
                Settings::parameters.detection_params[i].pupil_search_radius);
            visualizer_[i].drawPupil(
                feature_detectors_[i].getDistortedPupil(),
                feature_detectors_[i].getDistortedPupilRadius());
            visualizer_[i].drawEyeCentre(
                eye_estimators_[i].getEyeCentrePixelPosition());
            visualizer_[i].drawCorneaCentre(
                eye_estimators_[i].getCorneaCurvaturePixelPosition());

            if (ellipse_fitting_[i]) {
                visualizer_[i].drawGlintEllipse(
                    feature_detectors_[i].getDistortedEllipse());
            }
            visualizer_[i].drawGlints(
                feature_detectors_[i].getDistortedGlints());

            visualizer_[i].drawFps();
            visualizer_[i].show();
        }

        if (output_video_ui_[i].isOpened()) {
            output_video_ui_[i].write(visualizer_[i].getUiImage());
        }
    }
}

void EyeTracker::disableImageUpdate() {
    visualization_type_ = VisualizationType::DISABLED;
}

void EyeTracker::switchToCameraImage() {
    visualization_type_ = VisualizationType::CAMERA_IMAGE;
}

void EyeTracker::switchToPupilThreshImage() {
    visualization_type_ = VisualizationType::THRESHOLD_PUPIL;
}

void EyeTracker::switchToGlintThreshImage() {
    visualization_type_ = VisualizationType::THRESHOLD_GLINTS;
}

bool EyeTracker::shouldAppClose() {
    for (auto &i : camera_ids_) {
        if (!visualizer_[i].isWindowOpen()) {
            return true;
        }
    }
    return false;
}

float EyeTracker::getAvgFramerate() {
    return Visualizer::getAvgFramerate();
}

void EyeTracker::saveEyeData(cv::Point3f left_eye_pos,
                            cv::Point3f right_eye_pos) {
    fs::path eye_data_path = fs::path(settings_path_) / "eye_data";
    std::string timestamp = Utils::getCurrentTimeText();
    if (!std::filesystem::is_directory(eye_data_path)) {
        std::filesystem::create_directory(eye_data_path);
    }
    for (auto &i : camera_ids_) {
        std::string filename{eye_data_path
                             / (timestamp + "_" + std::to_string(i) + ".png")};
        imwrite(filename, analyzed_frame_[i]);
    }

    std::ofstream eye_pos_file{eye_data_path / "eye_pos.csv", std::ios::app};
    eye_pos_file << timestamp << "," << left_eye_pos.x << "," << left_eye_pos.y
                 << "," << left_eye_pos.z << "," << right_eye_pos.x << ","
                 << right_eye_pos.y << "," << right_eye_pos.z << "\n";
    eye_pos_file.close();
}

} // namespace et