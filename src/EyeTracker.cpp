#include "EyeTracker.hpp"
#include "Utils.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;

namespace et {
void EyeTracker::initialize(ImageProvider *image_provider, const std::string &settings_path,
                            const std::string &input_path, const bool enabled_cameras[], const bool ellipse_fitting[],
                            const bool enabled_kalman[], const bool enabled_template_matching[],
                            const bool distorted[], bool headless) {
    image_provider_ = image_provider;
    image_provider_->initialize();
    settings_path_ = settings_path;

    for (int i = 0; i < 2; i++) {
        if (enabled_cameras[i]) {
            camera_ids_.push_back(i);
        }
        feature_detectors_[i].initialize(settings_path_, enabled_kalman[i], enabled_template_matching[i], distorted[i],
                                         i);
        eye_estimators_[i].initialize(input_path, enabled_kalman[i], i);
        ellipse_fitting_[i] = ellipse_fitting[i];
    }
    if (!headless) {
        for (auto &i : camera_ids_) {
            visualizer_[i].initialize(i);
        }
    }

    M_left_ = cv::Mat::eye(4, 4, CV_64F);
    M_right_ = cv::Mat::eye(4, 4, CV_64F);

    // Initial shown image is raw camera feed, unless headless.
    if (headless) {
        visualization_type_ = VisualizationType::DISABLED;
    } else {
        visualization_type_ = VisualizationType::CAMERA_IMAGE;
    }
    initialized_ = true;
}

bool EyeTracker::analyzeNextFrame() {
    if (!initialized_) {
        return false;
    }

    per_user_transformation_blocker_.lock();
    for (auto &i : camera_ids_) {
        analyzed_frame_[i] = image_provider_->grabImage(i);
        if (analyzed_frame_[i].pupil.empty() || analyzed_frame_[i].glints.empty()) {
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
                eye_estimators_[i].calculatePupilDiameter(pupil, pupil_radius, cornea_centre);
            }

        } else {
            // Model-based approach: position of every glint needs to found.
            auto glints = feature_detectors_[i].getGlints();
            if (features_found) {
                eye_estimators_[i].getEyeFromModel(pupil, glints);
                cv::Vec3d cornea_centre{};
                eye_estimators_[i].getCorneaCurvaturePosition(cornea_centre);
                eye_estimators_[i].calculatePupilDiameter(pupil, pupil_radius, cornea_centre);
            }
        }
        if (output_video_[i].isOpened()) {
            output_video_[i].write(analyzed_frame_[i].glints);
        }

        feature_detectors_[i].updateGazeBuffer();
    }
    per_user_transformation_blocker_.unlock();
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

void EyeTracker::getPupilGlintVector(cv::Vec2f &pupil_glint_vector, int camera_id) {
    feature_detectors_[camera_id].getPupilGlintVector(pupil_glint_vector);
}

void EyeTracker::getPupilGlintVectorFiltered(cv::Vec2f &pupil_glint_vector, int camera_id) {
    feature_detectors_[camera_id].getPupilGlintVectorFiltered(pupil_glint_vector);
}

void EyeTracker::getEyeCentrePosition(cv::Vec3d &eye_centre, int camera_id) {
    cv::Vec3d base_eye_centre;
    eye_estimators_[camera_id].getEyeCentrePosition(base_eye_centre);
    //Convert Vec3d to homogeneous coordinates
    cv::Mat eye_centre_homogeneous =
        (cv::Mat_<float>(1, 4) << base_eye_centre[0], base_eye_centre[1], base_eye_centre[2], 1);

    //Multiply by M_left_ or M_right_ depending on camera_id
    cv::Mat eye_centre_homogeneous_transformed;
    if (camera_id == 0) {
        eye_centre_homogeneous_transformed = eye_centre_homogeneous * M_left_;
    } else {
        eye_centre_homogeneous_transformed = eye_centre_homogeneous * M_right_;
    }

    //Convert back to Vec3d
    eye_centre[0] = eye_centre_homogeneous_transformed.at<float>(0, 0);
    eye_centre[1] = eye_centre_homogeneous_transformed.at<float>(0, 1);
    eye_centre[2] = eye_centre_homogeneous_transformed.at<float>(0, 2);
}

void EyeTracker::getCorneaCentrePosition(cv::Vec3d &cornea_centre, int camera_id) {
    cv::Vec3d base_cornea_centre;
    eye_estimators_[camera_id].getCorneaCurvaturePosition(base_cornea_centre);
    //Convert Vec3d to homogeneous coordinates
    cv::Mat cornea_centre_homogeneous =
        (cv::Mat_<float>(1, 4) << base_cornea_centre[0], base_cornea_centre[1], base_cornea_centre[2], 1);

    //Multiply by M_left_ or M_right_ depending on camera_id
    cv::Mat cornea_centre_homogeneous_transformed;
    if (camera_id == 0) {
        cornea_centre_homogeneous_transformed = cornea_centre_homogeneous * M_left_;
    } else {
        cornea_centre_homogeneous_transformed = cornea_centre_homogeneous * M_right_;
    }

    //Convert back to Vec3d
    cornea_centre[0] = cornea_centre_homogeneous_transformed.at<float>(0, 0);
    cornea_centre[1] = cornea_centre_homogeneous_transformed.at<float>(0, 1);
    cornea_centre[2] = cornea_centre_homogeneous_transformed.at<float>(0, 2);
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
    output << frame_counter_ - 1 << "," << cornea_centre[0] << "," << cornea_centre[1] << "," << cornea_centre[2] << ","
           << eye_centre[0] << "," << eye_centre[1] << "," << eye_centre[2] << "\n";
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
            std::string filename = "videos/" + current_time + "_" + std::to_string(i) + ".mp4";
            std::clog << "Saving video to " << filename << "\n";
            output_video_[i].open(filename, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30,
                                  et::Settings::parameters.camera_params[i].region_of_interest, false);
            filename = "videos/" + current_time + "_" + std::to_string(i) + "_ui.mp4";
            output_video_ui_[i].open(filename, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30,
                                     et::Settings::parameters.camera_params[i].region_of_interest, true);
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
        std::string filename{"images/" + Utils::getCurrentTimeText() + "_" + std::to_string(i) + ".png"};
        imwrite(filename, analyzed_frame_[i].glints);
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
            visualizer_[i].prepareImage(analyzed_frame_[i].glints);
            break;
        case VisualizationType::THRESHOLD_PUPIL:
            visualizer_[i].prepareImage(feature_detectors_[i].getThresholdedPupilImage());
            break;
        case VisualizationType::THRESHOLD_GLINTS:
            visualizer_[i].prepareImage(feature_detectors_[i].getThresholdedGlintsImage());
            break;
        default:
            break;
        }
        if (visualization_type_ != VisualizationType::DISABLED) {
            visualizer_[i].drawBoundingCircle(Settings::parameters.detection_params[i].pupil_search_centre,
                                              Settings::parameters.detection_params[i].pupil_search_radius);
            visualizer_[i].drawPupil(feature_detectors_[i].getDistortedPupil(),
                                     feature_detectors_[i].getDistortedPupilRadius());
            visualizer_[i].drawEyeCentre(eye_estimators_[i].getEyeCentrePixelPosition());
            visualizer_[i].drawCorneaCentre(eye_estimators_[i].getCorneaCurvaturePixelPosition());

            if (ellipse_fitting_[i]) {
                visualizer_[i].drawGlintEllipse(feature_detectors_[i].getDistortedEllipse());
            }
            visualizer_[i].drawGlints(feature_detectors_[i].getDistortedGlints());

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

std::string EyeTracker::startEyeVideoRecording(const std::string &folder_name) {
    fs::path output_path = fs::path(settings_path_) / folder_name;
    std::string timestamp = Utils::getCurrentTimeText();
    if (!std::filesystem::is_directory(output_path)) {
        std::filesystem::create_directory(output_path);
    }

    for (auto &i : camera_ids_) {
        if (eye_video_[i].isOpened()) {
            continue;
        }
        std::string filename = output_path / (timestamp + "_" + std::to_string(i) + ".mp4");
        eye_video_[i].open(filename, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30,
                           et::Settings::parameters.camera_params[i].region_of_interest, false);
    }

    if (!eye_data_.is_open()) {
        eye_data_.open(output_path / (timestamp + ".csv"));
    }
    eye_frame_counter_ = 0;
    return timestamp;
}

void EyeTracker::stopEyeVideoRecording() {
    for (auto &i : camera_ids_) {
        if (eye_video_[i].isOpened()) {
            eye_video_[i].release();
        }
    }

    if (eye_data_.is_open()) {
        eye_data_.close();
    }
}

void EyeTracker::saveEyeData(const std::string &eye_data) {
    for (auto &i : camera_ids_) {
        eye_video_[i].write(analyzed_frame_[i].glints);
    }

    eye_data_ << eye_frame_counter_ << "," << eye_data << "\n";
    eye_frame_counter_++;
}

void EyeTracker::calibrateTransform(const cv::Mat &M_et_left, const cv::Mat &M_et_right,
                                    const std::string &plane_calibration_path,
                                    const std::string &gaze_calibration_path) {

    per_user_transformation_blocker_.lock();
    bool previous_state_left = feature_detectors_[0].setKalmanFiltering(false);
    bool previous_state_right = feature_detectors_[1].setKalmanFiltering(false);
    fs::path plane_calibration_video_left = fs::path(settings_path_) / (plane_calibration_path + "_0.mp4");
    fs::path plane_calibration_video_right = fs::path(settings_path_) / (plane_calibration_path + "_1.mp4");
    fs::path plane_calibration_csv = fs::path(settings_path_) / (plane_calibration_path + ".csv");

    fs::path gaze_calibration_video_left = fs::path(settings_path_) / (gaze_calibration_path + "_0.mp4");
    fs::path gaze_calibration_video_right = fs::path(settings_path_) / (gaze_calibration_path + "_1.mp4");
    fs::path gaze_calibration_csv = fs::path(settings_path_) / (gaze_calibration_path + ".csv");

    // Read left and right eye position from the first row of gaze_calibration.csv. The position is kept constant throughout the calibration.
    std::ifstream gaze_calibration_file(gaze_calibration_csv);
    std::string line;
    std::getline(gaze_calibration_file, line);
    //Csv line: frame number, left eye x, left eye y, left eye z, right eye x, right eye y, right eye z, marker x, marker y, marker z
    std::vector<std::string> tokens = Utils::split(line, ',');
    cv::Point3f left_eye_position_world(std::stof(tokens[1]), std::stof(tokens[2]), std::stof(tokens[3]));
    cv::Point3f right_eye_position_world(std::stof(tokens[4]), std::stof(tokens[5]), std::stof(tokens[6]));

    // Convert left and right eye position to homogeneous coordinates and multiply by M_et_left and M_et_right respectively.
    cv::Mat left_eye_position_homogeneous =
        (cv::Mat_<double>(1, 4) << left_eye_position_world.x, left_eye_position_world.y, left_eye_position_world.z, 1)
        * M_et_left;
    cv::Mat right_eye_position_homogeneous = (cv::Mat_<double>(1, 4) << right_eye_position_world.x,
                                              right_eye_position_world.y, right_eye_position_world.z, 1)
        * M_et_right;

    // Convert from homogeneous to cartesian
    cv::Point3f left_eye_position(
        left_eye_position_homogeneous.at<double>(0, 0) / left_eye_position_homogeneous.at<double>(0, 3),
        left_eye_position_homogeneous.at<double>(0, 1) / left_eye_position_homogeneous.at<double>(0, 3),
        left_eye_position_homogeneous.at<double>(0, 2) / left_eye_position_homogeneous.at<double>(0, 3));

    cv::Point3f right_eye_position(
        right_eye_position_homogeneous.at<double>(0, 0) / right_eye_position_homogeneous.at<double>(0, 3),
        right_eye_position_homogeneous.at<double>(0, 1) / right_eye_position_homogeneous.at<double>(0, 3),
        right_eye_position_homogeneous.at<double>(0, 2) / right_eye_position_homogeneous.at<double>(0, 3));

    std::vector<cv::Point3f> left_eye_positions_real;
    std::vector<cv::Point3f> right_eye_positions_real;

    std::vector<cv::Point3f> left_eye_positions_simulated;
    std::vector<cv::Point3f> right_eye_positions_simulated;

    cv::VideoCapture plane_calibration_video_left_capture(plane_calibration_video_left.string());
    cv::VideoCapture plane_calibration_video_right_capture(plane_calibration_video_right.string());

    cv::VideoCapture gaze_calibration_video_left_capture(gaze_calibration_video_left.string());
    cv::VideoCapture gaze_calibration_video_right_capture(gaze_calibration_video_right.string());

    // Read the first frame of the video
    cv::Mat frame_left;
    cv::Mat frame_right;

    std::ifstream plane_calibration_file(plane_calibration_csv);

    std::ifstream *files[] = {&plane_calibration_file, &gaze_calibration_file};

    for (int i = 0; i < 2; i++) {
        while (std::getline(*(files[i]), line)) {
            std::vector<std::string> tokens = Utils::split(line, ',');
            cv::Point3f corner_pos_world{};
            if (i == 0) {
                //Csv line: frame number, corner x, corner y, corner z
                corner_pos_world = cv::Point3f(std::stof(tokens[1]), std::stof(tokens[2]), std::stof(tokens[3]));
            } else {
                //Csv line: frame number, left eye x, left eye y, left eye z, right eye x, right eye y, right eye z, marker x, marker y, marker z
                corner_pos_world = cv::Point3f(std::stof(tokens[7]), std::stof(tokens[8]), std::stof(tokens[9]));
            }

            // Convert marker position to homogeneous coordinates and multiply by M_et_left and M_et_right respectively.
            cv::Mat corner_pos_homo =
                (cv::Mat_<double>(1, 4) << corner_pos_world.x, corner_pos_world.y, corner_pos_world.z, 1) * M_et_left;

            // Convert from homogeneous to cartesian
            cv::Point3f corner_pos(corner_pos_homo.at<double>(0, 0) / corner_pos_homo.at<double>(0, 3),
                                   corner_pos_homo.at<double>(0, 1) / corner_pos_homo.at<double>(0, 3),
                                   corner_pos_homo.at<double>(0, 2) / corner_pos_homo.at<double>(0, 3));

            // Calculate the nodal point position
            cv::Point3f nodal_point_position_left = Utils::calculateNodalPointPosition(corner_pos, left_eye_position);
            cv::Point3f nodal_point_position_right = Utils::calculateNodalPointPosition(corner_pos, right_eye_position);

            // Read frames from both videos
            if (i == 0) {
                plane_calibration_video_left_capture >> frame_left;
                plane_calibration_video_right_capture >> frame_right;
            } else {
                gaze_calibration_video_left_capture >> frame_left;
                gaze_calibration_video_right_capture >> frame_right;
            }

            cv::cvtColor(frame_left, frame_left, cv::COLOR_BGR2GRAY);
            cv::cvtColor(frame_right, frame_right, cv::COLOR_BGR2GRAY);

            feature_detectors_[0].preprocessImage({frame_left, frame_left});
            bool features_found = feature_detectors_[0].findPupil();
            features_found &= feature_detectors_[0].findEllipse();

            feature_detectors_[1].preprocessImage({frame_right, frame_right});
            features_found &= feature_detectors_[1].findPupil();
            features_found &= feature_detectors_[1].findEllipse();

            if (features_found) {
                cv::Point2f pupil_left = feature_detectors_[0].getPupil();
                cv::RotatedRect ellipse_left = feature_detectors_[0].getEllipse();
                cv::Point2f pupil_right = feature_detectors_[1].getPupil();
                cv::RotatedRect ellipse_right = feature_detectors_[1].getEllipse();

                eye_estimators_[0].getEyeFromPolynomial(pupil_left, ellipse_left);
                eye_estimators_[1].getEyeFromPolynomial(pupil_right, ellipse_right);
                cv::Vec3d cornea_centre_left{}, cornea_centre_right{}, eye_centre_left{}, eye_centre_right{};
                eye_estimators_[0].getCorneaCurvaturePosition(cornea_centre_left);
                eye_estimators_[0].getEyeCentrePosition(eye_centre_left);
                eye_estimators_[1].getCorneaCurvaturePosition(cornea_centre_right);
                eye_estimators_[1].getEyeCentrePosition(eye_centre_right);

                //Convert Vec3d to Point3f
                cv::Point3f cornea_centre_left_point(cornea_centre_left[0], cornea_centre_left[1],
                                                     cornea_centre_left[2]);
                cv::Point3f cornea_centre_right_point(cornea_centre_right[0], cornea_centre_right[1],
                                                      cornea_centre_right[2]);
                cv::Point3f eye_centre_left_point(eye_centre_left[0], eye_centre_left[1], eye_centre_left[2]);
                cv::Point3f eye_centre_right_point(eye_centre_right[0], eye_centre_right[1], eye_centre_right[2]);

                left_eye_positions_simulated.push_back(eye_centre_left_point);
                left_eye_positions_real.push_back(left_eye_position);

                left_eye_positions_simulated.push_back(cornea_centre_left_point);
                left_eye_positions_real.push_back(nodal_point_position_left);

                right_eye_positions_simulated.push_back(eye_centre_right_point);
                right_eye_positions_real.push_back(right_eye_position);

                right_eye_positions_simulated.push_back(cornea_centre_right_point);
                right_eye_positions_real.push_back(nodal_point_position_right);
            }
        }
    }

    // Convert the vectors to matrices
    cv::Mat left_eye_positions_real_mat(left_eye_positions_real.size(), 3, CV_32F);
    cv::Mat left_eye_positions_simulated_mat(left_eye_positions_simulated.size(), 3, CV_32F);
    cv::Mat right_eye_positions_real_mat(right_eye_positions_real.size(), 3, CV_32F);
    cv::Mat right_eye_positions_simulated_mat(right_eye_positions_simulated.size(), 3, CV_32F);
    for (int i = 0; i < left_eye_positions_real.size(); i++) {
        left_eye_positions_real_mat.at<float>(i, 0) = left_eye_positions_real[i].x;
        left_eye_positions_real_mat.at<float>(i, 1) = left_eye_positions_real[i].y;
        left_eye_positions_real_mat.at<float>(i, 2) = left_eye_positions_real[i].z;
    }
    for (int i = 0; i < left_eye_positions_simulated.size(); i++) {
        left_eye_positions_simulated_mat.at<float>(i, 0) = left_eye_positions_simulated[i].x;
        left_eye_positions_simulated_mat.at<float>(i, 1) = left_eye_positions_simulated[i].y;
        left_eye_positions_simulated_mat.at<float>(i, 2) = left_eye_positions_simulated[i].z;
    }
    for (int i = 0; i < right_eye_positions_real.size(); i++) {
        right_eye_positions_real_mat.at<float>(i, 0) = right_eye_positions_real[i].x;
        right_eye_positions_real_mat.at<float>(i, 1) = right_eye_positions_real[i].y;
        right_eye_positions_real_mat.at<float>(i, 2) = right_eye_positions_real[i].z;
    }
    for (int i = 0; i < right_eye_positions_simulated.size(); i++) {
        right_eye_positions_simulated_mat.at<float>(i, 0) = right_eye_positions_simulated[i].x;
        right_eye_positions_simulated_mat.at<float>(i, 1) = right_eye_positions_simulated[i].y;
        right_eye_positions_simulated_mat.at<float>(i, 2) = right_eye_positions_simulated[i].z;
    }

    // Convert Mat to homogeneous coordinates
    left_eye_positions_real_mat = Utils::convertToHomogeneous(left_eye_positions_real_mat);
    right_eye_positions_real_mat = Utils::convertToHomogeneous(right_eye_positions_real_mat);
    left_eye_positions_simulated_mat = Utils::convertToHomogeneous(left_eye_positions_simulated_mat);
    right_eye_positions_simulated_mat = Utils::convertToHomogeneous(right_eye_positions_simulated_mat);
    left_eye_positions_real_mat.convertTo(left_eye_positions_real_mat, CV_32F);
    right_eye_positions_real_mat.convertTo(right_eye_positions_real_mat, CV_32F);
    left_eye_positions_simulated_mat.convertTo(left_eye_positions_simulated_mat, CV_32F);
    right_eye_positions_simulated_mat.convertTo(right_eye_positions_simulated_mat, CV_32F);

    // Find transformation matrices between the simulated and real eye positions
    M_left_ = Utils::findTransformationMatrix(left_eye_positions_simulated_mat, left_eye_positions_real_mat);
    M_right_ = Utils::findTransformationMatrix(right_eye_positions_simulated_mat, right_eye_positions_real_mat);

    feature_detectors_[0].setKalmanFiltering(previous_state_left);
    feature_detectors_[1].setKalmanFiltering(previous_state_right);
    per_user_transformation_blocker_.unlock();
}

} // namespace et