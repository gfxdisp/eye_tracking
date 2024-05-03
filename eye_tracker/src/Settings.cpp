#include "eye_tracker/Settings.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>

using nlohmann::json;

namespace et {

    Parameters Settings::parameters{};

    std::filesystem::path Settings::settings_folder_{};

    void from_json(const json& j, CameraParams& camera_params) {
        j.at("serial_number").get_to(camera_params.serial_number);
        std::vector<int> i_data{};
        j.at("dimensions").get_to(i_data);
        camera_params.dimensions = cv::Size2i(i_data[0], i_data[1]);
        i_data.clear();
        j.at("region_of_interest").get_to(i_data);
        camera_params.region_of_interest = cv::Size2i(i_data[0], i_data[1]);
        i_data.clear();
        j.at("capture_offset").get_to(i_data);
        camera_params.capture_offset = cv::Size2i(i_data[0], i_data[1]);
        j.at("framerate").get_to(camera_params.framerate);
        j.at("pixel_clock").get_to(camera_params.pixel_clock);
        j.at("gamma").get_to(camera_params.gamma);
        j.at("distortion_coefficients").get_to(camera_params.distortion_coefficients);
        std::vector<double> d_data{};
        j.at("intrinsic_matrix").get_to(d_data);
        camera_params.intrinsic_matrix = cv::Mat(3, 3, CV_64FC1);
        for (int i = 0; i < 3; i++) {
            for (int k = 0; k < 3; k++) {
                camera_params.intrinsic_matrix.at<double>(cv::Point(i, k)) = d_data[i * 3 + k];
            }
        }
        d_data.clear();
        j.at("extrinsic_matrix").get_to(d_data);
        camera_params.extrinsic_matrix = cv::Mat(4, 4, CV_64FC1);
        for (int i = 0; i < 4; i++) {
            for (int k = 0; k < 4; k++) {
                camera_params.extrinsic_matrix.at<double>(cv::Point(i, k)) = d_data[i * 4 + k];
            }
        }
        j.at("gaze_shift").get_to(d_data);
        camera_params.gaze_shift = cv::Vec3d(d_data[0], d_data[1], d_data[2]);
    }

    void to_json(json& j, const CameraParams& camera_params) {
        j["serial_number"] = camera_params.serial_number;
        j["dimensions"] = {camera_params.dimensions.width, camera_params.dimensions.height};
        j["region_of_interest"] = {camera_params.region_of_interest.width, camera_params.region_of_interest.height};
        j["capture_offset"] = {camera_params.capture_offset.width, camera_params.capture_offset.height};

        j["framerate"] = camera_params.framerate;
        j["pixel_clock"] = camera_params.pixel_clock;
        j["gamma"] = camera_params.gamma;
        for (int i = 0; i < 3; i++) {
            for (int k = 0; k < 3; k++) {
                j["intrinsic_matrix"][i * 3 + k] = camera_params.intrinsic_matrix.at<double>(cv::Point(i, k));
            }
        }
        for (int i = 0; i < 4; i++) {
            for (int k = 0; k < 4; k++) {
                j["extrinsic_matrix"][i * 4 + k] = camera_params.extrinsic_matrix.at<double>(cv::Point(i, k));
            }
        }
        for (int i = 0; i < camera_params.distortion_coefficients.size(); i++) {
            j["distortion_coefficients"][i] = camera_params.distortion_coefficients[i];
        }
        for (int i = 0; i < 3; i++) {
            j["gaze_shift"][i] = camera_params.gaze_shift(i);
        }
    }

    void from_json(const json& j, DetectionParams& detection_params) {
        j.at("min_pupil_radius").get_to(detection_params.min_pupil_radius);
        j.at("max_pupil_radius").get_to(detection_params.max_pupil_radius);
        j.at("min_glint_radius").get_to(detection_params.min_glint_radius);
        j.at("max_glint_radius").get_to(detection_params.max_glint_radius);
        j.at("min_glint_bottom_hor_distance").get_to(detection_params.min_glint_bottom_hor_distance);
        j.at("max_glint_bottom_hor_distance").get_to(detection_params.max_glint_bottom_hor_distance);
        j.at("min_glint_bottom_vert_distance").get_to(detection_params.min_glint_bottom_vert_distance);
        j.at("max_glint_bottom_vert_distance").get_to(detection_params.max_glint_bottom_vert_distance);
        j.at("min_glint_right_hor_distance").get_to(detection_params.min_glint_right_hor_distance);
        j.at("max_glint_right_hor_distance").get_to(detection_params.max_glint_right_hor_distance);
        j.at("min_glint_right_vert_distance").get_to(detection_params.min_glint_right_vert_distance);
        j.at("max_glint_right_vert_distance").get_to(detection_params.max_glint_right_vert_distance);
        j.at("max_hor_glint_pupil_distance").get_to(detection_params.max_hor_glint_pupil_distance);
        j.at("max_vert_glint_pupil_distance").get_to(detection_params.max_vert_glint_pupil_distance);

        std::vector<double> data{};
        j.at("pupil_search_centre").get_to(data);
        detection_params.pupil_search_centre = {data[0], data[1]};

        j.at("pupil_search_radius").get_to(detection_params.pupil_search_radius);
    }

    void to_json(json& j, const DetectionParams& detection_params) {
        j["min_pupil_radius"] = detection_params.min_pupil_radius;
        j["min_pupil_radius"] = detection_params.min_pupil_radius;
        j["max_pupil_radius"] = detection_params.max_pupil_radius;
        j["max_pupil_radius"] = detection_params.max_pupil_radius;
        j["min_glint_radius"] = detection_params.min_glint_radius;
        j["max_glint_radius"] = detection_params.max_glint_radius;
        j["min_glint_bottom_hor_distance"] = detection_params.min_glint_bottom_hor_distance;
        j["max_glint_bottom_hor_distance"] = detection_params.max_glint_bottom_hor_distance;
        j["min_glint_bottom_vert_distance"] = detection_params.min_glint_bottom_vert_distance;
        j["max_glint_bottom_vert_distance"] = detection_params.max_glint_bottom_vert_distance;
        j["min_glint_right_hor_distance"] = detection_params.min_glint_right_hor_distance;
        j["max_glint_right_hor_distance"] = detection_params.max_glint_right_hor_distance;
        j["min_glint_right_vert_distance"] = detection_params.min_glint_right_vert_distance;
        j["max_glint_right_vert_distance"] = detection_params.max_glint_right_vert_distance;
        j["max_hor_glint_pupil_distance"] = detection_params.max_hor_glint_pupil_distance;
        j["max_vert_glint_pupil_distance"] = detection_params.max_vert_glint_pupil_distance;
        j["pupil_search_centre"] = {detection_params.pupil_search_centre.x, detection_params.pupil_search_centre.y};
        j["pupil_search_radius"] = detection_params.pupil_search_radius;
    }

    void from_json(const json& j, std::unordered_map<std::string, FeaturesParams>& features_params) {
        for (const auto& item: j.items()) {
            std::string name = item.key();
            auto value = item.value();
            value.at("pupil_threshold").get_to(features_params[name].pupil_threshold);
            value.at("glint_threshold").get_to(features_params[name].glint_threshold);
            value.at("exposure").get_to(features_params[name].exposure);
            std::vector<double> data{};
            value.at("position_offset").get_to(data);
            features_params[name].position_offset = {data[0], data[1], data[2]};
            data.clear();
            value.at("angle_offset").get_to(data);
            features_params[name].angle_offset = {data[0], data[1]};
            data.clear();
            value.at("polynomial_x").get_to(features_params[name].polynomial_x);
            value.at("polynomial_y").get_to(features_params[name].polynomial_y);
            value.at("polynomial_theta").get_to(features_params[name].polynomial_theta);
            value.at("polynomial_phi").get_to(features_params[name].polynomial_phi);
            value.at("marker_depth").get_to(features_params[name].marker_depth);
        }
    }

    void to_json(json& j, const std::unordered_map<std::string, FeaturesParams>& features_params) {
        for (const auto& item: features_params) {
            std::string name = item.first;
            j[name]["pupil_threshold"] = features_params.at(name).pupil_threshold;
            j[name]["glint_threshold"] = features_params.at(name).glint_threshold;
            j[name]["exposure"] = features_params.at(name).exposure;
            j[name]["position_offset"] = {
                    features_params.at(name).position_offset.x,
                    features_params.at(name).position_offset.y,
                    features_params.at(name).position_offset.z
            };
            j[name]["angle_offset"] = {features_params.at(name).angle_offset.x, features_params.at(name).angle_offset.y};
            j[name]["polynomial_x"] = features_params.at(name).polynomial_x;
            j[name]["polynomial_y"] = features_params.at(name).polynomial_y;
            j[name]["polynomial_theta"] = features_params.at(name).polynomial_theta;
            j[name]["polynomial_phi"] = features_params.at(name).polynomial_phi;
            j[name]["marker_depth"] = features_params.at(name).marker_depth;
        }
    }

    void from_json(const json& j, Coefficients& coefficients) {
        j.at("eye_centre_pos_x").get_to(coefficients.eye_centre_pos_x);
        j.at("eye_centre_pos_y").get_to(coefficients.eye_centre_pos_y);
        j.at("eye_centre_pos_z").get_to(coefficients.eye_centre_pos_z);
        j.at("theta").get_to(coefficients.theta);
        j.at("phi").get_to(coefficients.phi);
    }

    void to_json(json& j, const Coefficients& coefficients) {
        j["eye_centre_pos_x"] = coefficients.eye_centre_pos_x;
        j["eye_centre_pos_y"] = coefficients.eye_centre_pos_y;
        j["eye_centre_pos_z"] = coefficients.eye_centre_pos_z;
        j["theta"] = coefficients.theta;
        j["phi"] = coefficients.phi;
    }

    void from_json(const json& j, EyeMeasurements& setup_variables) {
        j.at("cornea_centre_distance").get_to(setup_variables.cornea_centre_distance);
        j.at("cornea_curvature_radius").get_to(setup_variables.cornea_curvature_radius);
        j.at("cornea_refraction_index").get_to(setup_variables.cornea_refraction_index);
        j.at("pupil_cornea_distance").get_to(setup_variables.pupil_cornea_distance);
        j.at("alpha").get_to(setup_variables.alpha);
        j.at("beta").get_to(setup_variables.beta);
    }

    void to_json(json& j, const EyeMeasurements& setup_variables) {
        j["cornea_centre_distance"] = setup_variables.cornea_centre_distance;
        j["cornea_curvature_radius"] = setup_variables.cornea_curvature_radius;
        j["cornea_refraction_index"] = setup_variables.cornea_refraction_index;
        j["pupil_cornea_distance"] = setup_variables.pupil_cornea_distance;
        j["alpha"] = setup_variables.alpha;
        j["beta"] = setup_variables.beta;
    }

    void from_json(const json& j, PolynomialParams& polynomial_params) {
        j.at("coefficients").get_to(polynomial_params.coefficients);
        j.at("eye_measurements").get_to(polynomial_params.eye_measurements);
    }

    void to_json(json& j, const PolynomialParams& polynomial_params) {
        j["coefficients"] = polynomial_params.coefficients;
        j["eye_measurements"] = polynomial_params.eye_measurements;
    }

    void from_json(const json& j, Parameters& parameters) {
        j.at("camera_params").at("left").get_to(parameters.camera_params[0]);
        j.at("camera_params").at("right").get_to(parameters.camera_params[1]);

        std::string_view side_names[] = {"left", "right"};
        for (int i = 0; i < 2; i++) {
            std::vector<std::vector<double>> data{};
            j.at("led_positions").at(side_names[i]).get_to(data);
            for (const auto& item: data) {
                parameters.leds_positions[i].push_back({item[0], item[1], item[2]});
            }
            data.clear();
            std::vector<cv::Point2f> ellipse_points{};
            for (int k = 0; k < parameters.leds_positions[i].size(); k++) {
                ellipse_points.emplace_back(parameters.leds_positions[i][k](0),
                                            parameters.leds_positions[i][k](1));
            }
            auto ellipse = cv::fitEllipse(ellipse_points);
            std::sort(parameters.leds_positions[i].begin(), parameters.leds_positions[i].end(),
                      [&ellipse](const cv::Vec3d& a, const cv::Vec3d& b) {
                          if ((a(0) < ellipse.center.x && b(0) < ellipse.center.x) || (a(0) > ellipse.center.x && b(0) > ellipse.center.x)) {
                              return a(1) < b(1);
                          } else {
                              return a(0) < b(0);
                          }
                      });
        }

        j.at("detection_params").at("left").get_to(parameters.detection_params[0]);
        j.at("detection_params").at("right").get_to(parameters.detection_params[1]);
        j.at("features_params").at("left").get_to(parameters.features_params[0]);
        j.at("features_params").at("right").get_to(parameters.features_params[1]);
        j.at("polynomial_params").at("left").get_to(parameters.polynomial_params[0]);
        j.at("polynomial_params").at("right").get_to(parameters.polynomial_params[1]);
    }

    void to_json(json& j, const Parameters& parameters) {
        j["camera_params"]["left"] = parameters.camera_params[0];
        j["camera_params"]["right"] = parameters.camera_params[1];
        std::vector<std::vector<double>> data{};
        for (int i = 0; i < parameters.leds_positions[0].size(); i++) {
            j["led_positions"]["left"][i] = {
                    parameters.leds_positions[0][i](0), parameters.leds_positions[0][i](1),
                    parameters.leds_positions[0][i](2)
            };
            j["led_positions"]["right"][i] = {
                    parameters.leds_positions[1][i](0), parameters.leds_positions[1][i](1),
                    parameters.leds_positions[1][i](2)
            };
        }

        j["detection_params"]["left"] = parameters.detection_params[0];
        j["detection_params"]["right"] = parameters.detection_params[1];
        j["features_params"]["left"] = parameters.features_params[0];
        j["features_params"]["right"] = parameters.features_params[1];
        j["polynomial_params"]["left"] = parameters.polynomial_params[0];
        j["polynomial_params"]["right"] = parameters.polynomial_params[1];
    }

    Settings::Settings(const std::string& settings_folder) {
        settings_folder_ = settings_folder;
        loadSettings();
    }

    void Settings::loadSettings() {
        auto capture_params_path = std::filesystem::path(settings_folder_) / "capture_params.json";
        std::clog << "Loading settings from " << capture_params_path << "\n";

        std::ifstream file(capture_params_path);
        json j;
        file >> j;
        file.close();
        parameters = j.get<Parameters>();
        parameters.user_params[0] = &parameters.features_params[0]["default"];
        parameters.user_params[1] = &parameters.features_params[1]["default"];
    }

    void Settings::saveSettings() {
        auto capture_params_path = std::filesystem::path(settings_folder_) / "capture_params.json";
        std::ofstream file(capture_params_path);
        json j{parameters};
        file << j[0];
        file.close();
    }
} // namespace et
