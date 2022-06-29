#include "Settings.hpp"

#include "json.hpp"

#include <algorithm>
#include <fstream>

using nlohmann::json;

namespace et {

Parameters Settings::parameters{};

void from_json(const json &j, CameraParams &camera_params) {
    j.at("name").get_to(camera_params.name);
    std::vector<int> idata{};
    j.at("dimensions").get_to(idata);
    camera_params.dimensions = cv::Size2i(idata[0], idata[1]);
    idata.clear();
    j.at("region_of_interest").get_to(idata);
    camera_params.region_of_interest = cv::Size2i(idata[0], idata[1]);
    idata.clear();
    j.at("capture_offset").get_to(idata);
    camera_params.capture_offset = cv::Size2i(idata[0], idata[1]);
    j.at("framerate").get_to(camera_params.framerate);
    j.at("exposure").get_to(camera_params.exposure);
    j.at("gamma").get_to(camera_params.gamma);
    std::vector<float> ddata{};
    j.at("intrinsic_matrix_opencv").get_to(ddata);
    camera_params.intrinsic_matrix = cv::Mat(3, 3, CV_32FC1);
    for (int i = 0; i < 3; i++) {
        for (int k = 0; k < 3; k++) {
            camera_params.intrinsic_matrix.at<float>(cv::Point(i, k)) =
                ddata[i * 3 + k];
        }
    }
    ddata.clear();
    j.at("distortion_coefficients").get_to(ddata);
    for (int i = 0; i < 5; i++) {
        camera_params.distortion_coefficients[i] = static_cast<float>(ddata[i]);
    }
}

void to_json(json &j, const CameraParams &camera_params) {
    j["name"] = camera_params.name;
    j["dimensions"] = {camera_params.dimensions.width,
                          camera_params.dimensions.height};
    j["region_of_interest"] = {camera_params.region_of_interest.width,
                                  camera_params.region_of_interest.height};
    j["capture_offset"] = {camera_params.capture_offset.width,
                              camera_params.capture_offset.height};

    j["framerate"] = camera_params.framerate;
    j["exposure"] = camera_params.exposure;
    j["gamma"] = camera_params.gamma;
    for (int i = 0; i < 3; i++) {
        for (int k = 0; k < 3; k++) {
            j["intrinsic_matrix_opencv"][i * 3 + k] =
                camera_params.intrinsic_matrix.at<float>(cv::Point(i, k));
        }
    }
    for (int i = 0; i < 5; i++) {
        j["distortion_coefficients"][i] =
            camera_params.distortion_coefficients[i];
    }
}

void from_json(const json &j, EyeParams &eye_params) {
    j.at("cornea_curvature_radius").get_to(eye_params.cornea_curvature_radius);
    j.at("pupil_cornea_distance").get_to(eye_params.pupil_cornea_distance);
    j.at("cornea_refraction_index").get_to(eye_params.cornea_refraction_index);
    j.at("eyeball_radius").get_to(eye_params.eyeball_radius);
    j.at("pupil_eye_centre_distance")
        .get_to(eye_params.pupil_eye_centre_distance);
}

void to_json(json &j, const EyeParams &eye_params) {
    j["cornea_curvature_radius"] = eye_params.cornea_curvature_radius;
    j["pupil_cornea_distance"] = eye_params.pupil_cornea_distance;
    j["cornea_refraction_index"] = eye_params.cornea_refraction_index;
    j["eyeball_radius"] = eye_params.eyeball_radius;
    j["pupil_eye_centre_distance"] = eye_params.pupil_eye_centre_distance;
}

void from_json(
    const json &j,
    std::unordered_map<std::string, FeaturesParams> &features_params) {
    for (const auto &item : j.items()) {
        std::string name = item.key();
        auto value = item.value();
        value.at("min_pupil_radius")
            .get_to(features_params[name].min_pupil_radius);
        value.at("max_pupil_radius")
            .get_to(features_params[name].max_pupil_radius);
        value.at("pupil_threshold")
            .get_to(features_params[name].pupil_threshold);
        value.at("glint_threshold")
            .get_to(features_params[name].glint_threshold);
        value.at("min_glint_radius")
            .get_to(features_params[name].min_glint_radius);
        value.at("max_glint_radius")
            .get_to(features_params[name].max_glint_radius);
        value.at("glint_bottom_hor_distance")
            .get_to(features_params[name].glint_bottom_hor_distance);
        value.at("glint_bottom_vert_distance")
            .get_to(features_params[name].glint_bottom_vert_distance);
        value.at("glint_right_hor_distance")
            .get_to(features_params[name].glint_right_hor_distance);
        value.at("glint_right_vert_distance")
            .get_to(features_params[name].glint_right_vert_distance);
        value.at("max_hor_glint_pupil_distance")
            .get_to(features_params[name].max_hor_glint_pupil_distance);
        value.at("max_vert_glint_pupil_distance")
            .get_to(features_params[name].max_vert_glint_pupil_distance);
        value.at("alpha").get_to(features_params[name].alpha);
        value.at("beta").get_to(features_params[name].beta);
    }
}

void to_json(
    json &j,
    const std::unordered_map<std::string, FeaturesParams> &features_params) {
    for (const auto &item : features_params) {
        std::string name = item.first;
        auto value = item.second;
        j[name]["min_pupil_radius"] =
            features_params.at(name).min_pupil_radius;
        j[name]["max_pupil_radius"] =
            features_params.at(name).max_pupil_radius;
        j[name]["pupil_threshold"] =
            features_params.at(name).pupil_threshold;
        j[name]["glint_threshold"] =
            features_params.at(name).glint_threshold;
        j[name]["min_glint_radius"] =
            features_params.at(name).min_glint_radius;
        j[name]["max_glint_radius"] =
            features_params.at(name).max_glint_radius;
        j[name]["glint_bottom_hor_distance"] =
            features_params.at(name).glint_bottom_hor_distance;
        j[name]["glint_bottom_vert_distance"] =
            features_params.at(name).glint_bottom_vert_distance;
        j[name]["glint_right_hor_distance"] =
            features_params.at(name).glint_right_hor_distance;
        j[name]["glint_right_vert_distance"] =
            features_params.at(name).glint_right_vert_distance;
        j[name]["min_pupil_radius"] =
            features_params.at(name).min_pupil_radius;
        j[name]["max_hor_glint_pupil_distance"] =
            features_params.at(name).max_hor_glint_pupil_distance;
        j[name]["max_vert_glint_pupil_distance"] =
            features_params.at(name).max_vert_glint_pupil_distance;
        j[name]["alpha"] = features_params.at(name).alpha;
        j[name]["beta"] = features_params.at(name).beta;
    }
}

void from_json(const json &j, Parameters &parameters) {
    j.at("camera_params").get_to(parameters.camera_params);
    std::vector<std::vector<float>> data{};
    j.at("led_positions").get_to(data);

    static cv::Vec3f origin{1e6, 1e6, 1e6};
    origin(0) = origin(1) = origin(2) = 1e6;
    for (const auto &item : data) {
        parameters.leds_positions.push_back({item[0], item[1], item[2]});
        for (int i = 0; i < 3; i++) {
            origin(i) = std::min(origin(i), item[i]);
        }
    }

    std::sort(
        parameters.leds_positions.begin(), parameters.leds_positions.end(),
        [](const auto &lhs, const auto &rhs) {
            float dist_lhs = cv::norm(cv::Vec3f(1, 2, 1).mul(lhs - origin));
            float dist_rhs = cv::norm(cv::Vec3f(1, 2, 1).mul(rhs - origin));
            return dist_lhs < dist_rhs;
        });

    j.at("eye_params").get_to(parameters.eye_params);
    j.at("features_params").get_to(parameters.features_params);
}

void to_json(json &j, const Parameters &parameters) {
    j["camera_params"] = parameters.camera_params;
    std::vector<std::vector<float>> data{};
    for (int i = 0; i < parameters.leds_positions.size(); i++) {
        j["led_positions"][i] = {parameters.leds_positions[i](0),
                                    parameters.leds_positions[i](1),
                                    parameters.leds_positions[i](2)};
    }
    j["eye_params"] = parameters.eye_params;
    j["features_params"] = parameters.features_params;
}

Settings::Settings(std::string file_path) {
    std::ifstream file(file_path);
    json j;
    file >> j;
    parameters = j.get<Parameters>();
    file.close();
}

void Settings::saveSettings(std::string file_path) {
    std::ofstream file(file_path);
    json j{parameters};
    file << j[0];
    file.close();
}
} // namespace et
