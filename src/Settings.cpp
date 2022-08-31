#include "Settings.hpp"

#include "json.hpp"

#include <algorithm>
#include <fstream>

using nlohmann::json;

namespace et {

Parameters Settings::parameters{};

void from_json(const json &j, CameraParams &camera_params) {
    j.at("serial_number").get_to(camera_params.serial_number);
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
    j.at("pupil_exposure").get_to(camera_params.pupil_exposure);
    j.at("glint_exposure").get_to(camera_params.glint_exposure);
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
    j.at("gaze_shift").get_to(ddata);
    camera_params.gaze_shift = cv::Vec3f(ddata[0], ddata[1], ddata[2]);
}

void to_json(json &j, const CameraParams &camera_params) {
    j["serial_number"] = camera_params.serial_number;
    j["dimensions"] = {camera_params.dimensions.width,
                       camera_params.dimensions.height};
    j["region_of_interest"] = {camera_params.region_of_interest.width,
                               camera_params.region_of_interest.height};
    j["capture_offset"] = {camera_params.capture_offset.width,
                           camera_params.capture_offset.height};

    j["framerate"] = camera_params.framerate;
    j["pupil_exposure"] = camera_params.pupil_exposure;
    j["glint_exposure"] = camera_params.glint_exposure;
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
    for (int i = 0; i < 3; i++) {
        j["gaze_shift"][i] = camera_params.gaze_shift(i);
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

void from_json(const json &j, DetectionParams &detection_params) {
    j.at("min_pupil_radius").get_to(detection_params.min_pupil_radius);
    j.at("max_pupil_radius").get_to(detection_params.max_pupil_radius);
    j.at("min_glint_radius").get_to(detection_params.min_glint_radius);
    j.at("max_glint_radius").get_to(detection_params.max_glint_radius);
    j.at("glint_bottom_hor_distance")
        .get_to(detection_params.glint_bottom_hor_distance);
    j.at("glint_bottom_vert_distance")
        .get_to(detection_params.glint_bottom_vert_distance);
    j.at("glint_right_hor_distance")
        .get_to(detection_params.glint_right_hor_distance);
    j.at("glint_right_vert_distance")
        .get_to(detection_params.glint_right_vert_distance);
    j.at("max_hor_glint_pupil_distance")
        .get_to(detection_params.max_hor_glint_pupil_distance);
    j.at("max_vert_glint_pupil_distance")
        .get_to(detection_params.max_vert_glint_pupil_distance);
    j.at("glint_dilate_size").get_to(detection_params.glint_dilate_size);
    j.at("glint_erode_size").get_to(detection_params.glint_erode_size);
    j.at("glint_close_size").get_to(detection_params.glint_close_size);
    j.at("pupil_dilate_size").get_to(detection_params.pupil_dilate_size);
    j.at("pupil_erode_size").get_to(detection_params.pupil_erode_size);
    j.at("pupil_close_size").get_to(detection_params.pupil_close_size);

    std::vector<float> data{};
    j.at("pupil_search_centre").at("left").get_to(data);
    detection_params.pupil_search_centre[0] = {data[0], data[1]};
    data.clear();
    j.at("pupil_search_centre").at("right").get_to(data);
    detection_params.pupil_search_centre[1] = {data[0], data[1]};

    j.at("pupil_search_radius")
        .at("left")
        .get_to(detection_params.pupil_search_radius[0]);
    j.at("pupil_search_radius")
        .at("right")
        .get_to(detection_params.pupil_search_radius[1]);
}

void to_json(json &j, const DetectionParams &detection_params) {
    j["min_pupil_radius"] = detection_params.min_pupil_radius;
    j["max_pupil_radius"] = detection_params.max_pupil_radius;
    j["min_glint_radius"] = detection_params.min_glint_radius;
    j["max_glint_radius"] = detection_params.max_glint_radius;
    j["glint_bottom_hor_distance"] = detection_params.glint_bottom_hor_distance;
    j["glint_bottom_vert_distance"] =
        detection_params.glint_bottom_vert_distance;
    j["glint_right_hor_distance"] = detection_params.glint_right_hor_distance;
    j["glint_right_vert_distance"] = detection_params.glint_right_vert_distance;
    j["max_hor_glint_pupil_distance"] =
        detection_params.max_hor_glint_pupil_distance;
    j["max_vert_glint_pupil_distance"] =
        detection_params.max_vert_glint_pupil_distance;
    j["glint_dilate_size"] = detection_params.glint_dilate_size;
    j["glint_erode_size"] = detection_params.glint_erode_size;
    j["glint_close_size"] = detection_params.glint_close_size;
    j["pupil_dilate_size"] = detection_params.pupil_dilate_size;
    j["pupil_erode_size"] = detection_params.pupil_erode_size;
    j["pupil_close_size"] = detection_params.pupil_close_size;
    j["pupil_search_centre"]["left"] = {
        detection_params.pupil_search_centre[0].x,
        detection_params.pupil_search_centre[0].y};
    j["pupil_search_centre"]["right"] = {
        detection_params.pupil_search_centre[1].x,
        detection_params.pupil_search_centre[1].y};
    j["pupil_search_radius"]["left"] = detection_params.pupil_search_radius[0];
    j["pupil_search_radius"]["right"] = detection_params.pupil_search_radius[1];
}

void from_json(
    const json &j,
    std::unordered_map<std::string, FeaturesParams> &features_params) {
    for (const auto &item : j.items()) {
        std::string name = item.key();
        auto value = item.value();
        value.at("pupil_threshold").at("left")
            .get_to(features_params[name].pupil_threshold[0]);
        value.at("pupil_threshold").at("right")
            .get_to(features_params[name].pupil_threshold[1]);
        value.at("glint_threshold").at("left")
            .get_to(features_params[name].glint_threshold[0]);
        value.at("glint_threshold").at("right")
            .get_to(features_params[name].glint_threshold[1]);
        value.at("alpha").get_to(features_params[name].alpha);
        value.at("beta").get_to(features_params[name].beta);
    }
}

void to_json(
    json &j,
    const std::unordered_map<std::string, FeaturesParams> &features_params) {
    for (const auto &item : features_params) {
        std::string name = item.first;
        j[name]["pupil_threshold"]["left"] = features_params.at(name).pupil_threshold[0];
        j[name]["pupil_threshold"]["right"] = features_params.at(name).pupil_threshold[1];
        j[name]["glint_threshold"]["left"] = features_params.at(name).glint_threshold[0];
        j[name]["glint_threshold"]["right"] = features_params.at(name).glint_threshold[1];
        j[name]["alpha"] = features_params.at(name).alpha;
        j[name]["beta"] = features_params.at(name).beta;
    }
}

void from_json(const json &j, Parameters &parameters) {
    j.at("camera_params").at("left").get_to(parameters.camera_params[0]);
    j.at("camera_params").at("right").get_to(parameters.camera_params[1]);

    std::string_view side_names[] = {"left", "right"};
    for (int i = 0; i < 2; i++) {
        std::vector<std::vector<float>> data{};
        j.at("led_positions").at(side_names[i]).get_to(data);

        static cv::Vec3f origin{1e6, 1e6, 1e6};
        origin(0) = origin(1) = origin(2) = 1e6;
        for (const auto &item : data) {
            parameters.leds_positions[i].push_back({item[0], item[1], item[2]});
            for (int k = 0; k < 3; k++) {
                origin(k) = std::min(origin(k), item[k]);
            }
        }
        data.clear();

        std::sort(parameters.leds_positions[i].begin(),
                  parameters.leds_positions[i].end(),
                  [](const auto &lhs, const auto &rhs) {
                      float dist_lhs =
                          cv::norm(cv::Vec3f(1, 2, 1).mul(lhs - origin));
                      float dist_rhs =
                          cv::norm(cv::Vec3f(1, 2, 1).mul(rhs - origin));
                      return dist_lhs < dist_rhs;
                  });
    }

    j.at("eye_params").get_to(parameters.eye_params);
    j.at("detection_params").get_to(parameters.detection_params);
    j.at("features_params").get_to(parameters.features_params);
}

void to_json(json &j, const Parameters &parameters) {
    j["camera_params"]["left"] = parameters.camera_params[0];
    j["camera_params"]["right"] = parameters.camera_params[1];
    std::vector<std::vector<float>> data{};
    for (int i = 0; i < parameters.leds_positions[0].size(); i++) {
        j["led_positions"]["left"][i] = {parameters.leds_positions[0][i](0),
                                         parameters.leds_positions[0][i](1),
                                         parameters.leds_positions[0][i](2)};
        j["led_positions"]["right"][i] = {parameters.leds_positions[1][i](0),
                                          parameters.leds_positions[1][i](1),
                                          parameters.leds_positions[1][i](2)};
    }
    j["eye_params"] = parameters.eye_params;
    j["detection_params"] = parameters.detection_params;
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
    file << j[0].dump(4);
    file.close();
}
} // namespace et
