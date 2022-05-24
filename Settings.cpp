#include "Settings.hpp"

#include "json.hpp"

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
    j.at("gamma").get_to(camera_params.gamma);
    std::vector<double> ddata{};
    j.at("intrinsic_matrix_opencv").get_to(ddata);
    for (int i = 0; i < 9; i++) {
        camera_params.intrinsic_matrix[i / 3][i % 3] =
            static_cast<float>(ddata[i]);
    }
    ddata.clear();
    j.at("distortion_coefficients").get_to(ddata);
    for (int i = 0; i < 5; i++) {
        camera_params.distortion_coefficients[i] = static_cast<float>(ddata[i]);
    }
}

void from_json(const json &j, std::vector<cv::Vec3f> &leds_positions) {
}

void from_json(const json &j, EyeParams &eye_params) {
    j.at("cornea_curvature_radius").get_to(eye_params.cornea_curvature_radius);
    j.at("pupil_cornea_distance").get_to(eye_params.pupil_cornea_distance);
    j.at("cornea_refraction_index").get_to(eye_params.cornea_refraction_index);
    j.at("eyeball_radius").get_to(eye_params.eyeball_radius);
    j.at("pupil_eye_centre_distance")
        .get_to(eye_params.pupil_eye_centre_distance);
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
        value.at("min_hor_glint_distance")
            .get_to(features_params[name].min_hor_glint_distance);
        value.at("max_hor_glint_distance")
            .get_to(features_params[name].max_hor_glint_distance);
        value.at("min_vert_glint_distance")
            .get_to(features_params[name].min_vert_glint_distance);
        value.at("max_vert_glint_distance")
            .get_to(features_params[name].max_vert_glint_distance);
    }
}

void from_json(const json &j, Parameters &parameters) {
    j.at("camera_params").get_to(parameters.camera_params);
    for (auto item : j.at("leds_params")) {
        std::vector<double> data{};
        item.at("position").get_to(data);
        parameters.leds_positions.emplace_back(static_cast<float>(data[0]),
                                               static_cast<float>(data[1]),
                                               static_cast<float>(data[2]));
    }
    j.at("eye_params").get_to(parameters.eye_params);
    j.at("features_params").get_to(parameters.features_params);
}

Settings::Settings(std::string file_path) {
    std::ifstream file(file_path);
    json j;
    file >> j;
    parameters = j.get<Parameters>();
    file.close();
}
}// namespace et
