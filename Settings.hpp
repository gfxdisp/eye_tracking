#ifndef EYE_TRACKER_SETTINGS_HPP
#define EYE_TRACKER_SETTINGS_HPP

#include <opencv2/opencv.hpp>

#include <string>
#include <unordered_map>
#include <vector>

namespace et {

struct CameraParams {
    std::string name{};
    cv::Size2i dimensions{};
    cv::Size2i region_of_interest{};
    cv::Size2i capture_offset{};
    float framerate{};
    float exposure{};
    float gamma{};
    cv::Mat intrinsic_matrix{};
    float distortion_coefficients[5]{};
};

struct EyeParams {
    float cornea_curvature_radius{};
    float pupil_cornea_distance{};
    float cornea_refraction_index{};
    float eyeball_radius{};
    float pupil_eye_centre_distance{};
};

struct FeaturesParams {
    float min_pupil_radius{};
    float max_pupil_radius{};
    int pupil_threshold{};
    int glint_threshold{};
    float min_glint_radius{};
    float max_glint_radius{};
    float min_hor_glint_distance{};
    float max_hor_glint_distance{};
    float min_vert_glint_distance{};
    float max_vert_glint_distance{};
    float max_hor_glint_pupil_distance{};
    float max_vert_glint_pupil_distance{};
};

struct Parameters {
    CameraParams camera_params{};
    std::vector<cv::Vec3f> leds_positions{};
    EyeParams eye_params{};
    std::unordered_map<std::string, FeaturesParams> features_params{};
    FeaturesParams *user_params{};
};

class Settings {
public:
    explicit Settings(std::string file_path);
    static Parameters parameters;
};
}// namespace et

#endif//EYE_TRACKER_SETTINGS_HPP
