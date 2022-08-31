#ifndef EYE_TRACKER_SETTINGS_HPP
#define EYE_TRACKER_SETTINGS_HPP

#include <opencv2/opencv.hpp>

#include <string>
#include <unordered_map>
#include <vector>

namespace et {

struct CameraParams {
    std::string serial_number{};
    cv::Size2i dimensions{};
    cv::Size2i region_of_interest{};
    cv::Size2i capture_offset{};
    float framerate{};
    float pupil_exposure{};
    float glint_exposure{};
    float gamma{};
    cv::Mat intrinsic_matrix{};
    float distortion_coefficients[5]{};
    cv::Vec3f gaze_shift{};
};

struct EyeParams {
    float cornea_curvature_radius{};
    float pupil_cornea_distance{};
    float cornea_refraction_index{};
    float eyeball_radius{};
    float pupil_eye_centre_distance{};
};

struct DetectionParams {
    float min_pupil_radius{};
    float max_pupil_radius{};
    float min_glint_radius{};
    float max_glint_radius{};
    float glint_bottom_hor_distance[2]{};
    float glint_bottom_vert_distance[2]{};
    float glint_right_hor_distance[2]{};
    float glint_right_vert_distance[2]{};
    float max_hor_glint_pupil_distance{};
    float max_vert_glint_pupil_distance{};
    int glint_dilate_size{};
    int glint_erode_size{};
    int glint_close_size{};
    int pupil_dilate_size{};
    int pupil_erode_size{};
    int pupil_close_size{};
    cv::Point2f pupil_search_centre[2]{};
    int pupil_search_radius[2]{};
};

struct FeaturesParams {
    int pupil_threshold[2]{};
    int glint_threshold[2]{};
    float alpha{};
    float beta{};
};

struct Parameters {
    CameraParams camera_params[2]{};
    std::vector<cv::Vec3f> leds_positions[2]{};
    EyeParams eye_params{};
    DetectionParams detection_params{};
    std::unordered_map<std::string, FeaturesParams> features_params{};
    FeaturesParams *user_params{};
};

class Settings {
public:
    explicit Settings(std::string file_path);
    void saveSettings(std::string file_path);
    static Parameters parameters;
};
}// namespace et

#endif//EYE_TRACKER_SETTINGS_HPP
