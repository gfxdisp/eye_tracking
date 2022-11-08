#ifndef HDRMFS_EYE_TRACKER_SETTINGS_HPP
#define HDRMFS_EYE_TRACKER_SETTINGS_HPP

#include <opencv2/opencv.hpp>

#include <string>
#include <unordered_map>
#include <vector>

namespace et {

/**
 * Internal camera parameters.
 */
struct CameraParams {
    // Serial number used to differentiate between left and right eye.
    std::string serial_number{};
    // Dimensions of the full image frame.
    cv::Size2i dimensions{};
    // Size of the region-of-interest extracted from the full image.
    cv::Size2i region_of_interest{};
    // Distance from top-left corner of the region-of-interest to the top-left
    // corner of the full image, measured in pixels separately for every axis.
    cv::Size2i capture_offset{};
    // Camera's framerate.
    float framerate{};
    // Camera's gamma.
    float gamma{};
    // Camera's pixel clock in MHz.
    float pixel_clock{};
    // Intrinsic matrix of the camera.
    cv::Mat intrinsic_matrix{};
    // Distortion coefficients of the camera.
    std::vector<float> distortion_coefficients{};
    // Constant shift needed to be added to the vector between eye and cornea centre.
    cv::Vec3f gaze_shift{};
};

/**
 * Internal parameters of the eye assumed to be constant.
 */
struct EyeParams {
    // Radius of the cornea.
    float cornea_curvature_radius{};
    // Distance between cornea centre and pupil centre.
    float pupil_cornea_distance{};
    // Refraction index inside the cornea.
    float cornea_refraction_index{};
    // Radius of the whole eye.
    float eyeball_radius{};
    // Distance between pupil centre and eye centre.
    float pupil_eye_centre_distance{};
    // Horizontal angle between optical and visual axis.
    float alpha{};
    // Vertical angle between optical and visual axis.
    float beta{};
};

/**
 * Parameters used to increase the eye features detection precision.
 */
struct DetectionParams {
    // Minimal radius of the pupil in pixels.
    float min_pupil_radius{};
    // Maximal radius of the pupil in pixels.
    float max_pupil_radius{};
    // Minimal radius of the glint in pixels.
    float min_glint_radius{};
    // Maximal radius of the glint in pixels.
    float max_glint_radius{};
    // Minimal horizontal distance between a glint and its bottom neighbour.
    float min_glint_bottom_hor_distance{};
    // Maximal horizontal distance between a glint and its bottom neighbour.
    float max_glint_bottom_hor_distance{};
    // Minimal vertical distance between a glint and its bottom neighbour.
    float min_glint_bottom_vert_distance{};
    // Maximal vertical distance between a glint and its bottom neighbour.
    float max_glint_bottom_vert_distance{};
    // Minimal horizontal distance between a glint and its right neighbour.
    float min_glint_right_hor_distance{};
    // Maximal horizontal distance between a glint and its right neighbour.
    float max_glint_right_hor_distance{};
    // Minimal vertical distance between a glint and its right neighbour.
    float min_glint_right_vert_distance{};
    // Maximal vertical distance between a glint and its right neighbour.
    float max_glint_right_vert_distance{};
    // Maximal horizontal distance between a glint and a pupil's centre.
    float max_hor_glint_pupil_distance{};
    // Maximal vertical distance between a glint and a pupil's centre.
    float max_vert_glint_pupil_distance{};
    // Centre of the circle aligned with the hole in the view piece in the image.
    cv::Point2f pupil_search_centre{};
    // Radius of the circle aligned with the hole in the view piece in the image.
    int pupil_search_radius{};
};

/**
 * Detection parameters calibrated individually per person.
 */
struct FeaturesParams {
    // Threshold value for pupil detection used by FeatureDetector.
    int pupil_threshold{};
    // Threshold value for glints detection used by FeatureDetector.
    int glint_threshold{};
    // Exposure of the camera in milliseconds.
    float exposure{};
};

/**
 * All parameters required for proper eye-tracking.
 */
struct Parameters {
    // Internal camera parameters. One struct per eye.
    CameraParams camera_params[2]{};
    // Vector of LED positions. One per eye.
    std::vector<cv::Vec3f> leds_positions[2]{};
    // Internal parameters of the eye assumed to be constant.
    EyeParams eye_params{};
    // Parameters used to increase the eye features detection precision.
    // One struct per eye.
    DetectionParams detection_params[2]{};
    // Detection parameters calibrated individually per person. One set per user.
    std::unordered_map<std::string, FeaturesParams> features_params[2]{};
    // Detection parameters for the current user.
    FeaturesParams *user_params[2]{};
};

/**
 * Loads settings from Json file and saves them back.
 */
class Settings {
public:
    /**
     * Loads settings from Json and stores in the parameters structure.
     * @param file_path Path to the json file.
     */
    explicit Settings(const std::string &file_path);
    /**
     * Save settings back to Json file form the parameters structure.
     * @param file_path Path to the saved json file.
     */
    static void saveSettings(const std::string &file_path);
    // Structure with all parameters required for proper eye-tracking.
    static Parameters parameters;
};
} // namespace et

#endif //HDRMFS_EYE_TRACKER_SETTINGS_HPP