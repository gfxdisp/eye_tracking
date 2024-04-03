#ifndef HDRMFS_EYE_TRACKER_SETTINGS_HPP
#define HDRMFS_EYE_TRACKER_SETTINGS_HPP

#include "eye_tracker/json.hpp"

#include <opencv2/opencv.hpp>

#include <string>
#include <unordered_map>
#include <vector>
#include <filesystem>

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
    double framerate{};
    // Camera's gamma.
    double gamma{};
    // Camera's pixel clock in MHz.
    double pixel_clock{};
    // Intrinsic matrix of the camera.
    cv::Mat intrinsic_matrix{};
    // Extrinsic matrix of the camera.
    cv::Mat extrinsic_matrix{};
    // Distortion coefficients of the camera.
    std::vector<double> distortion_coefficients{};
    // Constant shift needed to be added to the vector between eye and cornea centre.
    cv::Vec3d gaze_shift{};
};

/**
 * Parameters used to increase the eye features detection precision.
 */
struct DetectionParams {
    // Minimal radius of the pupil in pixels.
    double min_pupil_radius{};
    // Maximal radius of the pupil in pixels.
    double max_pupil_radius{};
    // Minimal radius of the glint in pixels.
    double min_glint_radius{};
    // Maximal radius of the glint in pixels.
    double max_glint_radius{};
    // Minimal horizontal distance between a glint and its bottom neighbour.
    double min_glint_bottom_hor_distance{};
    // Maximal horizontal distance between a glint and its bottom neighbour.
    double max_glint_bottom_hor_distance{};
    // Minimal vertical distance between a glint and its bottom neighbour.
    double min_glint_bottom_vert_distance{};
    // Maximal vertical distance between a glint and its bottom neighbour.
    double max_glint_bottom_vert_distance{};
    // Minimal horizontal distance between a glint and its right neighbour.
    double min_glint_right_hor_distance{};
    // Maximal horizontal distance between a glint and its right neighbour.
    double max_glint_right_hor_distance{};
    // Minimal vertical distance between a glint and its right neighbour.
    double min_glint_right_vert_distance{};
    // Maximal vertical distance between a glint and its right neighbour.
    double max_glint_right_vert_distance{};
    // Maximal horizontal distance between a glint and a pupil's centre.
    double max_hor_glint_pupil_distance{};
    // Maximal vertical distance between a glint and a pupil's centre.
    double max_vert_glint_pupil_distance{};
    // Centre of the circle aligned with the hole in the view piece in the image.
    cv::Point2d pupil_search_centre{};
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
    double exposure{};

    cv::Point3d position_offset{};

    cv::Point2d angle_offset{};

    std::vector<double> polynomial_x{};

    std::vector<double> polynomial_y{};

    std::vector<double> polynomial_theta{};

    std::vector<double> polynomial_phi{};
};


struct Coefficients {
    // Polynomial coefficients estimating x-axis position of the eye centre.
    std::vector<double> eye_centre_pos_x{};
    // Polynomial coefficients estimating y-axis position of the eye centre.
    std::vector<double> eye_centre_pos_y{};
    // Polynomial coefficients estimating z-axis position of the eye centre.
    std::vector<double> eye_centre_pos_z{};
    // Polynomial coefficients estimating x-axis direction of the gaze.
    std::vector<double> theta{};
    // Polynomial coefficients estimating y-axis direction of the gaze.
    std::vector<double> phi{};
};

struct EyeMeasurements {
    double cornea_centre_distance{};
    double cornea_curvature_radius{};
    double cornea_refraction_index{};
    double pupil_cornea_distance{};
    double alpha{};
    double beta{};
};

struct PolynomialParams {
    Coefficients coefficients{};
    EyeMeasurements eye_measurements{};
};

/**
 * All parameters required for proper eye-tracking.
 */
struct Parameters {
    // Internal camera parameters. One struct per eye.
    CameraParams camera_params[2]{};

    // Vector of LED positions. One per eye.
    std::vector<cv::Vec3d> leds_positions[2]{};

    // Parameters used to increase the eye features detection precision.
    // One struct per eye.
    DetectionParams detection_params[2]{};

    // Detection parameters calibrated individually per person. One set per user.
    std::unordered_map<std::string, FeaturesParams> features_params[2]{};
    // Detection parameters for the current user.
    FeaturesParams *user_params[2]{};

    // Default polynomial parameters.
    PolynomialParams polynomial_params[2]{};
};

/**
 * Loads settings from Json file and saves them back.
 */
class Settings {
public:
    /**
     * Loads settings from Json and stores in the parameters structure.
     * @param settings_folder Path to the json file.
     */
    explicit Settings(const std::string &settings_folder);

    static void loadSettings();
    /**
     * Save settings back to Json file form the parameters structure.
     * @param file_path Path to the saved json file.
     */
    static void saveSettings();
    // Structure with all parameters required for proper eye-tracking.
    static Parameters parameters;

    static std::filesystem::path settings_folder_;
};
} // namespace et

#endif //HDRMFS_EYE_TRACKER_SETTINGS_HPP
