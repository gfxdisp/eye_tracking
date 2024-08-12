#ifndef HDRMFS_EYE_TRACKER_SETTINGS_HPP
#define HDRMFS_EYE_TRACKER_SETTINGS_HPP

#include "eye_tracker/json.hpp"

#include <opencv2/opencv.hpp>

#include <string>
#include <unordered_map>
#include <vector>
#include <filesystem>

namespace et {
    struct CameraParams {
        std::string serial_number{};

        cv::Size2i dimensions{};

        cv::Size2i region_of_interest{};

        cv::Size2i capture_offset{};

        double framerate{};

        double gamma{};

        double pixel_clock{};

        cv::Mat intrinsic_matrix{};

        cv::Mat extrinsic_matrix{};

        std::vector<double> distortion_coefficients{};

        cv::Vec3d gaze_shift{};
    };

    struct DetectionParams {
        double min_pupil_radius{};

        double max_pupil_radius{};

        double min_glint_radius{};

        double max_glint_radius{};

        double min_glint_bottom_hor_distance{};

        double max_glint_bottom_hor_distance{};

        double min_glint_bottom_vert_distance{};

        double max_glint_bottom_vert_distance{};

        double min_glint_right_hor_distance{};

        double max_glint_right_hor_distance{};

        double min_glint_right_vert_distance{};

        double max_glint_right_vert_distance{};

        double max_hor_glint_pupil_distance{};

        double max_vert_glint_pupil_distance{};

        cv::Point2d pupil_search_centre{};

        int pupil_search_radius{};
    };

    struct FeaturesParams {
        int pupil_threshold{};

        int glint_threshold{};

        double exposure{};

        cv::Point3d position_offset{};

        std::vector<double> polynomial_theta{};

        std::vector<double> polynomial_phi{};

        double marker_depth{};
    };

    struct EyeParams {
        double cornea_centre_distance{};
        double cornea_curvature_radius{};
        double cornea_refraction_index{};
        double pupil_cornea_distance{};
        double alpha{};
        double beta{};
    };

    struct Parameters {
        CameraParams camera_params[2]{};


        std::vector<cv::Vec3d> leds_positions[2]{};

        DetectionParams detection_params[2]{};

        std::unordered_map<std::string, FeaturesParams> features_params[2]{};

        FeaturesParams* user_params[2]{};

        EyeParams eye_params[2]{};
    };

    class Settings {
    public:
        explicit Settings(const std::string& settings_folder);

        static void loadSettings();

        static void saveSettings();

        static Parameters parameters;

        static std::filesystem::path settings_folder_;
    };
} // namespace et

#endif //HDRMFS_EYE_TRACKER_SETTINGS_HPP
