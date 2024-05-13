#ifndef EYE_TRACKER_METAMODEL_HPP
#define EYE_TRACKER_METAMODEL_HPP

#include "eye_tracker/image/FeatureAnalyser.hpp"
#include "eye_tracker/optimizers/VisualAnglesOptimizer.hpp"
#include "eye_tracker/optimizers/PolynomialFit.hpp"
#include "eye_tracker/optimizers/OpticalAxisOptimizer.hpp"
#include "eye_tracker/optimizers/EyeAnglesOptimizer.hpp"
#include "CorneaOptimizer.hpp"
#include "eye_tracker/eye/ModelEyeEstimator.hpp"
#include "eye_tracker/eye/PolynomialEyeEstimator.hpp"

#include <memory>

namespace et {
    struct CalibrationSample {
        cv::Point3d eye_position;
        bool detected;
        int marker_id;
        cv::Point3d marker_position;
        cv::RotatedRect glint_ellipse;
        cv::Point2d pupil_position;
        double timestamp;
        double marker_time;
        std::vector<cv::Point2d> glints;
        std::vector<bool> glints_validity;
    };

    struct CalibrationInput {
        cv::Point3d eye_position;
        cv::Point3d cornea_position;
        cv::Vec2d angles;
        cv::Vec2d pcr_distance;
        double timestamp;
        bool detected;
    };

    struct CalibrationOutput {
        cv::Point3d eye_position;
        std::vector<cv::Point3d> marker_positions;
        std::vector<double> timestamps;
    };

    struct MetaModelData {
        cv::Point3d real_eye_position;
        std::vector<cv::Point3d> real_marker_positions;

        std::vector<cv::Point3d> estimated_eye_positions;

        std::vector<cv::Point3d> real_cornea_positions;
        std::vector<cv::Point3d> estimated_cornea_positions;

        std::vector<cv::Point2d> angle_offsets;

        std::vector<double> real_angles_theta;
        std::vector<double> real_angles_phi;
        std::vector<double> estimated_angles_theta;
        std::vector<double> estimated_angles_phi;

        std::vector<double> pcr_distances_x;
        std::vector<double> pcr_distances_y;
    };

    class MetaModel {
    public:
        static bool ransac;

        static bool calibration_enabled;

        explicit MetaModel(int camera_id);

        std::vector<std::vector<double>> findMetaModel(std::vector<CalibrationInput> const& calibration_input, CalibrationOutput const& calibration_output, bool from_scratch, const std::string& output_video_name);

        std::vector<std::vector<double>> findMetaModelFromFile(const std::string& calibration_input_video_path, const std::string& calibration_output_csv_path, bool from_scratch);

        std::vector<std::vector<double>> getEstimationsAtFrames(const std::string& calibration_input_video_path, const std::string& frames_csv);

    private:
        CorneaOptimizer* cornea_optimizer_{};
        cv::Ptr<cv::DownhillSolver::Function> cornea_minimizer_function_{};
        cv::Ptr<cv::DownhillSolver> cornea_solver_{};

        std::shared_ptr<EyeEstimator> eye_estimator_{};

        int camera_id_{};
    };
} // et

#endif //EYE_TRACKER_METAMODEL_HPP
