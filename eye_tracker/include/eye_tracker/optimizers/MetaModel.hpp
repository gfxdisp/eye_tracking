#ifndef EYE_TRACKER_METAMODEL_HPP
#define EYE_TRACKER_METAMODEL_HPP

#include "eye_tracker/image/FeatureAnalyser.hpp"
#include <eye_tracker/optimizers/OpticalAxisOptimizer.hpp>
#include <eye_tracker/eye/EyeEstimator.hpp>

#include <memory>

namespace et {
    struct CalibrationSample {
        cv::Point3d eye_position;
        bool detected;
        int marker_id;
        cv::Point3d marker_position;
        cv::RotatedRect glint_ellipse;
        cv::Point2d pupil_position;
        long int timestamp;
        long int marker_time;
        std::vector<cv::Point2d> glints;
        std::vector<bool> glints_validity;
    };

    struct CalibrationInput {
        cv::Point3d eye_position;
        cv::Point3d cornea_position;
        cv::Vec2d angles;
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
        std::vector<double> estimated_angles_theta;

        std::vector<double> real_angles_phi;
        std::vector<double> estimated_angles_phi;
    };

    class MetaModel {
    public:
        static bool ransac;

        explicit MetaModel(int camera_id);

        void findMetaModelOnline(std::vector<CalibrationInput> const& calibration_input, CalibrationOutput const& calibration_output) const;

        void findMetaModelOffline(const std::string& video_path) const;

    private:
        std::shared_ptr<OpticalAxisOptimizer> optical_axis_optimizer_{};
        cv::Ptr<cv::DownhillSolver::Function> cornea_minimizer_function_{};
        cv::Ptr<cv::DownhillSolver> cornea_solver_{};

        std::shared_ptr<EyeEstimator> eye_estimator_{};

        int camera_id_{};
    };
} // et

#endif //EYE_TRACKER_METAMODEL_HPP
