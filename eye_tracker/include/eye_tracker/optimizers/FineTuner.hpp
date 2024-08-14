#ifndef HDRMFS_EYE_TRACKER_FINE_TUNER_HPP
#define HDRMFS_EYE_TRACKER_FINE_TUNER_HPP

#include <eye_tracker/image/FeatureAnalyser.hpp>
#include <eye_tracker/optimizers/OpticalAxisOptimizer.hpp>
#include <eye_tracker/eye/EyeEstimator.hpp>

#include <memory>

namespace et {
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

    struct FineTuningData {
        cv::Point3d real_eye_position;
        std::vector<cv::Point3d> real_marker_positions;

        std::vector<cv::Point3d> estimated_eye_positions;

        std::vector<cv::Point3d> real_cornea_positions;
        std::vector<cv::Point3d> estimated_cornea_positions;

        std::vector<double> real_angles_theta;
        std::vector<double> estimated_angles_theta;

        std::vector<double> real_angles_phi;
        std::vector<double> estimated_angles_phi;
    };

    /**
     * @brief The FineTuner class is responsible for fine tuning the eye estimation parameters based on the user's calibration.
     */
    class FineTuner {
    public:
        static bool ransac;

        explicit FineTuner(int camera_id);

        /**
         * @brief Calculates the fine tuning parameters based on the calibration data.
         * @param calibration_input Vector of calibration input data, typically captured after calling Framework::startCalibration.
         * @param calibration_output The output of the calibration process, containing the real eye positions, marker positions and timestamps.
         */
        void calculate(std::vector<CalibrationInput> const& calibration_input, CalibrationOutput const& calibration_output) const;

        /**
         * @brief Calculates the fine tuning parameters based on the calibration data.
         * @param video_path Path to the video file captured during the calibration process.
         * @param camera_csv_path Path to the CSV file containing the calibration data, specifically the real eye positions, marker positions, and timestamps.
         */
        void calculate(const std::string& video_path, const std::string& camera_csv_path) const;

    private:
        std::shared_ptr<OpticalAxisOptimizer> optical_axis_optimizer_{};
        cv::Ptr<cv::DownhillSolver::Function> cornea_minimizer_function_{};
        cv::Ptr<cv::DownhillSolver> cornea_solver_{};

        std::shared_ptr<EyeEstimator> eye_estimator_{};

        int camera_id_{};
    };
} // et

#endif // HDRMFS_EYE_TRACKER_FINE_TUNER_HPP
