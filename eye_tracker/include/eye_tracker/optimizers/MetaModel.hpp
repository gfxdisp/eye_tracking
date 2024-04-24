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

    class MetaModel {
    public:
        MetaModel(int camera_id);

        void findMetaModel(const std::vector<CalibrationSample>& calibration_data, bool from_scratch);

    private:
        VisualAnglesOptimizer* visual_angles_optimizer_{};
        cv::Ptr<cv::DownhillSolver::Function> visual_angles_minimizer_function_{};
        cv::Ptr<cv::DownhillSolver> visual_angles_solver_{};

        OpticalAxisOptimizer* optical_axis_optimizer_{};
        cv::Ptr<cv::DownhillSolver::Function> optical_axis_minimizer_function_{};
        cv::Ptr<cv::DownhillSolver> optical_axis_solver_{};

        EyeAnglesOptimizer* eye_angles_optimizer_{};
        cv::Ptr<cv::DownhillSolver::Function> eye_angles_minimizer_function_{};
        cv::Ptr<cv::DownhillSolver> eye_angles_solver_{};

        EyeCentreOptimizer* eye_centre_optimizer_{};
        cv::Ptr<cv::DownhillSolver::Function> eye_centre_minimizer_function_{};
        cv::Ptr<cv::DownhillSolver> eye_centre_solver_{};

        CorneaOptimizer* cornea_optimizer_{};
        cv::Ptr<cv::DownhillSolver::Function> cornea_minimizer_function_{};
        cv::Ptr<cv::DownhillSolver> cornea_solver_{};

        std::shared_ptr<EyeEstimator> eye_estimator_{};

        int camera_id_{};
    };
} // et

#endif //EYE_TRACKER_METAMODEL_HPP
