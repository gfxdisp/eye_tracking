#ifndef EYE_TRACKER_METAMODEL_HPP
#define EYE_TRACKER_METAMODEL_HPP

#include "eye_tracker/image/FeatureAnalyser.hpp"
#include "eye_tracker/optimizers/VisualAnglesOptimizer.hpp"
#include "eye_tracker/optimizers/PolynomialFit.hpp"
#include "eye_tracker/optimizers/OpticalAxisOptimizer.hpp"
#include "eye_tracker/optimizers/EyeAnglesOptimizer.hpp"
#include "CorneaOptimizer.hpp"

#include <memory>

namespace et
{
    struct CalibrationData
    {
        cv::Point3d eye_position;
        std::vector<cv::Point3d> marker_positions;
        std::vector<cv::RotatedRect> glint_ellipses;
        std::vector<cv::Point2d> pupil_positions;
        std::vector<double> timestamps;
        std::vector<cv::Point2d> top_left_glints;
        std::vector<cv::Point2d> bottom_right_glints;
    };

    class MetaModel
    {
    public:
        MetaModel(int camera_id);

        void
        findMetaModel(const CalibrationData& calibration_data);

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


        int camera_id_{};
    };
} // et

#endif //EYE_TRACKER_METAMODEL_HPP
