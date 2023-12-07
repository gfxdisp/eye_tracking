#ifndef EYE_TRACKER_PIXELPOSOPTIMIZER_HPP
#define EYE_TRACKER_PIXELPOSOPTIMIZER_HPP

#include <opencv2/opencv.hpp>
#include "eye_tracker/eye/ModelEyeEstimator.hpp"
#include "eye_tracker/optimizers/VisualAnglesOptimizer.hpp"
#include "eye_tracker/optimizers/EyeAnglesOptimizer.hpp"

namespace et
{

    class PixelPosOptimizer : public cv::DownhillSolver::Function
    {
    public:
        void
        setParameters(std::shared_ptr<ModelEyeEstimator> model_eye_estimator, std::vector<cv::Point3d> front_corners,
                      std::vector<cv::Point3d> back_corners, VisualAnglesOptimizer *eye_centre_optimizer,
                      cv::Ptr<cv::DownhillSolver> eye_centre_solver, EyeAnglesOptimizer *eye_angles_optimizer,
                      cv::Ptr<cv::DownhillSolver> eye_angles_solver, cv::Point3d marker_pos);

    private:
        int getDims() const override;

        double calc(const double *x) const override;

        std::shared_ptr<ModelEyeEstimator> model_eye_estimator_{};
        std::vector<cv::Point3d> front_corners_{};
        std::vector<cv::Point3d> back_corners_{};

        VisualAnglesOptimizer *eye_centre_optimizer_{};
        cv::Ptr<cv::DownhillSolver> eye_centre_solver_{};

        EyeAnglesOptimizer *eye_angles_optimizer_{};
        cv::Ptr<cv::DownhillSolver> eye_angles_solver_{};
        cv::Point3d marker_pos_{};
    };

} // et

#endif //EYE_TRACKER_PIXELPOSOPTIMIZER_HPP
