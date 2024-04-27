#ifndef HDRMFS_EYE_TRACKER_MODELEYEESTIMATOR_HPP
#define HDRMFS_EYE_TRACKER_MODELEYEESTIMATOR_HPP

#include "EyeEstimator.hpp"
#include "eye_tracker/optimizers/NodalPointOptimizer.hpp"

#include <opencv2/core/optim.hpp>

namespace et
{
    class ModelEyeEstimator : public EyeEstimator
    {
    public:
        ModelEyeEstimator(int camera_id);
        ~ModelEyeEstimator();

        bool detectEye(EyeInfo& eye_info, cv::Point3d& eye_centre, cv::Point3d& nodal_point, cv::Vec2d& angles) override;

        bool invertDetectEye(EyeInfo& eye_info, const cv::Point3d& nodal_point, const cv::Point3d& eye_centre,
                             const EyeMeasurements& measurements);

    protected:
        // Function used to optimize cornea centre position in
        // the getEyeFromModel() method.
        NodalPointOptimizer* nodal_point_optimizer_{};
        // Pointer to a NodalPointOptimizer function.
        cv::Ptr<cv::DownhillSolver::Function> minimizer_function_{};
        // Downhill solver optimizer used to find cornea centre position.
        cv::Ptr<cv::DownhillSolver> solver_{};
    };
} // et

#endif //HDRMFS_EYE_TRACKER_MODELEYEESTIMATOR_HPP
