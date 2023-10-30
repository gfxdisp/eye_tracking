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
        ModelEyeEstimator(int camera_id, cv::Point3f eye_position);
        ~ModelEyeEstimator();

        bool detectEye(EyeInfo &eye_info, cv::Point3f &nodal_point, cv::Point3f &eye_centre, cv::Point3f &visual_axis) override;

    protected:
        // Function used to optimize cornea centre position in
        // the getEyeFromModel() method.
        NodalPointOptimizer *nodal_point_optimizer_{};
        // Pointer to a NodalPointOptimizer function.
        cv::Ptr<cv::DownhillSolver::Function> minimizer_function_{};
        // Downhill solver optimizer used to find cornea centre position.
        cv::Ptr<cv::DownhillSolver> solver_{};

        float pupil_eye_centre_distance_{};
    };

} // et

#endif //HDRMFS_EYE_TRACKER_MODELEYEESTIMATOR_HPP
