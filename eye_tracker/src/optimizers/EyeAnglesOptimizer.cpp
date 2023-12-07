#include "eye_tracker/optimizers/EyeAnglesOptimizer.hpp"
#include "eye_tracker/Utils.hpp"

namespace et
{
    void EyeAnglesOptimizer::setParameters(cv::Vec3d visual_axis, cv::Vec3d optical_axis)
    {
        visual_axis_ = visual_axis;
        optical_axis_ = optical_axis;
    }

    int EyeAnglesOptimizer::getDims() const
    {
        return 2;
    }

    double EyeAnglesOptimizer::calc(const double *x) const
    {
        double alpha = x[0];
        double beta = x[1];
        cv::Vec3d expected_visual_axis = Utils::opticalToVisualAxis(optical_axis_, alpha, beta);
        // Get angle between expected and actual optical axis
        double angle = std::acos(expected_visual_axis.dot(visual_axis_));
        return angle;
    }
} // et