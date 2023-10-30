#include "eye_tracker/optimizers/EyeAnglesOptimizer.hpp"
#include "eye_tracker/Utils.hpp"

namespace et
{
    void EyeAnglesOptimizer::setParameters(cv::Vec3f visual_axis, cv::Vec3f optical_axis)
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
        float alpha = x[0];
        float beta = x[1];
        cv::Vec3f expected_optical_axis = Utils::visualToOpticalAxis(visual_axis_, alpha, beta);
        // Get angle between expected and actual optical axis
        float angle = std::acos(expected_optical_axis.dot(optical_axis_));
        return angle;
    }
} // et