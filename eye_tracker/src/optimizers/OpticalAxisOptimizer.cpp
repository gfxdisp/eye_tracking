#include "../../include/eye_tracker/optimizers/OpticalAxisOptimizer.hpp"
#include "eye_tracker/Utils.hpp"

namespace et
{
    int OpticalAxisOptimizer::getDims() const
    {
        return 3;
    }

    double OpticalAxisOptimizer::calc(const double *x) const
    {
        cv::Point3f optical_axis = cv::Point3f(x[0], x[1], x[2]);
        optical_axis = optical_axis / cv::norm(optical_axis);
        cv::Point3f visual_axis = Utils::opticalToVisualAxis(optical_axis, alpha_, beta_);

        cv::Point3f cornea_centre = eye_centre_ + cornea_centre_distance_ * optical_axis;
        float distance = Utils::pointToLineDistance(cornea_centre, visual_axis, focus_point_);
        return distance;
    }

    void
    OpticalAxisOptimizer::setParameters(float alpha, float beta, float cornea_centre_distance, cv::Point3f eye_centre,
                                        cv::Point3f focus_point)
    {
        alpha_ = alpha;
        beta_ = beta;
        cornea_centre_distance_ = cornea_centre_distance;
        eye_centre_ = eye_centre;
        focus_point_ = focus_point;
    }
} // et