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
        cv::Point3d optical_axis = cv::Point3d(x[0], x[1], x[2]);
        optical_axis = optical_axis / cv::norm(optical_axis);
        cv::Point3d visual_axis = Utils::opticalToVisualAxis(optical_axis, alpha_, beta_);

        cv::Point3d cornea_centre = eye_centre_ + cornea_centre_distance_ * optical_axis;
        double distance = Utils::pointToLineDistance(cornea_centre, visual_axis, focus_point_);
        return distance;
    }

    void
    OpticalAxisOptimizer::setParameters(double alpha, double beta, double cornea_centre_distance, cv::Point3d eye_centre,
                                        cv::Point3d focus_point)
    {
        alpha_ = alpha;
        beta_ = beta;
        cornea_centre_distance_ = cornea_centre_distance;
        eye_centre_ = eye_centre;
        focus_point_ = focus_point;
    }
} // et