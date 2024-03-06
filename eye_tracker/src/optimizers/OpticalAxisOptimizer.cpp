#include "../../include/eye_tracker/optimizers/OpticalAxisOptimizer.hpp"
#include "eye_tracker/Utils.hpp"

namespace et
{
    int OpticalAxisOptimizer::getDims() const
    {
        return 3;
    }

    double OpticalAxisOptimizer::calc(const double* x) const
    {
        cv::Point3d optical_axis = cv::Point3d(x[0], x[1], x[2]);
        optical_axis = optical_axis / cv::norm(optical_axis);
        cv::Point3d visual_axis = Utils::opticalToVisualAxis(optical_axis, eye_measurements_.alpha,
                                                             eye_measurements_.beta);

        cv::Point3d cornea_centre = eye_centre_ + eye_measurements_.cornea_centre_distance * optical_axis;
        double distance = Utils::pointToLineDistance(cornea_centre, visual_axis, focus_point_);
        return distance;
    }

    void
    OpticalAxisOptimizer::setParameters(const EyeMeasurements& eye_measurements, const cv::Point3d& eye_centre,
                                        const cv::Point3d& focus_point)
    {
        eye_measurements_ = eye_measurements;
        eye_centre_ = eye_centre;
        focus_point_ = focus_point;
    }
} // et
