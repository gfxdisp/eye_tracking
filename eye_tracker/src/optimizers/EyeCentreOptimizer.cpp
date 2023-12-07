#include "eye_tracker/optimizers/EyeCentreOptimizer.hpp"
#include "eye_tracker/Utils.hpp"

namespace et
{
    void EyeCentreOptimizer::setParameters(cv::Point3d cross_point, std::vector<cv::Vec3d> visual_axes,
                                           std::vector<cv::Vec3d> optical_axes, double cornea_centre_distance)
    {
        cross_point_ = cross_point;
        visual_axes_ = visual_axes;
        optical_axes_ = optical_axes;
        cornea_centre_distance_ = cornea_centre_distance;
    }

    int EyeCentreOptimizer::getDims() const
    {
        return 3;
    }

    double EyeCentreOptimizer::calc(const double *x) const
    {
        double error = 0.0;
        for (int i = 0; i < visual_axes_.size(); i++)
        {
            cv::Vec3d nodal_point = cv::Vec3d(x[0] + optical_axes_[i][0] * cornea_centre_distance_,
                                              x[1] + optical_axes_[i][1] * cornea_centre_distance_,
                                              x[2] + optical_axes_[i][2] * cornea_centre_distance_);

            double distance = Utils::pointToLineDistance(nodal_point, visual_axes_[i], cross_point_);
            error += distance;
        }
        return error;
    }
} // et