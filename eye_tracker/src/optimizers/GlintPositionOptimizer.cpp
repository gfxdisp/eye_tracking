#include "eye_tracker/optimizers/GlintPositionOptimizer.hpp"

namespace et
{
    void GlintPositionOptimizer::setParameters(cv::Point3d nodal_point, double cornea_radius, cv::Point3d led_position,
                                               cv::Point3d camera_position)
    {
        nodal_point_ = nodal_point;
        cornea_radius_ = cornea_radius;
        led_position_ = led_position;
        camera_position_ = camera_position;
    }

    double GlintPositionOptimizer::calc(const double *x) const
    {

        cv::Vec3d v1 = led_position_ - nodal_point_;
        cv::normalize(v1, v1);
        cv::Vec3d v2 = camera_position_ - nodal_point_;
        cv::normalize(v2, v2);
        cv::Vec3d v3 = v1 + (v2 - v1) * tanh(x[0]);
        cv::normalize(v3, v3);

        cv::Vec3d glint_position = static_cast<cv::Vec3d>(nodal_point_) + v3 * cornea_radius_;

        cv::Vec3d o1 = static_cast<cv::Vec3d>(led_position_) - glint_position;
        cv::normalize(o1, o1);
        cv::Vec3d o2 = static_cast<cv::Vec3d>(camera_position_) - glint_position;
        cv::normalize(o2, o2);
        cv::Vec3d oo = glint_position - static_cast<cv::Vec3d>(nodal_point_);
        cv::normalize(oo, oo);

        double alf1 = std::acos(o1.dot(oo)) * 180 / CV_PI;
        double alf2 = std::acos(o2.dot(oo)) * 180 / CV_PI;
        double error = std::abs(alf1 - alf2);

        return error;
    }

    int GlintPositionOptimizer::getDims() const
    {
        return 2;
    }
} // et