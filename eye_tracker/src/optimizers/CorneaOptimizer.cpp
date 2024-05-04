#include <eye_tracker/Utils.hpp>
#include <eye_tracker/optimizers/CorneaOptimizer.hpp>

namespace et {
    void CorneaOptimizer::setParameters(const EyeMeasurements& eye_measurements, const cv::Vec3d& eye_centre,
        const cv::Vec3d& focus_point)
    {
        eye_measurements_ = eye_measurements;
        eye_centre_ = eye_centre;
        focus_point_ = focus_point;
    }

    int CorneaOptimizer::getDims() const
    {
        return 2;
    }

    double CorneaOptimizer::calc(const double* x) const
    {
        double theta = x[0];
        double phi = x[1];
        cv::Vec3d optical_axis{};
        cv::Vec3d visual_axis{};
        Utils::anglesToVector({theta, phi}, optical_axis);
        visual_axis = Utils::opticalToVisualAxis(optical_axis, eye_measurements_.alpha, eye_measurements_.beta);

        cv::Vec3d cornea = eye_centre_ + eye_measurements_.cornea_centre_distance * optical_axis;
        double k = cv::norm(focus_point_ - cornea);

        cv::Vec3d expected_focus_point = cornea + k * visual_axis;
        return cv::norm(expected_focus_point - focus_point_);
    }

    // void CorneaOptimizer::getGradient(const double* x, double* grad)
    // {
    //
    // }
} // et