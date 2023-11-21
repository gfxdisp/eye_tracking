#include "eye_tracker/optimizers/OpticalFromVisualAxisOptimizer.hpp"
#include "eye_tracker/Utils.hpp"

namespace et
{
    void OpticalFromVisualAxisOptimizer::setParameters(double alpha, double beta, cv::Point3d visual_axis)
    {
        alpha_ = alpha;
        beta_ = beta;
        visual_axis_ = visual_axis;
    }

    int OpticalFromVisualAxisOptimizer::getDims() const
    {
        return 2;
    }

    double OpticalFromVisualAxisOptimizer::calc(const double *x) const
    {
        double phi = x[0];
        double theta = x[1];

        cv::Mat R = Utils::getRotY(theta) * Utils::getRotX(phi);

        cv::Mat optical_axis_mat = R * (cv::Mat_<double>(3, 1) << 0, 0, -1);
        cv::Point3d optical_axis = cv::Point3d(optical_axis_mat.at<double>(0, 0), optical_axis_mat.at<double>(1, 0),
                                               optical_axis_mat.at<double>(2, 0));
        optical_axis = optical_axis / cv::norm(optical_axis);
        cv::Vec3d expected_visual_axis = Utils::opticalToVisualAxis(optical_axis, alpha_, beta_);

        double error = 0.0f;
        for (int i = 0; i < 3; i++) {
            error += std::pow(expected_visual_axis[i] - visual_axis_[i], 2);
        }
        return error;
    }
} // et