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
        cv::Vec3d optical_axis;
        optical_axis[0] = -sin(theta) * cos(phi);
        optical_axis[1] = sin(phi);
        optical_axis[2] = -cos(theta) * cos(phi);
        cv::Vec3d up_vector;
        up_vector[0] = sin(theta) * sin(phi);
        up_vector[1] = cos(phi);
        up_vector[2] = cos(theta) * sin(phi);
        cv::Vec3d right_vector = optical_axis.cross(up_vector);

        cv::Vec3d cornea = eye_centre_ + eye_measurements_.cornea_centre_distance * optical_axis;

        double k = cv::norm(focus_point_ - cornea);

        cv::Mat M1 = Utils::convertAxisAngleToRotationMatrix(up_vector, -eye_measurements_.alpha * CV_PI / 180.0);
        cv::Mat M2 = Utils::convertAxisAngleToRotationMatrix(M1 * right_vector, -eye_measurements_.beta * CV_PI / 180.0);
        cv::Mat visual_axis_mat = M2 * M1 * optical_axis;
        cv::Vec3d visual_axis = cv::Vec3d(visual_axis_mat.at<double>(0, 0), visual_axis_mat.at<double>(1, 0), visual_axis_mat.at<double>(2, 0));

        cv::Vec3d expected_focus_point = cornea + k * visual_axis;
        return cv::norm(expected_focus_point - focus_point_);
    }

    // void CorneaOptimizer::getGradient(const double* x, double* grad)
    // {
    //
    // }
} // et