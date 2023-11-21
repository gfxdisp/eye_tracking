#include "eye_tracker/optimizers/PixelPosOptimizerTest.hpp"

namespace et
{
    void PixelPosOptimizerTest::setParameters(std::shared_ptr<ModelEyeEstimator> model_eye_estimator,
                                              cv::Point3d nodal_point, cv::Point3d eye_centre)
    {
        model_eye_estimator_ = model_eye_estimator;
        nodal_point_ = nodal_point;
        eye_centre_ = eye_centre;
    }

    int PixelPosOptimizerTest::getDims() const
    {
        return 6;
    }

    double PixelPosOptimizerTest::calc(const double *x) const
    {
        EyeInfo eye_info;
        eye_info.pupil = cv::Point2d{x[0], x[1]};
        eye_info.glints = {cv::Point2d{x[2], x[3]}, cv::Point2d{x[4], x[5]}};

        cv::Point3d nodal_point, eye_centre, visual_axis;
        bool success = model_eye_estimator_->detectEye(eye_info, nodal_point, eye_centre, visual_axis);
        if (!success)
        {
            return 1000000;
        }

        auto optical_axis_ = nodal_point_ - eye_centre_;
        optical_axis_ = optical_axis_ / cv::norm(optical_axis_);
        auto calc_optical_axis = nodal_point - eye_centre;
        calc_optical_axis = calc_optical_axis / cv::norm(calc_optical_axis);

        double angle = std::acos(optical_axis_.dot(calc_optical_axis));

        double error = 0.0;
        error += std::pow(eye_centre_.x - eye_centre.x, 2);
        error += std::pow(eye_centre_.y - eye_centre.y, 2);
        error += std::pow(eye_centre_.z - eye_centre.z, 2);

//        error += std::pow(nodal_point_.x - nodal_point.x, 2);
//        error += std::pow(nodal_point_.y - nodal_point.y, 2);
//        error += std::pow(nodal_point_.z - nodal_point.z, 2);

//        error += std::pow(angle, 2);


        return error;
    }
} // et