#include "eye_tracker/optimizers/EyeCentreOptimizer.hpp"
#include "eye_tracker/Utils.hpp"

namespace et
{
    int EyeCentreOptimizer::getDims() const
    {
        return 3;
    }

    double EyeCentreOptimizer::calc(const double *x) const
    {
        double error = 0.0f;
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

    void
    EyeCentreOptimizer::setParameters(std::vector<cv::Point3d> front_corners, std::vector<cv::Point3d> back_corners,
                                      double alpha, double beta, double cornea_centre_distance)
    {
        visual_axes_.clear();
        optical_axes_.clear();

        int n = front_corners.size();

        cv::Mat S = cv::Mat::zeros(3, 3, CV_64F);
        cv::Mat C = cv::Mat::zeros(3, 1, CV_64F);
        cv::Mat directions = cv::Mat::zeros(n, 3, CV_64F);
        cv::Mat origins = cv::Mat::zeros(n, 3, CV_64F);

        for (int i = 0; i < n; i++)
        {
            cv::Vec3d visual_axis = {back_corners[i].x - front_corners[i].x, back_corners[i].y - front_corners[i].y,
                                     back_corners[i].z - front_corners[i].z};
            double norm = std::sqrt(visual_axis[0] * visual_axis[0] + visual_axis[1] * visual_axis[1] +
                                   visual_axis[2] * visual_axis[2]);
            visual_axis[0] /= norm;
            visual_axis[1] /= norm;
            visual_axis[2] /= norm;
            visual_axes_.push_back(visual_axis);
            cv::Vec3d optical_axis = Utils::visualToOpticalAxis(visual_axis, alpha, beta);
            optical_axes_.push_back(optical_axis);
            directions.at<double>(i, 0) = visual_axes_[i][0];
            directions.at<double>(i, 1) = visual_axes_[i][1];
            directions.at<double>(i, 2) = visual_axes_[i][2];

            origins.at<double>(i, 0) = back_corners[i].x;
            origins.at<double>(i, 1) = back_corners[i].y;
            origins.at<double>(i, 2) = back_corners[i].z;
        }
        cornea_centre_distance_ = cornea_centre_distance;

        cv::Mat eye = cv::Mat::eye(3, 3, CV_64F);

        for (int i = 0; i < 4; i++)
        {
            S += eye - directions.row(i).t() * directions.row(i);
            C += (eye - directions.row(i).t() * directions.row(i)) * origins.row(i).t();
        }

        cv::Mat intersection = S.inv(cv::DECOMP_SVD) * C;
        cross_point_ = {intersection.at<double>(0, 0), intersection.at<double>(1, 0), intersection.at<double>(2, 0)};
    }

    cv::Vec3d EyeCentreOptimizer::getCrossPoint() const
    {
        return cross_point_;
    }
} // et