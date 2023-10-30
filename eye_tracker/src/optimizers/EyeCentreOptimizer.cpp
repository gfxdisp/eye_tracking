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
        float error = 0.0f;
        for (int i = 0; i < visual_axes_.size(); i++)
        {
            cv::Vec3f nodal_point = cv::Vec3f(x[0] + optical_axes_[i][0] * cornea_centre_distance_,
                                              x[1] + optical_axes_[i][1] * cornea_centre_distance_,
                                              x[2] + optical_axes_[i][2] * cornea_centre_distance_);

            float distance = Utils::pointToLineDistance(nodal_point, visual_axes_[i], cross_point_);
            error += distance;
        }
        return error;
    }

    void
    EyeCentreOptimizer::setParameters(std::vector<cv::Point3f> front_corners, std::vector<cv::Point3f> back_corners,
                                      float alpha, float beta, float cornea_centre_distance)
    {
        visual_axes_.clear();
        optical_axes_.clear();

        int n = front_corners.size();

        cv::Mat S = cv::Mat::zeros(3, 3, CV_32F);
        cv::Mat C = cv::Mat::zeros(3, 1, CV_32F);
        cv::Mat directions = cv::Mat::zeros(n, 3, CV_32F);
        cv::Mat origins = cv::Mat::zeros(n, 3, CV_32F);

        for (int i = 0; i < n; i++)
        {
            cv::Vec3f visual_axis = {back_corners[i].x - front_corners[i].x, back_corners[i].y - front_corners[i].y,
                                     back_corners[i].z - front_corners[i].z};
            float norm = std::sqrt(visual_axis[0] * visual_axis[0] + visual_axis[1] * visual_axis[1] +
                                   visual_axis[2] * visual_axis[2]);
            visual_axis[0] /= norm;
            visual_axis[1] /= norm;
            visual_axis[2] /= norm;
            visual_axes_.push_back(visual_axis);
            optical_axes_.push_back(Utils::visualToOpticalAxis(visual_axis, alpha, beta));
            directions.at<float>(i, 0) = visual_axes_[i][0];
            directions.at<float>(i, 1) = visual_axes_[i][1];
            directions.at<float>(i, 2) = visual_axes_[i][2];

            origins.at<float>(i, 0) = back_corners[i].x;
            origins.at<float>(i, 1) = back_corners[i].y;
            origins.at<float>(i, 2) = back_corners[i].z;
        }
        cornea_centre_distance_ = cornea_centre_distance;

        cv::Mat eye = cv::Mat::eye(3, 3, CV_32F);

        for (int i = 0; i < 4; i++)
        {
            S += eye - directions.row(i).t() * directions.row(i);
            C += (eye - directions.row(i).t() * directions.row(i)) * origins.row(i).t();
        }

        cv::Mat intersection = S.inv(cv::DECOMP_SVD) * C;
        cross_point_ = {intersection.at<float>(0, 0), intersection.at<float>(1, 0), intersection.at<float>(2, 0)};
    }

    cv::Vec3f EyeCentreOptimizer::getCrossPoint() const
    {
        return cross_point_;
    }
} // et