#include "eye_tracker/optimizers/PupilPositionOptimizer.hpp"
#include "eye_tracker/Utils.hpp"

namespace et
{
    void PupilPositionOptimizer::setParameters(cv::Point3d nodal_point, cv::Point3d pupil_center, cv::Point3d camera_position, double cornea_radius, double refraction_index)
    {
        nodal_point_ = nodal_point;
        pupil_center_ = pupil_center;
        camera_position_ = camera_position;
        cornea_radius_ = cornea_radius;
        refraction_index_ = refraction_index;
    }

    double PupilPositionOptimizer::calc(const double *x) const
    {
        cv::Point3d ray_position = camera_position_;
        cv::Vec3d ray_direction;
        cv::multiply(static_cast<cv::Vec3d>(pupil_center_ - camera_position_), cv::Vec3d(x[0], x[1], x[2]),
                     ray_direction);

        cv::normalize(ray_direction, ray_direction);

        double t{};
        double error;
        bool intersection = Utils::getRaySphereIntersection(ray_position, ray_direction, nodal_point_, cornea_radius_, t);
        if (intersection)
        {
            cv::Point3d intersection_point = static_cast<cv::Vec3d>(ray_position) + t * ray_direction;
            cv::Vec3d n = intersection_point - nodal_point_;
            cv::normalize(n, n);
            cv::Vec3d rr = Utils::getRefractedRay(ray_direction, n, refraction_index_);
            cv::normalize(rr, rr);

            cv::Vec3d vv = intersection_point - pupil_center_;
            cv::normalize(vv, vv);

            error = std::acos((-rr).dot(vv)) * 180 / CV_PI;
        }
        else
        {
            error = 1000000;
        }
        return error;
    }

    int PupilPositionOptimizer::getDims() const
    {
        return 3;
    }
} // et