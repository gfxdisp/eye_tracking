#include "eye_tracker/optimizers/PupilPositionOptimizer.hpp"
#include "eye_tracker/Utils.hpp"

namespace et
{
    void PupilPositionOptimizer::setParameters(const cv::Vec3d& nodal_point, const cv::Vec3d& pupil_center,
                                               const cv::Vec3d& camera_position,
                                               const EyeMeasurements& eye_measurements)
    {
        nodal_point_ = nodal_point;
        pupil_center_ = pupil_center;
        camera_position_ = camera_position;
        eye_measurements_ = eye_measurements;
    }

    /*double PupilPositionOptimizer::calc(const double* x) const
    {
        cv::Point3d ray_position = camera_position_;
        cv::Vec3d ray_direction;
        cv::multiply(static_cast<cv::Vec3d>(pupil_center_ - camera_position_), cv::Vec3d(x[0], x[1], x[2]),
                     ray_direction);

        cv::normalize(ray_direction, ray_direction);

        double t{};
        double error;
        bool intersection = Utils::getRaySphereIntersection(ray_position, ray_direction, nodal_point_,
                                                            eye_measurements_.cornea_curvature_radius, t);
        if (intersection)
        {
            cv::Vec3d intersection_point = static_cast<cv::Vec3d>(ray_position) + t * ray_direction;
            cv::Vec3d n = intersection_point - nodal_point_;
            cv::normalize(n, n);
            cv::Vec3d rr = Utils::getRefractedRay(ray_direction, n, eye_measurements_.cornea_refraction_index);
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
    }*/

    double PupilPositionOptimizer::calc(const double* x) const
    {
        std::string login_command = "curl.exe -X POST -H \"Content-Type: application/json\" -d \"{\\\"username\\\":\\\"admin\\\",\\\"password\\\":\\\"admin\\\"}\" http://localhost:8080/api/auth/login";


        double theta = x[0];
        double phi = x[1];

        cv::Vec3d pupil_on_cornea;
        pupil_on_cornea[0] = nodal_point_[0] - eye_measurements_.cornea_curvature_radius * sin(theta) * cos(phi);
        pupil_on_cornea[1] = nodal_point_[1] + eye_measurements_.cornea_curvature_radius * sin(phi);
        pupil_on_cornea[2] = nodal_point_[2] - eye_measurements_.cornea_curvature_radius * cos(theta) * cos(phi);

        cv::Vec3d ray_direction = pupil_on_cornea - camera_position_;
        cv::normalize(ray_direction, ray_direction);

        cv::Vec3d normal = pupil_on_cornea - nodal_point_;
        cv::normalize(normal, normal);

        cv::Vec3d refracted_ray = Utils::getRefractedRay(ray_direction, normal, eye_measurements_.cornea_refraction_index);
        cv::normalize(refracted_ray, refracted_ray);

        double k = cv::norm(pupil_center_ - pupil_on_cornea);
        cv::Vec3d expected_pupil_position = pupil_on_cornea + k * refracted_ray;
        return cv::norm(expected_pupil_position - pupil_center_);
    }

    int PupilPositionOptimizer::getDims() const
    {
        return 2;
    }
} // et
