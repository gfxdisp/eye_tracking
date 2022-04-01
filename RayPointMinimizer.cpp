#include "RayPointMinimizer.hpp"

#include "EyeTracker.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace et {

double RayPointMinimizer::kk_{};
double RayPointMinimizer::lowest_error_{1e5};

RayPointMinimizer::RayPointMinimizer(const cv::Vec3d &np) : np_(np) {
}

int RayPointMinimizer::getDims() const {
    return 2;
}

double RayPointMinimizer::calc(const double *x) const {
    cv::Vec3d c{np_ + np2c_dir_ * x[0]};
    double t{0};

    double error{0};

    for (int i = 0; i < 2; i++) {
        bool intersected{
        EyeTracker::getRaySphereIntersection(screen_glint_[i], ray_dir_[i], c, EyeProperties::cornea_curvature_radius, t)};
        if (intersected && t > 0) {
            cv::Vec3d pp{screen_glint_[i] + t * ray_dir_[i]};
            cv::Vec3d vc{pp - c};
            cv::normalize(vc, vc);

            cv::Vec3d v1{np_ - pp};
            cv::normalize(v1, v1);

            cv::Vec3d v2{lp_[i] - pp};
            cv::normalize(v2, v2);

            double alf1{std::acos(v1.dot(vc))};
            double alf2{std::acos(v2.dot(vc))};
            error += std::abs(alf1 - alf2);
            
        } else {
            error += 1e5;
        }
    }

    if (error < lowest_error_) {
            lowest_error_ = error;
            kk_ = x[0];
        }

    return error;
}

void RayPointMinimizer::setParameters(const cv::Vec3d &np2c_dir, cv::Vec3d *screen_glint, cv::Vec3d *lp) {
    np2c_dir_ = np2c_dir;
    for (int i = 0; i < 2; i++) {
        screen_glint_[i] = screen_glint[i];
        lp_[i] = lp[i];
        ray_dir_[i] = np_ - screen_glint_[i];
        cv::normalize(ray_dir_[i], ray_dir_[i]);
    }

    lowest_error_ = 1e5;
}
}// namespace et