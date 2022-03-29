#include "RayPointMinimizer.hpp"

#include "EyeTracker.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace et {

cv::Vec3d RayPointMinimizer::pp_{cv::Vec3d()};
double RayPointMinimizer::kk_{};
double RayPointMinimizer::lowest_error_{1e5};

RayPointMinimizer::RayPointMinimizer(const cv::Vec3d &np) : np_(np) {
}

int et::RayPointMinimizer::getDims() const {
    return 2;
}

double et::RayPointMinimizer::calc(const double *x) const {
    cv::Vec3d c{np_ + np2c_dir_ * x[0]};
    double t{0};

    double error{1e5};

    bool intersected{
        EyeTracker::getRaySphereIntersection(screen_glint_, ray_dir_, c, EyeProperties::cornea_curvature_radius, t)};
    if (intersected && t > 0) {
        cv::Vec3d pp{screen_glint_ + t * ray_dir_};
        cv::Vec3d vc{pp - c};
        cv::normalize(vc, vc);

        cv::Vec3d v1{np_ - pp};
        cv::normalize(v1, v1);

        cv::Vec3d v2{lp_ - pp};
        cv::normalize(v2, v2);

        double alf1{std::acos(v1.dot(vc))};
        double alf2{std::acos(v2.dot(vc))};
        error = std::abs(alf1 - alf2);
        if (error < lowest_error_) {
            lowest_error_ = error;
            pp_ = pp;
            kk_ = x[0];
        }
    }

    return error;
}

void RayPointMinimizer::setParameters(const cv::Vec3d &np2c_dir, const cv::Vec3d &screen_glint, const cv::Vec3d &lp) {
    np2c_dir_ = np2c_dir;
    screen_glint_ = screen_glint;
    lp_ = lp;
    ray_dir_ = np_ - screen_glint_;
    cv::normalize(ray_dir_, ray_dir_);
    lowest_error_ = 1e5;
}
}// namespace et