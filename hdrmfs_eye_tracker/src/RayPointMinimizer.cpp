#include "RayPointMinimizer.hpp"
#include "EyeEstimator.hpp"
#include "Settings.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace et {

int RayPointMinimizer::getDims() const {
    return 2;
}

double RayPointMinimizer::calc(const double *x) const {
    cv::Vec3f c{np_ + np2c_dir_ * x[0]};
    double t{0};

    double error{0};

    // Finds the difference between reflecting and reflected LED vectors which
    // should be as low as possible.
    for (int i = 0; i < lp_.size(); i++) {
        bool intersected{EyeEstimator::getRaySphereIntersection(
            screen_glint_[i], ray_dir_[i], c,
            Settings::parameters.eye_params.cornea_curvature_radius, t)};
        if (intersected && t > 0) {
            cv::Vec3f pp{screen_glint_[i] + t * ray_dir_[i]};
            cv::Vec3f vc{pp - c};
            cv::normalize(vc, vc);

            cv::Vec3f v1{np_ - pp};
            cv::normalize(v1, v1);

            cv::Vec3f v2{lp_[i] - pp};
            cv::normalize(v2, v2);

            double alf1{std::acos(v1.dot(vc))};
            double alf2{std::acos(v2.dot(vc))};
            error += std::abs(alf1 - alf2);

        } else {
            error += 1e5;
        }
    }

    return error;
}

void RayPointMinimizer::setParameters(const cv::Vec3f &np2c_dir,
                                      cv::Vec3f *screen_glint,
                                      std::vector<cv::Vec3f> &lp) {
    if (!initialized_) {
        std::cerr << "RayPointMinimizer not initialized! Run initialize() "
                     "method first.\n";
        return;
    }
    np2c_dir_ = np2c_dir;
    for (int i = 0; i < lp.size(); i++) {
        screen_glint_[i] = screen_glint[i];
        lp_[i] = lp[i];
        ray_dir_[i] = np_ - screen_glint_[i];
        cv::normalize(ray_dir_[i], ray_dir_[i]);
    }
}

void RayPointMinimizer::initialize() {
    np_ = cv::Vec3f(0.0);
    unsigned long n_leds = Settings::parameters.leds_positions[0].size();
    screen_glint_.resize(n_leds);
    lp_.resize(n_leds);
    ray_dir_.resize(n_leds);
    initialized_ = true;
}
}// namespace et