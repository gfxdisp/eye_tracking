#include "BayesMinimizer.hpp"

namespace et {
BayesMinimizer::BayesMinimizer() {
    glints_sigma_ = 20.0f;
    centre_sigma_ = 10.0f;
    radius_sigma_ = 5.0f;
    glints_.resize(3);
}

int BayesMinimizer::getDims() const {
    return 3;
}

double BayesMinimizer::calc(const double *x) const {
    cv::Point2d centre{x[0], x[1]};
    double radius{x[2]};
    double total_value{0.0};
    double glint_value{0.0};
    double centre_value{0.0};
    double radius_value{0.0};
    double value{0.0};
    for (int i = 0; i < 3; i++) {
        value = 0.0;
        value += (glints_[i].x - centre.x) * (glints_[i].x - centre.x);
        value += (glints_[i].y - centre.y) * (glints_[i].y - centre.y);
        value -= radius * radius;
        glint_value += value * value;
    }
    glint_value /= glints_sigma_;

    value = 0.0;
    value += (previous_centre_.x - centre.x) * (previous_centre_.x - centre.x);
    value += (previous_centre_.y - centre.y) * (previous_centre_.y - centre.y);
    centre_value = value / centre_sigma_;

    value = 0.0;
    value += (previous_radius_ - radius) * (previous_radius_ - radius);
    radius_value = value / radius_sigma_;
    total_value = glint_value + centre_value + radius_value;
    return total_value;
}

void BayesMinimizer::setParameters(const std::vector<cv::Point2f> &glints,
                                   const cv::Point2d &previous_centre,
                                   double previous_radius) {
    for (int i = 0; i < 3; i++) {
        glints_[i] = glints[i];
    }
    previous_centre_ = previous_centre;
    previous_radius_ = previous_radius;
}
} // namespace et