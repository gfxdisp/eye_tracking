#include "eye_tracker/optimizers/GlintCircleOptimizer.hpp"

namespace et
{
    int GlintCircleOptimizer::getDims() const
    {
        return 3;
    }

    double GlintCircleOptimizer::calc(const double *x) const
    {
        cv::Point2d centre{x[0], x[1]};
        double radius{x[2]};
        double temp_value;
        double glint_value{0.0};
        // Error equal on the distance between the previous circle and glints
        for (auto glint: glints_)
        {
            temp_value = 0.0;
            temp_value += (glint.x - centre.x) * (glint.x - centre.x);
            temp_value += (glint.y - centre.y) * (glint.y - centre.y);
            temp_value -= radius * radius;
            glint_value += temp_value * temp_value;
        }
        glint_value /= glints_sigma_ * (int) glints_.size();

        // Error equal on the distance between the previous circle's centre
        // and the centre of the circle based on glints.
        temp_value = 0.0;
        temp_value += (previous_centre_.x - centre.x) * (previous_centre_.x - centre.x);
        temp_value += (previous_centre_.y - centre.y) * (previous_centre_.y - centre.y);
        double centre_value = temp_value / centre_sigma_;

        // Error equal on the difference between the previous circle's radius
        // and the radius of the circle based on glints.
        temp_value = 0.0;
        temp_value += (previous_radius_ - radius) * (previous_radius_ - radius);
        double radius_value = temp_value / radius_sigma_;
        double total_value = glint_value + centre_value + radius_value;
        return total_value;
    }

    void GlintCircleOptimizer::setParameters(const std::vector<cv::Point2d> &glints, const cv::Point2d &previous_centre,
                                             double previous_radius)
    {
        glints_ = glints;
        previous_centre_ = previous_centre;
        previous_radius_ = previous_radius;
    }
} // namespace et