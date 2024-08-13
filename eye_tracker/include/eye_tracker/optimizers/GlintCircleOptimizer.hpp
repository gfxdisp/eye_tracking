#ifndef HDRMFS_EYE_TRACKER_GLINT_CIRCLE_OPTIMIZER_HPP
#define HDRMFS_EYE_TRACKER_GLINT_CIRCLE_OPTIMIZER_HPP

#include <opencv2/core/optim.hpp>

namespace et {
    class GlintCircleOptimizer : public cv::DownhillSolver::Function {
    public:
        void setParameters(const std::vector<cv::Point2d>& glints, const cv::Point2d& previous_centre, double previous_radius);

    private:
        int getDims() const override;

        double calc(const double* x) const override;

        std::vector<cv::Point2d> glints_{};

        cv::Point2d previous_centre_{};
        double previous_radius_{};

        double glints_sigma_{2};
        double centre_sigma_{2};
        double radius_sigma_{5};
    };
} // namespace et

#endif //HDRMFS_EYE_TRACKER_GLINT_CIRCLE_OPTIMIZER_HPP
