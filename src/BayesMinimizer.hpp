#ifndef HDRMFS_EYE_TRACKER_BAYESMINIMIZER_HPP
#define HDRMFS_EYE_TRACKER_BAYESMINIMIZER_HPP

#include <opencv2/core/optim.hpp>

namespace et {

class BayesMinimizer : public cv::DownhillSolver::Function {
public:
    BayesMinimizer();
    [[nodiscard]] int getDims() const override;
    double calc(const double *x) const override;

    void setParameters(const std::vector<cv::Point2f> &glints,
                       const cv::Point2d &previous_centre,
                       double previous_radius);

private:
    std::vector<cv::Point2f> glints_{};
    double glints_sigma_{};
    cv::Point2d previous_centre_{};
    double centre_sigma_{};
    double previous_radius_{};
    double radius_sigma_{};
};

} // namespace et

#endif //HDRMFS_EYE_TRACKER_BAYESMINIMIZER_HPP
