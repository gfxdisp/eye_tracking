#ifndef EYE_TRACKER__RAYPOINTMINIMIZER_HPP_
#define EYE_TRACKER__RAYPOINTMINIMIZER_HPP_

#include <opencv2/opencv.hpp>

namespace et {
class RayPointMinimizer : public cv::MinProblemSolver::Function {
public:
    explicit RayPointMinimizer(const cv::Vec3d &np);
    [[nodiscard]] int getDims() const override;
    double calc(const double *x) const override;

    void setParameters(const cv::Vec3d &np2c_dir, const cv::Vec3d &screen_glint, const cv::Vec3d &lp);

    static cv::Vec3d pp_;
    static double kk_;
    static double lowest_error_;

private:
    cv::Vec3d np_{};
    cv::Vec3d np2c_dir_{};
    cv::Vec3d screen_glint_{};
    cv::Vec3d lp_{};
    cv::Vec3d ray_dir_{};
};

}// namespace et

#endif// EYE_TRACKER__RAYPOINTMINIMIZER_HPP_
