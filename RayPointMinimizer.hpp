#ifndef EYE_TRACKER__RAYPOINTMINIMIZER_HPP_
#define EYE_TRACKER__RAYPOINTMINIMIZER_HPP_

#include <opencv2/opencv.hpp>

namespace et {
class RayPointMinimizer : public cv::DownhillSolver::Function {
public:
    explicit RayPointMinimizer(const cv::Vec3d &np);
    [[nodiscard]] int getDims() const override;
    double calc(const double *x) const override;

    void setParameters(const cv::Vec3d &np2c_dir, cv::Vec3d *screen_glint, cv::Vec3d *lp);

    static double kk_;
    static double lowest_error_;

private:
    cv::Vec3d np_{};
    cv::Vec3d np2c_dir_{};
    cv::Vec3d screen_glint_[2]{};
    cv::Vec3d lp_[2]{};
    cv::Vec3d ray_dir_[2]{};
};

}// namespace et

#endif// EYE_TRACKER__RAYPOINTMINIMIZER_HPP_
