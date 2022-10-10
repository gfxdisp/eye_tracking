#ifndef EYE_TRACKER__RAYPOINTMINIMIZER_HPP_
#define EYE_TRACKER__RAYPOINTMINIMIZER_HPP_

#include <opencv2/opencv.hpp>

#include <vector>

namespace et {
class RayPointMinimizer : public cv::DownhillSolver::Function {
public:
    void initialize();

    void setParameters(const cv::Vec3f &np2c_dir, cv::Vec3f *screen_glint,
                       std::vector<cv::Vec3f> &lp);

    static double kk_;
    static double lowest_error_;

private:
    cv::Vec3f np_{};
    cv::Vec3f np2c_dir_{};
    std::vector<cv::Vec3f> screen_glint_{};
    std::vector<cv::Vec3f> lp_{};
    std::vector<cv::Vec3f> ray_dir_{};
    bool initialized_{false};

    [[nodiscard]] int getDims() const override;
    double calc(const double *x) const override;
};

}// namespace et

#endif// EYE_TRACKER__RAYPOINTMINIMIZER_HPP_
