#ifndef HDRMFS_EYE_TRACKER_RAY_POINT_MINIMIZER_HPP
#define HDRMFS_EYE_TRACKER_RAY_POINT_MINIMIZER_HPP

#include <opencv2/opencv.hpp>

#include <vector>

namespace et {
/**
 * Finds a position of the cornea centre based on the positions of LEDs and their
 * corresponding reflections.
 */
class RayPointMinimizer : public cv::DownhillSolver::Function {
public:
    /**
     * Sets the initial parameters of the optimization.
     */
    void initialize();

    /**
     * Sets the parameters needed for minimization.
     * @param np2c_dir Direction vector between camera's nodal point and
     * cornea centre.
     * @param screen_glint Vector of glints in image space corresponding to LEDs.
     * @param lp Vector of LED positions in camera space corresponding to glints.
     */
    void setParameters(const cv::Vec3f &np2c_dir, cv::Vec3f *screen_glint,
                       std::vector<cv::Vec3f> &lp);

private:
    // Camera space position of the nodal point - always (0,0,0).
    cv::Vec3f np_{};
    // Direction vector between camera's nodal point and cornea centre.
    cv::Vec3f np2c_dir_{};
    // Vector of glints in image space corresponding to LEDs.
    std::vector<cv::Vec3f> screen_glint_{};
    // Vector of LED positions in camera space corresponding to glints.
    std::vector<cv::Vec3f> lp_{};
    // Vector of directions between camera's nodal point and glints.
    std::vector<cv::Vec3f> ray_dir_{};
    // Set to true after running initialize() method.
    bool initialized_{false};

    /**
     * Internal cv::DownhillSolver method used to calculate number of optimized
     * parameters.
     * @return number of optimized parameters.
     */
    [[nodiscard]] int getDims() const override;
    /**
     * Internal cv::DownhillSolver method used to run a single optimization
     * iteration.
     * @param x set of parameters for which the function is checked.
     * @return total error computed for the parameters.
     */
    double calc(const double *x) const override;
};

} // namespace et

#endif // HDRMFS_EYE_TRACKER_RAY_POINT_MINIMIZER_HPP
