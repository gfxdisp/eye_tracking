#ifndef HDRMFS_EYE_TRACKER_GLINT_CIRCLE_OPTIMIZER_HPP
#define HDRMFS_EYE_TRACKER_GLINT_CIRCLE_OPTIMIZER_HPP

#include <opencv2/core/optim.hpp>

namespace et {

/** Finds a circle, on which glints lie. The position is calculated
 * based on the circle from previous frame and a vector of glints.
 */
class GlintCircleOptimizer : public cv::DownhillSolver::Function {
public:
    /**
     * Sets the parameters needed for minimization.
     * @param glints a vector of glint positions in the current frame.
     * @param previous_centre a centre position of the circle fit
     * in the previous frame.
     * @param previous_radius a radius of the circle fit in the previous frame.
     */
    void setParameters(const std::vector<cv::Point2f> &glints,
                       const cv::Point2d &previous_centre,
                       double previous_radius);

private:
    // A vector of glints in the current frame.
    std::vector<cv::Point2f> glints_{};
    // Expected error of the glint positions (in pixels).
    double glints_sigma_{2};
    // Position of the circle's centre from the previous frame.
    cv::Point2d previous_centre_{};
    // Expected error of the previous circle centre position (in pixels).
    double centre_sigma_{2};
    // Radius of the circle from the previous frame.
    double previous_radius_{};
    // Expected error of the previous circle radius (in pixels).
    double radius_sigma_{5};

    /**
     * Internal cv::DownhillSolver method used to calculate number of optimized
     * parameters.
     * @return number of optimized parameters: 3 (x, y, radius).
     */
    [[nodiscard]] int getDims() const override;

    /**
     * Internal cv::DownhillSolver method used to run a single optimization
     * iteration, finding a circle, on which glints lie.
     * @param x set of parameters for which the function is checked: x, y, radius.
     * @return total error computed for the parameters.
     */
    double calc(const double *x) const override;
};

} // namespace et

#endif //HDRMFS_EYE_TRACKER_GLINT_CIRCLE_OPTIMIZER_HPP
