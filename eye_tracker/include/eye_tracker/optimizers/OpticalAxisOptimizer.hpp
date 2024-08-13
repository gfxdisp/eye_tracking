#ifndef HDRMFS_EYE_TRACKER_OPTICAL_AXIS_OPTIMIZER_HPP
#define HDRMFS_EYE_TRACKER_OPTICAL_AXIS_OPTIMIZER_HPP

#include <eye_tracker/Settings.hpp>

#include <opencv2/core/optim.hpp>

namespace et {
    class OpticalAxisOptimizer : public cv::ConjGradSolver::Function {
    public:
        void setParameters(const EyeParams& eye_measurements, const cv::Vec3d& eye_centre, const cv::Vec3d& focus_point);

    private:
        int getDims() const override;

        double calc(const double* x) const override;

        EyeParams eye_measurements_{};
        cv::Vec3d eye_centre_{};
        cv::Vec3d focus_point_{};
    };
} // et

#endif //HDRMFS_EYE_TRACKER_OPTICAL_AXIS_OPTIMIZER_HPP
