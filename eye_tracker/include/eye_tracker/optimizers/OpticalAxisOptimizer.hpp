#ifndef OPTICALAXISOPTIMIZER_HPP
#define OPTICALAXISOPTIMIZER_HPP

#include <eye_tracker/Settings.hpp>
#include <opencv2/core/optim.hpp>

namespace et {

class OpticalAxisOptimizer : public cv::ConjGradSolver::Function
{
public:
    void setParameters(const EyeParams& eye_measurements, const cv::Vec3d& eye_centre,
                       const cv::Vec3d& focus_point);

public:
    [[nodiscard]] int getDims() const override;

    double calc(const double* x) const override;

    // void getGradient(const double* x, double* grad) override;

private:

    EyeParams eye_measurements_{};
    cv::Vec3d eye_centre_{};
    cv::Vec3d focus_point_{};
};

} // et

#endif //OPTICALAXISOPTIMIZER_HPP
