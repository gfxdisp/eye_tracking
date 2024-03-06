#ifndef CORNEAOPTIMIZER_HPP
#define CORNEAOPTIMIZER_HPP

#include <eye_tracker/Settings.hpp>
#include <opencv2/core/optim.hpp>

namespace et {

class CorneaOptimizer : public cv::ConjGradSolver::Function
{
public:
    void setParameters(const EyeMeasurements& eye_measurements, const cv::Vec3d& eye_centre,
                       const cv::Vec3d& focus_point);

public:
    int getDims() const override;

    double calc(const double* x) const override;

    // void getGradient(const double* x, double* grad) override;

private:

    EyeMeasurements eye_measurements_{};
    cv::Vec3d eye_centre_{};
    cv::Vec3d focus_point_{};
};

} // et

#endif //CORNEAOPTIMIZER_HPP
