#ifndef EYEFORPIXELOPTIMIZER_HPP
#define EYEFORPIXELOPTIMIZER_HPP

#include <eye_tracker/Settings.hpp>
#include <opencv2/core/optim.hpp>

namespace et
{
    class ModelEyeEstimator;

    class EyeForPixelOptimizer : public cv::DownhillSolver::Function
    {
    public:
        void setParameters(const std::shared_ptr<ModelEyeEstimator>& eye_estimator, const cv::Point2d& pixel, const EyeMeasurements& eye_measurements, double depth);
        double calc(const double *x) const override;
        int getDims() const override;
    private:
        std::shared_ptr<ModelEyeEstimator> eye_estimator_;
        cv::Point2d pixel_{};
        EyeMeasurements eye_measurements_{};
        double depth_{};
    };
} // et

#endif //EYEFORPIXELOPTIMIZER_HPP
