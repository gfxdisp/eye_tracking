#ifndef EYE_TRACKER_AGGREGATEDPOLYNOMIALOPTIMIZER_HPP
#define EYE_TRACKER_AGGREGATEDPOLYNOMIALOPTIMIZER_HPP

#include "eye_tracker/eye/PolynomialEyeEstimator.hpp"

#include <opencv2/opencv.hpp>

namespace et
{

    class AggregatedPolynomialOptimizer : public cv::DownhillSolver::Function
    {
    public:
        std::vector<cv::Point2d> pupils{};
        std::vector<cv::RotatedRect> ellipses{};

        std::vector<cv::Point3d> eye_centres{};
        std::vector<cv::Vec3d> visual_axes{};

        double calc(const double *x) const override;

        constexpr static double ACCEPTABLE_ANGLE_ERROR = 0.5;
//        constexpr static double ACCEPTABLE_ANGLE_ERROR = 1.0;

        constexpr static double ACCEPTABLE_DISTANCE_ERROR = 0.1553;
//        constexpr static double ACCEPTABLE_DISTANCE_ERROR = 0.4105;
    private:
        int getDims() const override;

        PolynomialEyeEstimator polynomial_eye_estimator_{0};
    };

} // et

#endif //EYE_TRACKER_AGGREGATEDPOLYNOMIALOPTIMIZER_HPP
