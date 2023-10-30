#ifndef EYE_TRACKER_AGGREGATEDPOLYNOMIALOPTIMIZER_HPP
#define EYE_TRACKER_AGGREGATEDPOLYNOMIALOPTIMIZER_HPP

#include "eye_tracker/eye/PolynomialEyeEstimator.hpp"

#include <opencv2/opencv.hpp>

namespace et
{

    class AggregatedPolynomialOptimizer : public cv::DownhillSolver::Function
    {
    public:
        std::vector<cv::Point2f> pupils{};
        std::vector<cv::RotatedRect> ellipses{};

        std::vector<cv::Point3f> eye_centres{};
        std::vector<cv::Vec3f> visual_axes{};

        double calc(const double *x) const override;

        constexpr static float ACCEPTABLE_ANGLE_ERROR = 0.5f;
//        constexpr static float ACCEPTABLE_ANGLE_ERROR = 1.0f;

        constexpr static float ACCEPTABLE_DISTANCE_ERROR = 0.1553f;
//        constexpr static float ACCEPTABLE_DISTANCE_ERROR = 0.4105f;
    private:
        int getDims() const override;

        PolynomialEyeEstimator polynomial_eye_estimator_{0};
    };

} // et

#endif //EYE_TRACKER_AGGREGATEDPOLYNOMIALOPTIMIZER_HPP
