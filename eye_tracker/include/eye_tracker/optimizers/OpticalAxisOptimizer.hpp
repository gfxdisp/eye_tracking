#ifndef EYE_TRACKER_OPTICALAXISOPTIMIZER_HPP
#define EYE_TRACKER_OPTICALAXISOPTIMIZER_HPP

#include <eye_tracker/Settings.hpp>
#include <opencv2/opencv.hpp>

namespace et
{
    class OpticalAxisOptimizer : public cv::DownhillSolver::Function
    {
    public:
        void setParameters(const EyeMeasurements& eye_measurements, const cv::Point3d& eye_centre,
                           const cv::Point3d& focus_point);

    public:
        int getDims() const override;

        double calc(const double* x) const override;

    private:

        EyeMeasurements eye_measurements_{};
        cv::Point3d eye_centre_{};
        cv::Point3d focus_point_{};
    };
} // et

#endif //EYE_TRACKER_OPTICALAXISOPTIMIZER_HPP
