#ifndef EYE_TRACKER_OPTICALAXISOPTIMIZER_HPP
#define EYE_TRACKER_OPTICALAXISOPTIMIZER_HPP

#include <opencv2/opencv.hpp>

namespace et
{

    class OpticalAxisOptimizer : public cv::DownhillSolver::Function
    {
    public:
        void setParameters(double alpha, double beta, double cornea_centre_distance, cv::Point3d eye_centre,
                           cv::Point3d focus_point);

    private:
        int getDims() const override;

        double calc(const double *x) const override;

        double alpha_{};
        double beta_{};
        double cornea_centre_distance_{};
        cv::Point3d eye_centre_{};
        cv::Point3d focus_point_{};
    };

} // et

#endif //EYE_TRACKER_OPTICALAXISOPTIMIZER_HPP
