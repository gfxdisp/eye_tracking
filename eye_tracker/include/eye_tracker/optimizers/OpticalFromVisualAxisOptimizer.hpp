#ifndef EYE_TRACKER_OPTICALFROMVISUALAXISOPTIMIZER_HPP
#define EYE_TRACKER_OPTICALFROMVISUALAXISOPTIMIZER_HPP

#include <opencv2/opencv.hpp>

namespace et
{

    class OpticalFromVisualAxisOptimizer : public cv::DownhillSolver::Function
    {
    public:
        void setParameters(double alpha, double beta, cv::Point3d visual_axis);

    private:
        int getDims() const override;

        double calc(const double *x) const override;

        double alpha_{};
        double beta_{};
        cv::Vec3d visual_axis_{};
    };

} // et

#endif //EYE_TRACKER_OPTICALFROMVISUALAXISOPTIMIZER_HPP
