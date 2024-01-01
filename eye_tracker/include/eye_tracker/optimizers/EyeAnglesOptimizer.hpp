#ifndef EYE_TRACKER_EYEANGLESOPTIMIZER_HPP
#define EYE_TRACKER_EYEANGLESOPTIMIZER_HPP

#include <opencv2/opencv.hpp>

namespace et
{

    class EyeAnglesOptimizer : public cv::DownhillSolver::Function
    {
    public:
        void setParameters(cv::Vec3d visual_axis, cv::Vec3d optical_axis);

    private:
        int getDims() const override;

        double calc(const double *x) const override;

        cv::Vec3d visual_axis_{};
        cv::Vec3d optical_axis_{};
    };

} // et

#endif //EYE_TRACKER_EYEANGLESOPTIMIZER_HPP