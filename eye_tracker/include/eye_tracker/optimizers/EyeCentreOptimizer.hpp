#ifndef EYE_TRACKER_EYECENTREOPTIMIZER_HPP
#define EYE_TRACKER_EYECENTREOPTIMIZER_HPP

#include <opencv2/opencv.hpp>

namespace et
{

    class EyeCentreOptimizer : public cv::DownhillSolver::Function
    {
    public:
        void
        setParameters(cv::Point3d cross_point, std::vector<cv::Vec3d> visual_axes, std::vector<cv::Vec3d> optical_axes,
                      double cornea_centre_distance);

    private:

        int getDims() const override;

        double calc(const double *x) const override;

        cv::Point3d cross_point_{};
        std::vector<cv::Vec3d> visual_axes_{};
        std::vector<cv::Vec3d> optical_axes_{};
        double cornea_centre_distance_{};
    };

} // et

#endif //EYE_TRACKER_EYECENTREOPTIMIZER_HPP
