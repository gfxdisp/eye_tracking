#ifndef EYE_TRACKER_GLINTPOSITIONOPTIMIZER_HPP
#define EYE_TRACKER_GLINTPOSITIONOPTIMIZER_HPP

#include <opencv2/opencv.hpp>

namespace et
{

    class GlintPositionOptimizer : public cv::DownhillSolver::Function
    {
    public:
        void setParameters(cv::Point3d nodal_point, double cornea_radius, cv::Point3d led_position, cv::Point3d camera_position);

    private:
        double calc(const double *x) const override;

        int getDims() const override;

        cv::Point3d nodal_point_{};
        double cornea_radius_{};
        cv::Point3d led_position_{};
        cv::Point3d camera_position_{};
    };

} // et

#endif //EYE_TRACKER_GLINTPOSITIONOPTIMIZER_HPP
