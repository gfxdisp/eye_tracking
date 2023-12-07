#ifndef EYE_TRACKER_PUPILPOSITIONOPTIMIZER_HPP
#define EYE_TRACKER_PUPILPOSITIONOPTIMIZER_HPP

#include <opencv2/opencv.hpp>

namespace et
{

    class PupilPositionOptimizer : public cv::DownhillSolver::Function
    {
    public:
        void setParameters(cv::Point3d nodal_point, cv::Point3d pupil_center, cv::Point3d camera_position, double cornea_radius, double refraction_index);
    private:
        double calc(const double *x) const override;

        int getDims() const override;

        cv::Point3d nodal_point_{};
        cv::Point3d pupil_center_{};
        cv::Point3d camera_position_{};

        double cornea_radius_{};
        double refraction_index_{};
    };

} // et

#endif //EYE_TRACKER_PUPILPOSITIONOPTIMIZER_HPP
