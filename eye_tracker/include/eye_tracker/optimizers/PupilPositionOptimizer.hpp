#ifndef EYE_TRACKER_PUPILPOSITIONOPTIMIZER_HPP
#define EYE_TRACKER_PUPILPOSITIONOPTIMIZER_HPP

#include <eye_tracker/Settings.hpp>
#include <opencv2/opencv.hpp>

namespace et
{

    class PupilPositionOptimizer : public cv::DownhillSolver::Function
    {
    public:
        void setParameters(const cv::Vec3d& nodal_point, const cv::Vec3d& pupil_center, const cv::Vec3d& camera_position, const EyeMeasurements& eye_measurements);
        double calc(const double *x) const override;

        int getDims() const override;
    private:
        cv::Vec3d nodal_point_{};
        cv::Vec3d pupil_center_{};
        cv::Vec3d camera_position_{};

        EyeMeasurements eye_measurements_{};
    };

} // et

#endif //EYE_TRACKER_PUPILPOSITIONOPTIMIZER_HPP
