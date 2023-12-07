#ifndef EYE_TRACKER_VISUALANGLESOPTIMIZER_HPP
#define EYE_TRACKER_VISUALANGLESOPTIMIZER_HPP

#include <opencv2/opencv.hpp>
#include "eye_tracker/optimizers/EyeCentreOptimizer.hpp"

namespace et
{

    class VisualAnglesOptimizer : public cv::DownhillSolver::Function
    {
    public:
        VisualAnglesOptimizer();
        void setParameters(std::vector<cv::Point3d> front_corners, std::vector<cv::Point3d> back_corners, double alpha,
                           double beta, double cornea_centre_distance);
        cv::Vec3d getCrossPoint() const;

        double calc(const double *x) const override;
    private:

        int getDims() const override;

        cv::Vec3d cross_point_{};
        std::vector<cv::Vec3d> visual_axes_{};
        std::vector<cv::Vec3d> optical_axes_{};
        double cornea_centre_distance_{};

        std::vector<cv::Point3d> front_corners_{};
        std::vector<cv::Point3d> back_corners_{};
    };

} // et

#endif //EYE_TRACKER_VISUALANGLESOPTIMIZER_HPP
