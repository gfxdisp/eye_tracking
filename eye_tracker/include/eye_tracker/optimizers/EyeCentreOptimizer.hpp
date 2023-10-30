#ifndef EYE_TRACKER_EYECENTREOPTIMIZER_HPP
#define EYE_TRACKER_EYECENTREOPTIMIZER_HPP

#include <opencv2/opencv.hpp>

namespace et
{

    class EyeCentreOptimizer : public cv::DownhillSolver::Function
    {
    public:
        void setParameters(std::vector<cv::Point3f> front_corners, std::vector<cv::Point3f> back_corners, float alpha,
                           float beta, float cornea_centre_distance);

        cv::Vec3f getCrossPoint() const;

        double calc(const double *x) const override;
    private:

        int getDims() const override;

        cv::Vec3f cross_point_{};
        std::vector<cv::Vec3f> visual_axes_{};
        std::vector<cv::Vec3f> optical_axes_{};
        float cornea_centre_distance_{};
    };

} // et

#endif //EYE_TRACKER_EYECENTREOPTIMIZER_HPP
