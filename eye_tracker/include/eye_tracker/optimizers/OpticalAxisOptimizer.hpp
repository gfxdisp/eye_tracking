#ifndef EYE_TRACKER_OPTICALAXISOPTIMIZER_HPP
#define EYE_TRACKER_OPTICALAXISOPTIMIZER_HPP

#include <opencv2/opencv.hpp>

namespace et
{

    class OpticalAxisOptimizer : public cv::DownhillSolver::Function
    {
    public:
        void setParameters(float alpha, float beta, float cornea_centre_distance, cv::Point3f eye_centre,
                           cv::Point3f focus_point);

    private:
        int getDims() const override;

        double calc(const double *x) const override;

        float alpha_{};
        float beta_{};
        float cornea_centre_distance_{};
        cv::Point3f eye_centre_{};
        cv::Point3f focus_point_{};
    };

} // et

#endif //EYE_TRACKER_OPTICALAXISOPTIMIZER_HPP
