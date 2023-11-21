#ifndef EYE_TRACKER_PIXELPOSOPTIMIZERTEST_HPP
#define EYE_TRACKER_PIXELPOSOPTIMIZERTEST_HPP

#include <opencv2/opencv.hpp>
#include "eye_tracker/eye/ModelEyeEstimator.hpp"

namespace et
{

    class PixelPosOptimizerTest : public cv::DownhillSolver::Function
    {
    public:
        void setParameters(std::shared_ptr<ModelEyeEstimator> model_eye_estimator, cv::Point3d nodal_point,
                           cv::Point3d eye_centre);
    private:
        int getDims() const override;
        double calc(const double *x) const override;
        std::shared_ptr<ModelEyeEstimator> model_eye_estimator_{};
        cv::Point3d nodal_point_{};
        cv::Point3d eye_centre_{};
    };

} // et

#endif //EYE_TRACKER_PIXELPOSOPTIMIZERTEST_HPP
