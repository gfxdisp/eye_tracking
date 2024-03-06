#include <eye_tracker/eye/EyeEstimator.hpp>
#include <eye_tracker/eye/ModelEyeEstimator.hpp>
#include <eye_tracker/optimizers/EyeForPixelOptimizer.hpp>

namespace et {
    void EyeForPixelOptimizer::setParameters(const std::shared_ptr<ModelEyeEstimator>& eye_estimator, const cv::Point2d& pixel, const EyeMeasurements& eye_measurements, double depth)
    {
        eye_estimator_ = eye_estimator;
        pixel_ = pixel;
        eye_measurements_ = eye_measurements;
        depth_ = depth;
    }

    double EyeForPixelOptimizer::calc(const double* x) const
    {
        cv::Vec3d eye_centre{x[0], x[1], depth_};
        cv::Vec3d nodal_point = eye_centre + cv::Vec3d(0, 0, -1) * eye_measurements_.cornea_centre_distance;
        EyeInfo eye_info{};
        eye_estimator_->invertDetectEye(eye_info, nodal_point, eye_centre, eye_measurements_);
        double error = cv::norm(eye_info.pupil - pixel_);
        return error;
    }

    int EyeForPixelOptimizer::getDims() const
    {
        return 2;
    }
} // et