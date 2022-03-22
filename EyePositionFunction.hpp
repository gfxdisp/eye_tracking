#ifndef EYE_TRACKER__EYEPOSITIONFUNCTION_HPP_
#define EYE_TRACKER__EYEPOSITIONFUNCTION_HPP_

#include "EyeTracker.hpp"

#include <opencv2/opencv.hpp>

namespace et {
class EyePositionFunction : public cv::MinProblemSolver::Function {
public:
    EyePositionFunction(EyeTracker *eye_tracker, cv::Point2f pupil_pixel_position, cv::Point2f *glints_pixel_positions,
                        const cv::Vec3d &ground_truth);

    [[nodiscard]] int getDims() const override;

    double calc(const double *x) const override;

    static SetupLayout setup_layout_;

private:
    EyeTracker *eye_tracker_{};
    cv::Point2f pupil_pixel_position_{};
    cv::Point2f *glints_pixel_positions_{};
    cv::Vec3d ground_truth_{};
    static double best_error_;
};

}// namespace et

#endif//EYE_TRACKER__EYEPOSITIONFUNCTION_HPP_
