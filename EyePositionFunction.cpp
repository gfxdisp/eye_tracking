#include "EyePositionFunction.hpp"

#include <iostream>
#include <utility>

namespace et {

SetupLayout EyePositionFunction::setup_layout_{};
double EyePositionFunction::best_error_{1e5};

EyePositionFunction::EyePositionFunction(EyeTracker *eye_tracker, cv::Point2f pupil_pixel_position,
                                         cv::Point2f *glints_pixel_positions, const cv::Vec3d &ground_truth)
    : eye_tracker_(eye_tracker), pupil_pixel_position_(std::move(pupil_pixel_position)),
      glints_pixel_positions_(glints_pixel_positions), ground_truth_(ground_truth) {
}

int EyePositionFunction::getDims() const {
    return 9;
}

double EyePositionFunction::calc(const double *x) const {

    SetupLayout setup_layout{};
    setup_layout.camera_lambda = 27.119;
    setup_layout.camera_nodal_point_position = cv::Vec3d(x[0], x[1], x[2]);
    setup_layout.led_positions[0] = cv::Vec3d(x[3], x[4], x[5]);
    setup_layout.led_positions[1] = cv::Vec3d(x[6], x[7], x[8]);

    eye_tracker_->setNewSetupLayout(setup_layout);
    EyePosition eye_position = eye_tracker_->calculateJoined(pupil_pixel_position_, glints_pixel_positions_);
    double error{cv::norm(*eye_position.cornea_curvature - ground_truth_)};

    if (abs(x[4] - x[7]) > 10.0f || abs(x[5] - x[8]) > 10.0f || abs(x[3] - x[6]) > 50.0f || abs(x[3] - x[6]) < 20.0f
        || x[3] > x[0] || x[6] < x[0] || abs((x[6] - x[0]) - (x[0] - x[3])) > 25.0f) {
        error = 1e5;
    }

    if (error < best_error_) {
        best_error_ = error;
        setup_layout_ = setup_layout;
    }
    return error;
}
}// namespace et
