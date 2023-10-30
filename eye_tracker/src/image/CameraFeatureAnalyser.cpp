#include "eye_tracker/image/CameraFeatureAnalyser.hpp"
#include "eye_tracker/image/position/ContourPositionEstimator.hpp"
#include "eye_tracker/image/preprocess/CameraImagePreprocessor.hpp"
#include "eye_tracker/image/temporal_filter/ContinuousTemporalFilterer.hpp"

namespace et
{
    CameraFeatureAnalyser::CameraFeatureAnalyser(int camera_id) : FeatureAnalyser(camera_id)
    {
        image_preprocessor_ = std::make_shared<CameraImagePreprocessor>(camera_id);
        temporal_filterer_ = std::make_shared<ContinuousTemporalFilterer>(camera_id);
        position_estimator_ = std::make_shared<ContourPositionEstimator>(camera_id);
        intrinsic_matrix_ = &Settings::parameters.camera_params[camera_id].intrinsic_matrix;
        capture_offset_ = &Settings::parameters.camera_params[camera_id].capture_offset;
        distortion_coefficients_ = &Settings::parameters.camera_params[camera_id].distortion_coefficients;
    }

    cv::Point2f CameraFeatureAnalyser::undistort(cv::Point2f point)
    {
        // Convert from 0-based C++ coordinates to 1-based Matlab coordinates and add offset.
        cv::Point2f new_point{point.x + 1 + capture_offset_->width, point.y + 1 + capture_offset_->height};

        std::vector<cv::Point2f> points{point};
        std::vector<cv::Point2f> new_points{new_point};

        cv::undistortPoints(points, new_points, *intrinsic_matrix_, *distortion_coefficients_);

        new_point = new_points[0];
        // Remove normalization
        new_point.x *= intrinsic_matrix_->at<float>(0, 0);
        new_point.y *= intrinsic_matrix_->at<float>(1, 1);
        new_point.x += intrinsic_matrix_->at<float>(0, 2);
        new_point.y += intrinsic_matrix_->at<float>(1, 2);

        // Convert back to 0-based C++ coordinates and subtract offset.
        new_point.x -= 1 + capture_offset_->width;
        new_point.y -= 1 + capture_offset_->height;

        return new_point;
    }

    cv::Point2f CameraFeatureAnalyser::distort(cv::Point2f point)
    {
        float cx = intrinsic_matrix_->at<float>(0, 2);
        float cy = intrinsic_matrix_->at<float>(1, 2);
        float fx = intrinsic_matrix_->at<float>(0, 0);
        float fy = intrinsic_matrix_->at<float>(1, 1);

        float x = (point.x - cx) / fx;
        float y = (point.y - cy) / fy;

        float r2 = x * x + y * y;

        float k1 = (*distortion_coefficients_)[0];
        float k2 = (*distortion_coefficients_)[1];
        float p1 = (*distortion_coefficients_)[2];
        float p2 = (*distortion_coefficients_)[3];
        float k3 = (*distortion_coefficients_)[4];

        float x_distorted = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2) +
                            2 * p1 * x * y + p2 * (r2 + 2 * x * x);
        float y_distorted = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2) +
                            p1 * (r2 + 2 * y * y) + 2 * p2 * x * y;

        return cv::Point2f(x_distorted * fx + cx, y_distorted * fy + cy);
    }
} // et