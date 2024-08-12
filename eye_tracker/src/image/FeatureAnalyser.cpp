#include <eye_tracker/image/FeatureAnalyser.hpp>

#include <memory>
#include <cmath>

namespace et {
    FeatureAnalyser::FeatureAnalyser(int camera_id) : camera_id_(camera_id) {
        auto& leds_positions = Settings::parameters.leds_positions[camera_id];
        glint_locations_distorted_.resize(leds_positions.size());
        glint_locations_undistorted_.resize(leds_positions.size());
        glint_validity_.resize(leds_positions.size());

        image_preprocessor_ = std::make_shared<ImagePreprocessor>(camera_id);
        temporal_filterer_ = std::make_shared<TemporalFilterer>(camera_id);
        feature_estimator_ = std::make_shared<FeatureEstimator>(camera_id);
        intrinsic_matrix_ = &Settings::parameters.camera_params[camera_id].intrinsic_matrix;
        capture_offset_ = &Settings::parameters.camera_params[camera_id].capture_offset;
        distortion_coefficients_ = &Settings::parameters.camera_params[camera_id].distortion_coefficients;
    }

    void FeatureAnalyser::getPupilDistorted(cv::Point2d& pupil_location_distorted) {
        mtx_features_.lock();
        pupil_location_distorted = pupil_location_distorted_;
        mtx_features_.unlock();
    }

    void FeatureAnalyser::getPupilUndistorted(cv::Point2d& pupil_position_undistorted) {
        mtx_features_.lock();
        pupil_position_undistorted = pupil_location_undistorted_;
        mtx_features_.unlock();
    }

    void FeatureAnalyser::getPupilRadiusUndistorted(double& pupil_radius_undistorted) {
        mtx_features_.lock();
        pupil_radius_undistorted = pupil_radius_undistorted_;
        mtx_features_.unlock();
    }

    void FeatureAnalyser::getPupilRadiusDistorted(double& pupil_radius_distorted) {
        mtx_features_.lock();
        pupil_radius_distorted = pupil_radius_distorted_;
        mtx_features_.unlock();
    }

    void FeatureAnalyser::getGlints(std::vector<cv::Point2d>& glints) {
        mtx_features_.lock();
        glints = glint_locations_undistorted_;
        mtx_features_.unlock();
    }

    void FeatureAnalyser::getDistortedGlints(std::vector<cv::Point2d>& glints) {
        mtx_features_.lock();
        glints = glint_locations_distorted_;
        mtx_features_.unlock();
    }

    void FeatureAnalyser::getGlintsValidity(std::vector<bool>& glint_validity) {
        mtx_features_.lock();
        glint_validity = glint_validity_;
        mtx_features_.unlock();
    }

    void FeatureAnalyser::getThresholdedPupilImage(cv::Mat& image) {
        mtx_features_.lock();
        image = thresholded_pupil_image_.clone();
        mtx_features_.unlock();
    }

    void FeatureAnalyser::getThresholdedGlintsImage(cv::Mat& image) {
        mtx_features_.lock();
        image = thresholded_glints_image_.clone();
        mtx_features_.unlock();
    }

    void FeatureAnalyser::preprocessImage(const EyeImage& image) {
        EyeImage output{.frame = cv::Mat(image.frame.rows, image.frame.cols, CV_8UC1)};
        image_preprocessor_->preprocess(image, thresholded_pupil_image_, thresholded_glints_image_);
        frame_num_ = image.frame_num;
    }

    bool FeatureAnalyser::findPupil() {
        cv::Point2d estimated_pupil_location{};
        double estimated_pupil_radius{};
        const bool success = feature_estimator_->findPupil(thresholded_pupil_image_, estimated_pupil_location, estimated_pupil_radius);
        if (!success) {
            return false;
        }

        temporal_filterer_->filterPupil(estimated_pupil_location, estimated_pupil_radius);

        pupil_location_distorted_ = estimated_pupil_location;
        pupil_radius_distorted_ = estimated_pupil_radius;

        const auto centre_undistorted = undistort(estimated_pupil_location);
        cv::Point2d pupil_left_side = estimated_pupil_location;
        pupil_left_side.x -= estimated_pupil_radius;
        cv::Point2d pupil_right_side = estimated_pupil_location;
        pupil_right_side.x += estimated_pupil_radius;

        const auto left_side_undistorted = undistort(pupil_left_side);
        const auto right_side_undistorted = undistort(pupil_right_side);
        const double radius_undistorted = (right_side_undistorted.x - left_side_undistorted.x) * 0.5;
        mtx_features_.lock();
        pupil_location_undistorted_ = centre_undistorted;
        pupil_radius_undistorted_ = radius_undistorted;
        mtx_features_.unlock();

        return true;
    }

    bool FeatureAnalyser::findEllipsePoints() {
        std::vector<cv::Point2f> ellipse_points{};
        const static int leds_per_side = static_cast<int>(et::Settings::parameters.leds_positions[camera_id_].size()) / 2;
        std::fill(glint_validity_.begin(), glint_validity_.end(), false);

        const bool success = feature_estimator_->findGlints(thresholded_glints_image_, ellipse_points);
        if (!success) {
            return false;
        }

        temporal_filterer_->filterGlints(ellipse_points);
        if (ellipse_points.size() < 5) {
            return false;
        }

        cv::RotatedRect ellipse = cv::fitEllipse(ellipse_points);
        int left_glints = static_cast<int>(std::count_if(ellipse_points.begin(), ellipse_points.end(), [&ellipse](const cv::Point2d& pt) {
            return pt.x < ellipse.center.x;
        }));
        int right_glints = static_cast<int>(ellipse_points.size()) - left_glints;

        if (left_glints == 0 || right_glints == 0) {
            return false;
        }

        // The points are sorted from the bottom left side of the ellipse to the top left, and then from the bottom right to the top right.
        std::sort(ellipse_points.begin(), ellipse_points.end(), [ellipse](cv::Point2f a, cv::Point2f b) {
            if ((a.x < ellipse.center.x && b.x < ellipse.center.x) || (a.x > ellipse.center.x && b.x > ellipse.center.x)) {
                return a.y > b.y;
            } else {
                return a.x < b.x;
            }
        });

        cv::Point2d glints_centre = ellipse.center;
        auto calculateRadius = [](const std::vector<cv::Point2d>& points, const cv::Point2d& center) {
            return std::accumulate(points.begin(), points.end(), 0.0, [&center](const double sum, const cv::Point2d& point) {
                return sum + cv::norm(point - center);
            }) / static_cast<double>(points.size());
        };

        static std::vector<cv::Point2d> glint_locations_distorted(leds_per_side * 2);
        static std::vector<bool> glint_validity(leds_per_side * 2);

        glint_locations_distorted = glint_locations_distorted_;

        auto processGlints = [&](const std::vector<cv::Point2d>& sorted_glints, const int start_index, const int glints_count, const cv::Point2d& center, const double radius) {
            glint_locations_distorted[start_index] = sorted_glints[0];
            glint_validity[start_index] = true;
            double expected_angle_between_glints = atan2(sorted_glints[1].y - center.y, sorted_glints[1].x - center.x) - atan2(sorted_glints[0].y - center.y, sorted_glints[0].x - center.x);
            int counter = 1;
            for (int i = 1; counter < glints_count; i++) {
                double previous_angle = atan2(glint_locations_distorted[start_index + counter - 1].y - center.y, glint_locations_distorted[start_index + counter - 1].x - center.x);
                // If there are no more glints, we add a virtual glint.
                if (i >= sorted_glints.size()) {
                    double current_angle = previous_angle + expected_angle_between_glints;
                    glint_locations_distorted[start_index + counter] = {center.x + radius * std::cos(current_angle), center.y + radius * std::sin(current_angle)};
                    glint_validity[start_index + counter] = false;
                    counter++;
                    i--;
                    continue;
                }
                double current_angle = atan2(sorted_glints[i].y - center.y, sorted_glints[i].x - center.x);
                if (current_angle - previous_angle < -M_PI)
                    previous_angle -= 2 * M_PI;
                if (current_angle - previous_angle > M_PI)
                    previous_angle += 2 * M_PI;
                // If the angle between the glints is close to the expected angle, we treat the glint as valid.
                if (std::abs(current_angle - previous_angle - expected_angle_between_glints) < 0.15) {
                    glint_locations_distorted[start_index + counter] = sorted_glints[i];
                    glint_validity[start_index + counter] = true;
                    counter++;
                }
                // If the angle between the glints is not close to the expected angle, we add a virtual glint.
                else {
                    double angle = previous_angle + expected_angle_between_glints;
                    glint_locations_distorted[start_index + counter] = {center.x + radius * std::cos(angle), center.y + radius * std::sin(angle)};
                    glint_validity[start_index + counter] = false;
                    counter++;
                    i--;
                }
            }
        };

        std::vector<cv::Point2d> sorted_left_glints(ellipse_points.begin(), ellipse_points.begin() + left_glints);
        double left_glints_radius = calculateRadius(sorted_left_glints, glints_centre);
        processGlints(sorted_left_glints, 0, leds_per_side, glints_centre, left_glints_radius);

        std::vector<cv::Point2d> sorted_right_glints(ellipse_points.begin() + left_glints, ellipse_points.end());
        double right_glints_radius = calculateRadius(sorted_right_glints, glints_centre);
        processGlints(sorted_right_glints, leds_per_side, leds_per_side, glints_centre, right_glints_radius);

        ellipse_points.clear();
        for (int i = 0; i < leds_per_side * 2; i++) {
            if (glint_validity[i]) {
                ellipse_points.push_back(undistort(glint_locations_distorted[i]));
            }
        }

        if (ellipse_points.size() < 5) {
            ellipse.center = glints_centre;
        } else {
            ellipse = cv::fitEllipse(ellipse_points);
        }

        mtx_features_.lock();
        glint_validity_ = glint_validity;
        for (int i = 0; i < leds_per_side * 2; i++) {
            glint_locations_undistorted_[i] = undistort(glint_locations_distorted[i]) + static_cast<cv::Point2d>(ellipse.center) - static_cast<cv::Point2d>(ellipse.center);
            glint_locations_distorted_[i] = distort(glint_locations_undistorted_[i]);
        }
        mtx_features_.unlock();

        return true;
    }

    cv::Point2d FeatureAnalyser::undistort(const cv::Point2d point) const {
        cv::Point2d new_point{point};

        const std::vector<cv::Point2d> points{{point.x + 1 + capture_offset_->width, point.y + 1 + capture_offset_->height}};
        std::vector new_points{new_point};

        cv::undistortPoints(points, new_points, intrinsic_matrix_->t(), *distortion_coefficients_);

        new_point = new_points[0];
        new_point.x *= intrinsic_matrix_->at<double>(0, 0);
        new_point.y *= intrinsic_matrix_->at<double>(1, 1);
        new_point.x += intrinsic_matrix_->at<double>(2, 0);
        new_point.y += intrinsic_matrix_->at<double>(2, 1);

        new_point.x -= 1 + capture_offset_->width;
        new_point.y -= 1 + capture_offset_->height;

        return new_point;
    }

    cv::Point2d FeatureAnalyser::distort(const cv::Point2d point) const {
        const double cx = intrinsic_matrix_->at<double>(2, 0);
        const double cy = intrinsic_matrix_->at<double>(2, 1);
        const double fx = intrinsic_matrix_->at<double>(0, 0);
        const double fy = intrinsic_matrix_->at<double>(1, 1);

        const double x = (point.x - cx) / fx;
        const double y = (point.y - cy) / fy;

        const double r2 = x * x + y * y;

        const double k1 = (*distortion_coefficients_)[0];
        const double k2 = (*distortion_coefficients_)[1];
        const double p1 = (*distortion_coefficients_)[2];
        const double p2 = (*distortion_coefficients_)[3];
        const double k3 = (*distortion_coefficients_)[4];

        const double x_distorted = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2) + 2 * p1 * x * y + p2 * (r2 + 2 * x * x);
        const double y_distorted = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2) + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y;

        return {x_distorted * fx + cx, y_distorted * fy + cy};
    }
} // namespace et
