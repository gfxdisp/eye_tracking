#include "eye_tracker/image/FeatureAnalyser.hpp"
#include "eye_tracker/image/position/ContourPositionEstimator.hpp"

#include <opencv2/cudaimgproc.hpp>
#include <memory>
#include <cmath>
#include <iostream>

namespace et {
    FeatureAnalyser::FeatureAnalyser(int camera_id) {
        camera_id_ = camera_id;
        glint_locations_distorted_.resize(Settings::parameters.leds_positions[camera_id].size());
        glint_locations_undistorted_.resize(Settings::parameters.leds_positions[camera_id].size());
        glint_validity_.resize(Settings::parameters.leds_positions[camera_id].size());
    }

    cv::Point2d FeatureAnalyser::getPupilUndistorted() {
        return pupil_location_undistorted_;
    }

    cv::Point2d FeatureAnalyser::getPupilDistorted() {
        return pupil_location_distorted_;
    }

    void FeatureAnalyser::getPupilUndistorted(cv::Point2d& pupil) {
        mtx_features_.lock();
        pupil = pupil_location_undistorted_;
        mtx_features_.unlock();
    }

    void FeatureAnalyser::getPupilBuffered(cv::Point2d& pupil) {
        mtx_features_.lock();
        pupil = pupil_location_buffered_;
        mtx_features_.unlock();
    }

    int FeatureAnalyser::getPupilRadiusUndistorted() const {
        return pupil_radius_undistorted_;
    }

    int FeatureAnalyser::getPupilRadiusDistorted() const {
        return pupil_radius_distorted_;
    }

    std::vector<cv::Point2d>* FeatureAnalyser::getGlints() {
        return &glint_locations_undistorted_;
    }

    std::vector<cv::Point2d>* FeatureAnalyser::getDistortedGlints() {
        return &glint_locations_distorted_;
    }

    std::vector<bool>* FeatureAnalyser::getGlintsValidity() {
        return &glint_validity_;
    }

    void FeatureAnalyser::preprocessImage(const EyeImage& image) {
        EyeImage output{
                .pupil = cv::Mat(image.pupil.rows, image.pupil.cols, CV_8UC1),
                .glints = cv::Mat(image.glints.rows, image.glints.cols, CV_8UC1)
        };
        image_preprocessor_->preprocess(image, output);
        pupil_thresholded_image_ = output.pupil;
        glints_thresholded_image_ = output.glints;
        frame_num_ = image.frame_num;
    }

    bool FeatureAnalyser::findPupil() {
        cv::Point2d estimated_pupil_location{};
        double estimated_pupil_radius{};
        bool success = position_estimator_->findPupil(pupil_thresholded_image_, estimated_pupil_location,
                                                      estimated_pupil_radius);
        if (!success) {
            return false;
        }

        temporal_filterer_->filterPupil(estimated_pupil_location, estimated_pupil_radius);

        pupil_location_distorted_ = estimated_pupil_location;
        pupil_radius_distorted_ = estimated_pupil_radius;

        auto centre_undistorted = undistort(estimated_pupil_location);
        cv::Point2d pupil_left_side = estimated_pupil_location;
        pupil_left_side.x -= estimated_pupil_radius;
        cv::Point2d pupil_right_side = estimated_pupil_location;
        pupil_right_side.x += estimated_pupil_radius;

        auto left_side_undistorted = undistort(pupil_left_side);
        auto right_side_undistorted = undistort(pupil_right_side);
        double radius_undistorted = (right_side_undistorted.x - left_side_undistorted.x) * 0.5;
        mtx_features_.lock();
        pupil_location_undistorted_ = centre_undistorted;
        pupil_radius_undistorted_ = radius_undistorted;
        mtx_features_.unlock();

        return true;
    }

    bool FeatureAnalyser::findEllipsePoints() {
        std::vector<cv::Point2f> ellipse_points{};
        int leds_per_side = (int) et::Settings::parameters.leds_positions[camera_id_].size() / 2;
        bool success = position_estimator_->findGlints(glints_thresholded_image_, ellipse_points);

        for (int i = 0; i < leds_per_side * 2; i++) {
            glint_validity_[i] = false;
        }

        if (!success) {
            return false;
        }

        temporal_filterer_->filterGlints(ellipse_points);
        if (ellipse_points.size() < 5) {
            return false;
        }

        cv::RotatedRect ellipse = cv::fitEllipse(ellipse_points);
        int left_glints = 0;
        int right_glints = 0;

        for (int i = 0; i < ellipse_points.size(); i++) {
            if (ellipse_points[i].x < ellipse.center.x) {
                left_glints++;
            } else {
                right_glints++;
            }
        }

        std::sort(ellipse_points.begin(), ellipse_points.end(), [ellipse](cv::Point2f a, cv::Point2f b) {
            if ((a.x < ellipse.center.x && b.x < ellipse.center.x) || (a.x > ellipse.center.x && b.x > ellipse.center.x)) {
                return a.y > b.y;
            } else {
                return a.x < b.x;
            }
        });

        if (left_glints < 2 || right_glints < 2) {
            return false;
        }

        {
            std::vector<cv::Point2d> sorted_left_glints{};
            for (int i = 0; i < left_glints; i++) {
                sorted_left_glints.push_back(ellipse_points[i]);
            }

            cv::Point2d left_glints_centre = ellipse.center;
            double left_glints_radius = 0;

            // Calculate the radius of the left glints.
            for (const auto& glint: sorted_left_glints) {
                left_glints_radius += cv::norm(glint - left_glints_centre);
            }
            left_glints_radius /= (int) sorted_left_glints.size();


            glint_locations_distorted_[0] = sorted_left_glints[0];
            glint_validity_[0] = true;
            double current_angle = atan2(sorted_left_glints[1].y - left_glints_centre.y, sorted_left_glints[1].x - left_glints_centre.x);
            double previous_angle = atan2(sorted_left_glints[0].y - left_glints_centre.y, sorted_left_glints[0].x - left_glints_centre.x);

            double expected_angle = current_angle - previous_angle;
            int counter = 1;
            for (int i = 1; counter < leds_per_side; i++) {
                previous_angle = atan2(glint_locations_distorted_[counter - 1].y - left_glints_centre.y, glint_locations_distorted_[counter - 1].x - left_glints_centre.x);
                if (i >= sorted_left_glints.size()) {
                    double angle = previous_angle + expected_angle;
                    glint_locations_distorted_[counter].x = left_glints_centre.x + left_glints_radius * std::cos(angle);
                    glint_locations_distorted_[counter].y = left_glints_centre.y + left_glints_radius * std::sin(angle);
                    glint_validity_[counter] = false;
                    counter++;
                    i--;
                    continue;
                }
                current_angle = atan2(sorted_left_glints[i].y - left_glints_centre.y, sorted_left_glints[i].x - left_glints_centre.x);
                if (current_angle - previous_angle < -M_PI) {
                    previous_angle -= 2 * M_PI;
                }
                if (current_angle - previous_angle > M_PI) {
                    previous_angle += 2 * M_PI;
                }

                if (std::abs(current_angle - previous_angle - expected_angle) < 0.15) {
                    glint_locations_distorted_[counter] = sorted_left_glints[i];
                    glint_validity_[counter] = true;
                    counter++;
                } else {
                    double angle = previous_angle + expected_angle;
                    glint_locations_distorted_[counter].x = left_glints_centre.x + left_glints_radius * std::cos(angle);
                    glint_locations_distorted_[counter].y = left_glints_centre.y + left_glints_radius * std::sin(angle);
                    glint_validity_[counter] = false;
                    counter++;
                    i--;
                }
            }
        }

        {
            std::vector<cv::Point2d> sorted_right_glints{};
            for (int i = 0; i < right_glints; i++) {
                sorted_right_glints.push_back(ellipse_points[left_glints + i]);
            }

            cv::Point2d right_glints_centre = ellipse.center;
            double right_glints_radius = 0;

            // Calculate the radius of the left glints.
            for (const auto& glint: sorted_right_glints) {
                right_glints_radius += cv::norm(glint - right_glints_centre);
            }
            right_glints_radius /= (int) sorted_right_glints.size();


            glint_locations_distorted_[leds_per_side] = sorted_right_glints[0];
            glint_validity_[leds_per_side] = true;
            double current_angle = atan2(sorted_right_glints[1].y - right_glints_centre.y, sorted_right_glints[1].x - right_glints_centre.x);
            double previous_angle = atan2(sorted_right_glints[0].y - right_glints_centre.y, sorted_right_glints[0].x - right_glints_centre.x);

            double expected_angle = current_angle - previous_angle;
            int counter = 1;
            for (int i = 1; counter < leds_per_side; i++) {
                previous_angle = atan2(glint_locations_distorted_[leds_per_side + counter - 1].y - right_glints_centre.y, glint_locations_distorted_[leds_per_side + counter - 1].x - right_glints_centre.x);
                if (i >= sorted_right_glints.size()) {
                    double angle = previous_angle + expected_angle;
                    glint_locations_distorted_[leds_per_side + counter].x = right_glints_centre.x + right_glints_radius * std::cos(angle);
                    glint_locations_distorted_[leds_per_side + counter].y = right_glints_centre.y + right_glints_radius * std::sin(angle);
                    glint_validity_[leds_per_side + counter] = false;
                    counter++;
                    i--;
                    continue;
                }
                current_angle = atan2(sorted_right_glints[i].y - right_glints_centre.y, sorted_right_glints[i].x - right_glints_centre.x);
                if (current_angle - previous_angle < -M_PI) {
                    previous_angle -= 2 * M_PI;
                }
                if (current_angle - previous_angle > M_PI) {
                    previous_angle += 2 * M_PI;
                }

                if (std::abs(current_angle - previous_angle - expected_angle) < 0.15) {
                    glint_locations_distorted_[leds_per_side + counter] = sorted_right_glints[i];
                    glint_validity_[leds_per_side + counter] = true;
                    counter++;
                } else {
                    double angle = previous_angle + expected_angle;
                    glint_locations_distorted_[leds_per_side + counter].x = right_glints_centre.x + right_glints_radius * std::cos(angle);
                    glint_locations_distorted_[leds_per_side + counter].y = right_glints_centre.y + right_glints_radius * std::sin(angle);
                    glint_validity_[leds_per_side + counter] = false;
                    counter++;
                    i--;
                }
            }
        }

        ellipse_points.clear();
        for (int i = 0; i < leds_per_side * 2; i++) {
            if (glint_validity_[i]) {
                ellipse_points.push_back(undistort(glint_locations_distorted_[i]));
            }
        }

        if (ellipse_points.size() < 5) {
            return false;
        }
        ellipse = cv::fitEllipse(ellipse_points);
        glint_ellipse_undistorted_ = ellipse;

//        temporal_filterer_->filterEllipse(glint_ellipse_undistorted_);

        mtx_features_.lock();
        for (int i = 0; i < ellipse_points.size(); i++) {
            glint_locations_undistorted_[i] = undistort(glint_locations_distorted_[i]) + (cv::Point2d) glint_ellipse_undistorted_.center - (cv::Point2d) ellipse.center;
            glint_locations_distorted_[i] = distort(glint_locations_undistorted_[i]);
        }

        glint_represent_undistorted_ = glint_ellipse_undistorted_.center;
        mtx_features_.unlock();

        return true;
    }

    cv::Mat FeatureAnalyser::getThresholdedPupilImage() {
        cv::Mat image_ = pupil_thresholded_image_.clone();
        return image_;
    }

    cv::Mat FeatureAnalyser::getThresholdedGlintsImage() {
        mtx_features_.lock();
        cv::Mat image = glints_thresholded_image_.clone();
        mtx_features_.unlock();
        return image;
    }

    void FeatureAnalyser::getPupilGlintVector(cv::Vec2d& pupil_glint_vector) {
        mtx_features_.lock();
        pupil_glint_vector = pupil_location_undistorted_ - glint_represent_undistorted_;
        mtx_features_.unlock();
    }

    void FeatureAnalyser::getPupilGlintVectorFiltered(cv::Vec2d& pupil_glint_vector) {
        mtx_features_.lock();
        pupil_glint_vector = pupil_location_buffered_ - glint_location_filtered_;
        mtx_features_.unlock();
    }

    cv::RotatedRect FeatureAnalyser::getEllipseUndistorted() {
        return glint_ellipse_undistorted_;
    }

    void FeatureAnalyser::getEllipseUndistorted(cv::RotatedRect& ellipse) {
        mtx_features_.lock();
        ellipse = glint_ellipse_undistorted_;
        mtx_features_.unlock();
    }

    void FeatureAnalyser::getFrameNum(int& frame_num) {
        mtx_features_.lock();
        frame_num = frame_num_;
        mtx_features_.unlock();
    }

    cv::RotatedRect FeatureAnalyser::getEllipseDistorted() {
        return glint_ellipse_distorted_;
    }

    void FeatureAnalyser::setGazeBufferSize(uint8_t value) {
        buffer_size_ = value;
    }

    void FeatureAnalyser::updateGazeBuffer() {
        if (pupil_location_buffer_.size() != buffer_size_ || glint_location_buffer_.size() != buffer_size_) {
            pupil_location_buffer_.resize(buffer_size_);
            glint_location_buffer_.resize(buffer_size_);
            buffer_idx_ = 0;
            buffer_summed_count_ = 0;
            pupil_location_summed_.x = 0.0;
            pupil_location_summed_.y = 0.0;
            glint_location_summed_.x = 0.0;
            glint_location_summed_.y = 0.0;
        }

        if (buffer_summed_count_ == buffer_size_) {
            pupil_location_summed_ -= pupil_location_buffer_[buffer_idx_];
            glint_location_summed_ -= glint_location_buffer_[buffer_idx_];
        }

        pupil_location_buffer_[buffer_idx_] = pupil_location_undistorted_;
        glint_location_buffer_[buffer_idx_] = glint_represent_undistorted_;
        pupil_location_summed_ += pupil_location_buffer_[buffer_idx_];
        glint_location_summed_ += glint_location_buffer_[buffer_idx_];

        if (buffer_summed_count_ != buffer_size_) {
            buffer_summed_count_++;
        }

        mtx_features_.lock();
        pupil_location_buffered_ = pupil_location_summed_ / buffer_summed_count_;
        glint_location_filtered_ = glint_location_summed_ / buffer_summed_count_;
        mtx_features_.unlock();

        buffer_idx_ = (buffer_idx_ + 1) % buffer_size_;
    }
} // namespace et