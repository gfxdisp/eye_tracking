#include "eye_tracker/image/FeatureAnalyser.hpp"
#include "eye_tracker/image/position/ContourPositionEstimator.hpp"

#include <opencv2/cudaimgproc.hpp>
#include <memory>
#include <iostream>

namespace et
{
    FeatureAnalyser::FeatureAnalyser(int camera_id)
    {
        glint_locations_distorted_.resize(Settings::parameters.leds_positions[camera_id].size());
        glint_locations_undistorted_.resize(Settings::parameters.leds_positions[camera_id].size());
    }

    cv::Point2d FeatureAnalyser::getPupilUndistorted()
    {
        return pupil_location_undistorted_;
    }

    cv::Point2d FeatureAnalyser::getPupilDistorted()
    {
        return pupil_location_distorted_;
    }

    void FeatureAnalyser::getPupilUndistorted(cv::Point2d &pupil)
    {
        mtx_features_.lock();
        pupil = pupil_location_undistorted_;
        mtx_features_.unlock();
    }

    void FeatureAnalyser::getPupilBuffered(cv::Point2d &pupil)
    {
        mtx_features_.lock();
        pupil = pupil_location_buffered_;
        mtx_features_.unlock();
    }

    int FeatureAnalyser::getPupilRadiusUndistorted() const
    {
        return pupil_radius_undistorted_;
    }

    int FeatureAnalyser::getPupilRadiusDistorted() const
    {
        return pupil_radius_distorted_;
    }

    std::vector<cv::Point2d> *FeatureAnalyser::getGlints()
    {
        return &glint_locations_undistorted_;
    }

    std::vector<cv::Point2d> *FeatureAnalyser::getDistortedGlints()
    {
        return &glint_locations_distorted_;
    }

    void FeatureAnalyser::preprocessImage(const EyeImage &image)
    {
        EyeImage output{.pupil = cv::Mat(image.pupil.rows, image.pupil.cols, CV_8UC1),
                        .glints = cv::Mat(image.glints.rows, image.glints.cols, CV_8UC1)};
        image_preprocessor_->preprocess(image, output);
        pupil_thresholded_image_ = output.pupil;
        glints_thresholded_image_ = output.glints;
        frame_num_ = image.frame_num;
    }

    bool FeatureAnalyser::findPupil()
    {
        cv::Point2d estimated_pupil_location{};
        double estimated_pupil_radius{};
        bool success = position_estimator_->findPupil(pupil_thresholded_image_, estimated_pupil_location,
                                                      estimated_pupil_radius);

        if (!success)
        {
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

    bool FeatureAnalyser::findEllipsePoints()
    {
        std::vector<cv::Point2f> ellipse_points{};
        bool success = position_estimator_->findGlints(glints_thresholded_image_, ellipse_points);
        if (!success)
        {
            return false;
        }

        temporal_filterer_->filterGlints(ellipse_points);
        if (ellipse_points.size() < 8)
        {
            return false;
        }

        glint_ellipse_distorted_ = cv::fitEllipse(ellipse_points);

        glint_locations_distorted_[0] = glint_ellipse_distorted_.center;
        glint_locations_distorted_[1] = glint_ellipse_distorted_.center;

        for (auto &point: ellipse_points)
        {
            if (point.x < glint_ellipse_distorted_.center.x && point.y < glint_locations_distorted_[0].y)
            {
                glint_locations_distorted_[0] = point;
            }
            else if (point.x > glint_ellipse_distorted_.center.x && point.y > glint_locations_distorted_[1].y)
            {
                glint_locations_distorted_[1] = point;
            }
        }

        for (auto &point: ellipse_points)
        {
            point = undistort(point);
        }

        glint_ellipse_undistorted_ = cv::fitEllipse(ellipse_points);

        temporal_filterer_->filterEllipse(glint_ellipse_undistorted_);

        mtx_features_.lock();
        for (int i = 0; i < 2; i++)
        {
            glint_locations_undistorted_[i] =
                    (cv::Point2d) glint_ellipse_undistorted_.center - (cv::Point2d) glint_ellipse_distorted_.center + glint_locations_distorted_[i];
        }
        glint_represent_undistorted_ = glint_ellipse_undistorted_.center;
        mtx_features_.unlock();

        return true;
    }

    cv::Mat FeatureAnalyser::getThresholdedPupilImage()
    {
        cv::Mat image_ = pupil_thresholded_image_.clone();
        return image_;
    }

    cv::Mat FeatureAnalyser::getThresholdedGlintsImage()
    {
        mtx_features_.lock();
        cv::Mat image = glints_thresholded_image_.clone();
        mtx_features_.unlock();
        return image;
    }

    void FeatureAnalyser::getPupilGlintVector(cv::Vec2d &pupil_glint_vector)
    {
        mtx_features_.lock();
        pupil_glint_vector = pupil_location_undistorted_ - glint_represent_undistorted_;
        mtx_features_.unlock();
    }

    void FeatureAnalyser::getPupilGlintVectorFiltered(cv::Vec2d &pupil_glint_vector)
    {
        mtx_features_.lock();
        pupil_glint_vector = pupil_location_buffered_ - glint_location_filtered_;
        mtx_features_.unlock();
    }

    cv::RotatedRect FeatureAnalyser::getEllipseUndistorted()
    {
        return glint_ellipse_undistorted_;
    }

    void FeatureAnalyser::getEllipseUndistorted(cv::RotatedRect &ellipse)
    {
        mtx_features_.lock();
        ellipse = glint_ellipse_undistorted_;
        mtx_features_.unlock();
    }

    void FeatureAnalyser::getFrameNum(int &frame_num)
    {
        mtx_features_.lock();
        frame_num = frame_num_;
        mtx_features_.unlock();
    }

    cv::RotatedRect FeatureAnalyser::getEllipseDistorted()
    {
        return glint_ellipse_distorted_;
    }

    void FeatureAnalyser::setGazeBufferSize(uint8_t value)
    {
        buffer_size_ = value;
    }

    void FeatureAnalyser::updateGazeBuffer()
    {
        if (pupil_location_buffer_.size() != buffer_size_ || glint_location_buffer_.size() != buffer_size_)
        {
            pupil_location_buffer_.resize(buffer_size_);
            glint_location_buffer_.resize(buffer_size_);
            buffer_idx_ = 0;
            buffer_summed_count_ = 0;
            pupil_location_summed_.x = 0.0;
            pupil_location_summed_.y = 0.0;
            glint_location_summed_.x = 0.0;
            glint_location_summed_.y = 0.0;
        }

        if (buffer_summed_count_ == buffer_size_)
        {
            pupil_location_summed_ -= pupil_location_buffer_[buffer_idx_];
            glint_location_summed_ -= glint_location_buffer_[buffer_idx_];
        }

        pupil_location_buffer_[buffer_idx_] = pupil_location_undistorted_;
        glint_location_buffer_[buffer_idx_] = glint_represent_undistorted_;
        pupil_location_summed_ += pupil_location_buffer_[buffer_idx_];
        glint_location_summed_ += glint_location_buffer_[buffer_idx_];

        if (buffer_summed_count_ != buffer_size_)
        {
            buffer_summed_count_++;
        }

        mtx_features_.lock();
        pupil_location_buffered_ = pupil_location_summed_ / buffer_summed_count_;
        glint_location_filtered_ = glint_location_summed_ / buffer_summed_count_;
        mtx_features_.unlock();

        buffer_idx_ = (buffer_idx_ + 1) % buffer_size_;
    }
} // namespace et