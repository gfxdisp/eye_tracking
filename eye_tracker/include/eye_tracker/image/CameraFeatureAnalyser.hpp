#ifndef EYE_TRACKER_CAMERAFEATUREANALYSER_HPP
#define EYE_TRACKER_CAMERAFEATUREANALYSER_HPP

#include "eye_tracker/image/FeatureAnalyser.hpp"

namespace et
{

    class CameraFeatureAnalyser : public FeatureAnalyser
    {
    public:
        explicit CameraFeatureAnalyser(int camera_id);

        cv::Point2d undistort(cv::Point2d point) override;

        cv::Point2d distort(cv::Point2d point) override;
    protected:
        // Intrinsic matrix of the camera.
        cv::Mat *intrinsic_matrix_{};
        // Distance from top-left corner of the region-of-interest to the top-left
        // corner of the full image, measured in pixels separately for every axis.
        cv::Size2i *capture_offset_{};
        // Distortion coefficients of the camera.
        std::vector<double> *distortion_coefficients_{};
    };

} // et

#endif //EYE_TRACKER_CAMERAFEATUREANALYSER_HPP
