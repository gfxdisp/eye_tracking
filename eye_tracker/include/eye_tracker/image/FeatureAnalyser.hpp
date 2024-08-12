#ifndef HDRMFS_EYE_TRACKER_FEATURE_ANALYSER_HPP
#define HDRMFS_EYE_TRACKER_FEATURE_ANALYSER_HPP

#include <eye_tracker/input/ImageProvider.hpp>
#include <eye_tracker/image/FeatureEstimator.hpp>
#include <eye_tracker/image/ImagePreprocessor.hpp>
#include <eye_tracker/image/TemporalFilterer.hpp>

#include <mutex>
#include <vector>


namespace et {
    class FeatureAnalyser {
    public:
        explicit FeatureAnalyser(int camera_id);

        void preprocessImage(const EyeImage& image);

        bool findPupil();

        bool findEllipsePoints();

        void getPupilUndistorted(cv::Point2d& pupil_position_undistorted);

        void getPupilDistorted(cv::Point2d& pupil_location_distorted);

        void getPupilRadiusUndistorted(double& pupil_radius_undistorted);

        void getPupilRadiusDistorted(double& pupil_radius_distorted);

        void getGlints(std::vector<cv::Point2d>& glints);

        void getDistortedGlints(std::vector<cv::Point2d>& glints);

        void getGlintsValidity(std::vector<bool>& glint_validity);

        void getThresholdedPupilImage(cv::Mat& image);

        void getThresholdedGlintsImage(cv::Mat& image);

        cv::Point2d undistort(cv::Point2d point) const;

        cv::Point2d distort(cv::Point2d point) const;

    protected:
        int camera_id_{};

        std::shared_ptr<ImagePreprocessor> image_preprocessor_{};

        std::shared_ptr<TemporalFilterer> temporal_filterer_{};

        std::shared_ptr<FeatureEstimator> feature_estimator_{};

        std::mutex mtx_features_{};

        cv::Point2d pupil_location_distorted_{};

        cv::Point2d pupil_location_undistorted_{};

        double pupil_radius_distorted_{0};

        double pupil_radius_undistorted_{0};

        std::vector<cv::Point2d> glint_locations_distorted_{};

        std::vector<cv::Point2d> glint_locations_undistorted_{};

        std::vector<bool> glint_validity_{};

        int frame_num_{};

        cv::Mat thresholded_pupil_image_;

        cv::Mat thresholded_glints_image_;

        cv::Mat* intrinsic_matrix_{};

        cv::Size2i* capture_offset_{};

        std::vector<double>* distortion_coefficients_{};
    };
} // namespace et

#endif //HDRMFS_EYE_TRACKER_FEATURE_ANALYSER_HPP
