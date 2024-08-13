#ifndef HDRMFS_EYE_TRACKER_FEATURE_ESTIMATOR_HPP
#define HDRMFS_EYE_TRACKER_FEATURE_ESTIMATOR_HPP

namespace et {
    class FeatureEstimator {
    public:
        explicit FeatureEstimator(int camera_id);

        bool findPupil(const cv::Mat& image, cv::Point2d& pupil_position, double& radius);

        bool findGlints(const cv::Mat& image, std::vector<cv::Point2f>& glints);

    protected:
        std::vector<std::vector<cv::Point> > contours_{};

        cv::Point2d pupil_search_centre_;

        int pupil_search_radius_;

        double min_pupil_radius_;

        double max_pupil_radius_;

        static double euclideanDistance(const cv::Point2d& p, const cv::Point2d& q) {
            cv::Point2d diff = p - q;
            return cv::sqrt(diff.x * diff.x + diff.y * diff.y);
        }

        cv::Size2i template_size_{};
    };
} // et


#endif //HDRMFS_EYE_TRACKER_FEATURE_ESTIMATOR_HPP
