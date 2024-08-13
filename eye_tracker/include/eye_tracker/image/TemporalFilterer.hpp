#ifndef HDRMFS_EYE_TRACKER_TEMPORAL_FILTERER_HPP
#define HDRMFS_EYE_TRACKER_TEMPORAL_FILTERER_HPP

#include <eye_tracker/optimizers/GlintCircleOptimizer.hpp>

#include <vector>
#include <opencv2/core/types.hpp>
#include <opencv2/video/tracking.hpp>

namespace et {
    class TemporalFilterer {
    public:
        explicit TemporalFilterer(int camera_id);

        void filterPupil(cv::Point2d& pupil, double& radius);

        void filterGlints(std::vector<cv::Point2f>& glints);

        ~TemporalFilterer();

        static bool ransac;

    protected:
        static cv::KalmanFilter createPixelKalmanFilter(const cv::Size2i& resolution, double framerate);

        static cv::KalmanFilter createRadiusKalmanFilter(const double& min_radius, const double& max_radius, double framerate);

        cv::KalmanFilter pupil_kalman_{};

        cv::KalmanFilter glints_kalman_{};

        cv::KalmanFilter pupil_radius_kalman_{};

        std::shared_ptr<GlintCircleOptimizer> bayes_minimizer_{};

        cv::Ptr<cv::DownhillSolver::Function> bayes_minimizer_func_{};

        cv::Ptr<cv::DownhillSolver> bayes_solver_{};

        cv::Point2d circle_centre_{};

        double circle_radius_{};

        int camera_id_{};

        static double euclideanDistance(const cv::Point2d& p, const cv::Point2d& q) {
            cv::Point2d diff = p - q;
            return cv::sqrt(diff.x * diff.x + diff.y * diff.y);
        }
    };
} // et

#endif //HDRMFS_EYE_TRACKER_TEMPORAL_FILTERER_HPP
