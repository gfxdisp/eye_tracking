#ifndef HDRMFS_EYE_TRACKER_CONTINUOUSTEMPORALFILTERER_HPP
#define HDRMFS_EYE_TRACKER_CONTINUOUSTEMPORALFILTERER_HPP

#include "eye_tracker/image/temporal_filter/TemporalFilterer.hpp"
#include "eye_tracker/optimizers/GlintCircleOptimizer.hpp"

#include <opencv2/video/tracking.hpp>

namespace et
{

    class ContinuousTemporalFilterer : public TemporalFilterer
    {
    public:
        ContinuousTemporalFilterer(int camera_id);

        void filterPupil(cv::Point2d &pupil, double &radius) override;

        void filterGlints(std::vector<cv::Point2f> &glints) override;

        void filterEllipse(cv::RotatedRect &ellipse) override;

        ~ContinuousTemporalFilterer();

    protected:
        /**
         * Creates a 4x4 Kalman Filter assuming its input vector consists of
         * XY pixel position and XY velocity.
         * @param resolution Dimensions of the image space.
         * @param framerate Estimated system framerate used to calculate velocity.
         * @return Kalman filter.
         */
        cv::KalmanFilter createPixelKalmanFilter(const cv::Size2i &resolution, double framerate);

        /**
         * Creates a 2x2 Kalman Filter assuming its input vector consists of
         * a pixel pupil radius and velocity.
         * @param min_radius Minimum expected radius of the pupil in pixels.
         * @param max_radius Maximum expected radius of the pupil in pixels.
         * @param framerate Estimated system framerate used to calculate velocity.
         * @return Kalman filter.
         */
        cv::KalmanFilter createRadiusKalmanFilter(const double &min_radius, const double &max_radius, double framerate);

        /**
         * Creates a 10x10 Kalman Filter assuming its input vector consists of
         * a XY ellipse centre, ellipse width, ellipse height, ellipse angle,
         * and velocity of all mentioned parameters.
         * @param resolution Dimensions of the image space.
         * @param framerate Estimated system framerate used to calculate velocity.
         * @return Kalman filter.
         */
        cv::KalmanFilter createEllipseKalmanFilter(const cv::Size2i &resolution, double framerate);

        // Kalman filter used to correct noisy pupil position.
        cv::KalmanFilter pupil_kalman_{};
        // Kalman filter used to correct noise glint positions.
        cv::KalmanFilter glints_kalman_{};
        // Kalman filter used to correct noisy pupil radius.
        cv::KalmanFilter pupil_radius_kalman_{};
        // Kalman filter used to correct noisy glint ellipse parameters.
        cv::KalmanFilter glint_ellipse_kalman_{};

        // Function used to find a circle based on a set of glints and its
        // expected position.
        GlintCircleOptimizer *bayes_minimizer_{};
        // Pointer to a BayesMinimizer function.
        cv::Ptr<cv::DownhillSolver::Function> bayes_minimizer_func_{};
        // Downhill solver optimizer used to find a glint circle.
        cv::Ptr<cv::DownhillSolver> bayes_solver_{};
        // Expected position of a circle on which glints lie.
        cv::Point2d circle_centre_{};
        // Expected radius of a circle on which glints lie.
        double circle_radius_{};

        cv::Size2i *region_of_interest_{};

        /**
         * Converts cv::Mat to cv::Point2f
         * @param m Matrix to be converted.
         * @return Converted point.
         */
        static inline cv::Point2d toPoint(cv::Mat m)
        {
            return {m.at<double>(0, 0), m.at<double>(0, 1)};
        }

        /**
         * Extracts a single value from cv::Mat.
         * @param m Matrix with a value to extract.
         * @return Extracted value.
         */
        static inline double toValue(cv::Mat m)
        {
            return m.at<double>(0, 0);
        }

        static inline double euclideanDistance(const cv::Point2d &p, const cv::Point2d &q)
        {
            cv::Point2d diff = p - q;
            return cv::sqrt(diff.x * diff.x + diff.y * diff.y);
        }
    };

} // et

#endif //HDRMFS_EYE_TRACKER_CONTINUOUSTEMPORALFILTERER_HPP
