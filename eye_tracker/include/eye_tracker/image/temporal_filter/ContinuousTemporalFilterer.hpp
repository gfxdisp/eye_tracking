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

        void filterPupil(cv::Point2f &pupil, float &radius) override;

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
        cv::KalmanFilter createPixelKalmanFilter(const cv::Size2i &resolution, float framerate);

        /**
         * Creates a 2x2 Kalman Filter assuming its input vector consists of
         * a pixel pupil radius and velocity.
         * @param min_radius Minimum expected radius of the pupil in pixels.
         * @param max_radius Maximum expected radius of the pupil in pixels.
         * @param framerate Estimated system framerate used to calculate velocity.
         * @return Kalman filter.
         */
        cv::KalmanFilter createRadiusKalmanFilter(const float &min_radius, const float &max_radius, float framerate);

        /**
         * Creates a 10x10 Kalman Filter assuming its input vector consists of
         * a XY ellipse centre, ellipse width, ellipse height, ellipse angle,
         * and velocity of all mentioned parameters.
         * @param resolution Dimensions of the image space.
         * @param framerate Estimated system framerate used to calculate velocity.
         * @return Kalman filter.
         */
        cv::KalmanFilter createEllipseKalmanFilter(const cv::Size2i &resolution, float framerate);

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
        static inline cv::Point2f toPoint(cv::Mat m)
        {
            return {(float) m.at<double>(0, 0), (float) m.at<double>(0, 1)};
        }

        /**
         * Extracts a single value from cv::Mat.
         * @param m Matrix with a value to extract.
         * @return Extracted value.
         */
        static inline float toValue(cv::Mat m)
        {
            return (float) m.at<double>(0, 0);
        }

        static inline float euclideanDistance(const cv::Point2f &p, const cv::Point2f &q)
        {
            cv::Point2f diff = p - q;
            return cv::sqrt(diff.x * diff.x + diff.y * diff.y);
        }
    };

} // et

#endif //HDRMFS_EYE_TRACKER_CONTINUOUSTEMPORALFILTERER_HPP
