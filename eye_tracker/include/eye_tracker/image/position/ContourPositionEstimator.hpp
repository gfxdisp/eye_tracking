#ifndef HDRMFS_EYE_TRACKER_CONTOURPOSITIONESTIMATOR_HPP
#define HDRMFS_EYE_TRACKER_CONTOURPOSITIONESTIMATOR_HPP

#include "eye_tracker/image/position/FeatureEstimator.hpp"

namespace et
{

    class ContourPositionEstimator : public FeatureEstimator
    {
    public:
        ContourPositionEstimator(int camera_id);

        bool findPupil(cv::Mat &image, cv::Point2d &pupil_position, double &radius) override;

        bool findGlints(cv::Mat &image, std::vector<cv::Point2f> &glints) override;

    protected:
        // Vectors of all contours that are expected to be pupil or glints in
        // findPupil(), findGlints(), and findEllipse().
        std::vector<std::vector<cv::Point>> contours_{};

        // Centre of the circle aligned with the hole in the view piece in the image.
        cv::Point2d *pupil_search_centre_{};
        // Radius of the circle aligned with the hole in the view piece in the image.
        int *pupil_search_radius_{};
        // Minimal radius of the pupil in pixels.
        double *min_pupil_radius_{};
        // Maximal radius of the pupil in pixels.
        double *max_pupil_radius_{};

        static inline double euclideanDistance(const cv::Point2d &p, const cv::Point2d &q)
        {
            cv::Point2d diff = p - q;
            return cv::sqrt(diff.x * diff.x + diff.y * diff.y);
        }
    };

} // et

#endif //HDRMFS_EYE_TRACKER_CONTOURPOSITIONESTIMATOR_HPP
