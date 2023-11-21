#ifndef HDRMFS_EYE_TRACKER_FEATUREESTIMATOR_HPP
#define HDRMFS_EYE_TRACKER_FEATUREESTIMATOR_HPP

#include <opencv2/core/mat.hpp>

namespace et
{
    class FeatureEstimator
    {
    public:
        FeatureEstimator(int camera_id);

        virtual bool findPupil(cv::Mat &image, cv::Point2d &pupil_position, double &radius) = 0;

        virtual bool findGlints(cv::Mat &image, std::vector<cv::Point2f> &glints) = 0;
    };
} // et


#endif //HDRMFS_EYE_TRACKER_FEATUREESTIMATOR_HPP
