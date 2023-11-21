#ifndef HDRMFS_EYE_TRACKER_TEMPORALFILTERER_HPP
#define HDRMFS_EYE_TRACKER_TEMPORALFILTERER_HPP

#include <vector>
#include <opencv2/core/types.hpp>

namespace et
{

    class TemporalFilterer
    {
    public:
        TemporalFilterer(int camera_id);

        virtual void filterPupil(cv::Point2d &pupil, double &radius) = 0;

        virtual void filterGlints(std::vector<cv::Point2f> &glints) = 0;

        virtual void filterEllipse(cv::RotatedRect &ellipse) = 0;


    };

} // et

#endif //HDRMFS_EYE_TRACKER_TEMPORALFILTERER_HPP
