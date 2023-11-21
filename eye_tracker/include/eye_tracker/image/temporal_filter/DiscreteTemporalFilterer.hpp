#ifndef HDRMFS_EYE_TRACKER_DISCRETETEMPORALFILTERER_HPP
#define HDRMFS_EYE_TRACKER_DISCRETETEMPORALFILTERER_HPP

#include "eye_tracker/image/temporal_filter/TemporalFilterer.hpp"

namespace et
{

    class DiscreteTemporalFilterer : public TemporalFilterer
    {
    public:
        DiscreteTemporalFilterer(int camera_id);

        void filterPupil(cv::Point2d &pupil, double &radius) override;

        void filterGlints(std::vector<cv::Point2f> &glints) override;

        void filterEllipse(cv::RotatedRect &ellipse) override;
    };

} // et

#endif //HDRMFS_EYE_TRACKER_DISCRETETEMPORALFILTERER_HPP
