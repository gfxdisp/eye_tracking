#ifndef EYE_TRACKER_BLENDERCONTINUOUSFEATUREANALYSER_HPP
#define EYE_TRACKER_BLENDERCONTINUOUSFEATUREANALYSER_HPP

#include "eye_tracker/image/FeatureAnalyser.hpp"

namespace et
{

    class BlenderContinuousFeatureAnalyser : public FeatureAnalyser
    {
    public:
        explicit BlenderContinuousFeatureAnalyser(int camera_id);

        cv::Point2d undistort(cv::Point2d point) override;

        cv::Point2d distort(cv::Point2d point) override;
    };

} // et

#endif //EYE_TRACKER_BLENDERCONTINUOUSFEATUREANALYSER_HPP
