#ifndef EYE_TRACKER_BLENDERDISCRETEFEATUREANALYSER_HPP
#define EYE_TRACKER_BLENDERDISCRETEFEATUREANALYSER_HPP

#include "eye_tracker/image/FeatureAnalyser.hpp"

namespace et
{

    class BlenderDiscreteFeatureAnalyser : public FeatureAnalyser
    {
    public:
        explicit BlenderDiscreteFeatureAnalyser(int camera_id);

        cv::Point2f undistort(cv::Point2f point) override;

        cv::Point2f distort(cv::Point2f point) override;
    };

} // et

#endif //EYE_TRACKER_BLENDERDISCRETEFEATUREANALYSER_HPP
