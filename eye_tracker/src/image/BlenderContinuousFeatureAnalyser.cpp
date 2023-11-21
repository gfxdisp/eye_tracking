#include "eye_tracker/image/BlenderContinuousFeatureAnalyser.hpp"
#include "eye_tracker/image/preprocess/BlenderImagePreprocessor.hpp"
#include "eye_tracker/image/temporal_filter/ContinuousTemporalFilterer.hpp"
#include "eye_tracker/image/position/ContourPositionEstimator.hpp"

namespace et
{
    BlenderContinuousFeatureAnalyser::BlenderContinuousFeatureAnalyser(int camera_id) : FeatureAnalyser(camera_id)
    {
        image_preprocessor_ = std::make_shared<BlenderImagePreprocessor>(camera_id);
        temporal_filterer_ = std::make_shared<ContinuousTemporalFilterer>(camera_id);
        position_estimator_ = std::make_shared<ContourPositionEstimator>(camera_id);
    }

    cv::Point2d BlenderContinuousFeatureAnalyser::undistort(cv::Point2d point)
    {
        return point;
    }

    cv::Point2d BlenderContinuousFeatureAnalyser::distort(cv::Point2d point)
    {
        return point;
    }
} // et