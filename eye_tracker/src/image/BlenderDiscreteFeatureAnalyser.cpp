#include "eye_tracker/image/BlenderDiscreteFeatureAnalyser.hpp"
#include "eye_tracker/image/preprocess/BlenderImagePreprocessor.hpp"
#include "eye_tracker/image/temporal_filter/DiscreteTemporalFilterer.hpp"
#include "eye_tracker/image/position/ContourPositionEstimator.hpp"

namespace et
{
    BlenderDiscreteFeatureAnalyser::BlenderDiscreteFeatureAnalyser(int camera_id) : FeatureAnalyser(camera_id)
    {
        image_preprocessor_ = std::make_shared<BlenderImagePreprocessor>(camera_id);
        temporal_filterer_ = std::make_shared<DiscreteTemporalFilterer>(camera_id);
        position_estimator_ = std::make_shared<ContourPositionEstimator>(camera_id);
    }

    cv::Point2d BlenderDiscreteFeatureAnalyser::undistort(cv::Point2d point)
    {
        return point;
    }

    cv::Point2d BlenderDiscreteFeatureAnalyser::distort(cv::Point2d point)
    {
        return point;
    }
} // et