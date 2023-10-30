#include "eye_tracker/frameworks/AnimationFramework.hpp"
#include "eye_tracker/input/InputVideo.hpp"
#include "eye_tracker/image/BlenderContinuousFeatureAnalyser.hpp"
#include "eye_tracker/eye/PolynomialEyeEstimator.hpp"

namespace et
{
    AnimationFramework::AnimationFramework(int camera_id, bool headless, const std::string &input_video_path) : Framework(camera_id, headless)
    {
        image_provider_ = std::make_shared<InputVideo>(input_video_path);
        feature_detector_ = std::make_shared<BlenderContinuousFeatureAnalyser>(camera_id);
        eye_estimator_ = std::make_shared<PolynomialEyeEstimator>(camera_id);
    }
} // et