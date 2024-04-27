#include "eye_tracker/frameworks/VideoCameraFramework.hpp"
#include "eye_tracker/input/InputVideo.hpp"
#include "eye_tracker/image/CameraFeatureAnalyser.hpp"
#include "eye_tracker/eye/PolynomialEyeEstimator.hpp"

namespace et
{
    VideoCameraFramework::VideoCameraFramework(int camera_id, bool headless, const std::string &input_video_path, bool loop) : Framework(camera_id, headless)
    {
        image_provider_ = std::make_shared<InputVideo>(input_video_path, loop);
        feature_detector_ = std::make_shared<CameraFeatureAnalyser>(camera_id);
        eye_estimator_ = std::make_shared<ModelEyeEstimator>(camera_id);
    }
} // et