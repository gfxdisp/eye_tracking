#include "eye_tracker/frameworks/OnlineCameraFramework.hpp"
#include "eye_tracker/input/IdsCamera.hpp"
#include "eye_tracker/eye/PolynomialEyeEstimator.hpp"
#include "eye_tracker/image/CameraFeatureAnalyser.hpp"
#include "eye_tracker/eye/ModelEyeEstimator.hpp"


#include <memory>

namespace et
{
    OnlineCameraFramework::OnlineCameraFramework(int camera_id, bool headless) : Framework(camera_id, headless)
    {
        image_provider_ = std::make_shared<IdsCamera>(camera_id);
        feature_detector_ = std::make_shared<CameraFeatureAnalyser>(camera_id);
        eye_estimator_ = std::make_shared<ModelEyeEstimator>(camera_id);
    }
} // et