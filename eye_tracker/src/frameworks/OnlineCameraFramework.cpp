#include "eye_tracker/frameworks/OnlineCameraFramework.hpp"
#include "eye_tracker/input/IdsCamera.hpp"
#include "eye_tracker/image/FeatureAnalyser.hpp"
#include "eye_tracker/eye/EyeEstimator.hpp"


#include <memory>

namespace et {
    OnlineCameraFramework::OnlineCameraFramework(int camera_id, const bool headless) : Framework(camera_id, headless) {
        image_provider_ = std::make_shared<IdsCamera>(camera_id);
        feature_detector_ = std::make_shared<FeatureAnalyser>(camera_id);
        eye_estimator_ = std::make_shared<EyeEstimator>(camera_id);
    }
} // et
