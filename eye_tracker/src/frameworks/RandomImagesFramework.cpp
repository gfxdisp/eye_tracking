#include "eye_tracker/frameworks/RandomImagesFramework.hpp"
#include "eye_tracker/input/InputImages.hpp"
#include "eye_tracker/eye/PolynomialEyeEstimator.hpp"
#include "eye_tracker/image/BlenderDiscreteFeatureAnalyser.hpp"

namespace et
{
    RandomImagesFramework::RandomImagesFramework(int camera_id, bool headless, const std::string &images_folder_path)
            : Framework(camera_id, headless)
    {
        image_provider_ = std::make_shared<InputImages>(images_folder_path);
        feature_detector_ = std::make_shared<BlenderDiscreteFeatureAnalyser>(camera_id);
        eye_estimator_ = std::make_shared<PolynomialEyeEstimator>(camera_id);
    }
} // et