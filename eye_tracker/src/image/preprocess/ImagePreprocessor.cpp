#include "eye_tracker/image/preprocess/ImagePreprocessor.hpp"

namespace et
{
    ImagePreprocessor::ImagePreprocessor(int camera_id)
    {
        pupil_threshold_ = &Settings::parameters.user_params[camera_id]->pupil_threshold;
        glint_threshold_ = &Settings::parameters.user_params[camera_id]->glint_threshold;

        gpu_image_.create(Settings::parameters.camera_params[camera_id].region_of_interest, CV_8UC1);
        pupil_thresholded_image_gpu_.create(Settings::parameters.camera_params[camera_id].region_of_interest, CV_8UC1);
        glints_thresholded_image_gpu_.create(Settings::parameters.camera_params[camera_id].region_of_interest, CV_8UC1);
    }
} // et