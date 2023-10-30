#include "eye_tracker/image/preprocess/BlenderImagePreprocessor.hpp"

#include <opencv2/cudaarithm.hpp>

namespace et
{
    BlenderImagePreprocessor::BlenderImagePreprocessor(int camera_id) : ImagePreprocessor(camera_id)
    {
    }

    void BlenderImagePreprocessor::preprocess(const EyeImage &input, EyeImage &output)
    {
        gpu_image_.upload(input.pupil);

        cv::cuda::threshold(gpu_image_, pupil_thresholded_image_gpu_, *pupil_threshold_, 255, cv::THRESH_BINARY_INV);

        gpu_image_.upload(input.glints);
        cv::cuda::threshold(gpu_image_, glints_thresholded_image_gpu_, *glint_threshold_, 255, cv::THRESH_BINARY);

        pupil_thresholded_image_gpu_.download(output.pupil);
        glints_thresholded_image_gpu_.download(output.glints);
    }
} // et
