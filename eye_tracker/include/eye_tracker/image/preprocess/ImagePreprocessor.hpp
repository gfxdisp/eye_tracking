#ifndef HDRMFS_EYE_TRACKER_IMAGEPREPROCESSOR_HPP
#define HDRMFS_EYE_TRACKER_IMAGEPREPROCESSOR_HPP

#include "eye_tracker/input/ImageProvider.hpp"

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/opencv.hpp>

namespace et
{

    class ImagePreprocessor
    {
    public:
        ImagePreprocessor(int camera_id);

        virtual void preprocess(const EyeImage &input, EyeImage &output) = 0;
    protected:

        // GPU matrix to which the camera image is directly uploaded.
        cv::cuda::GpuMat gpu_image_{};
        // GPU matrix which contains the thresholded image used for pupil detection
        // with findPupil().
        cv::cuda::GpuMat pupil_thresholded_image_gpu_{};
        // GPU matrix which contains the thresholded image used for glints detection
        // with findGlints() or findEllipse().
        cv::cuda::GpuMat glints_thresholded_image_gpu_{};

        // Threshold value for pupil detection used in preprocessImage().
        int *pupil_threshold_{};
        // Threshold value for glints detection used in preprocessImage().
        int *glint_threshold_{};
    };

} // et

#endif //HDRMFS_EYE_TRACKER_IMAGEPREPROCESSOR_HPP
