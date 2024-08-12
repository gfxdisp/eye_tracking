#ifndef HDRMFS_EYE_TRACKER_IMAGEPREPROCESSOR_HPP
#define HDRMFS_EYE_TRACKER_IMAGEPREPROCESSOR_HPP

#include "eye_tracker/input/ImageProvider.hpp"

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/opencv.hpp>

namespace et {
    class ImagePreprocessor {
    public:
        explicit ImagePreprocessor(int camera_id);

        void preprocess(const EyeImage& eye_image, cv::Mat& thresholded_pupil, cv::Mat& thresholded_glints);

    protected:
        cv::cuda::GpuMat gpu_image_{};

        cv::cuda::GpuMat pupil_thresholded_image_gpu_{};

        cv::cuda::GpuMat glints_thresholded_image_gpu_{};

        cv::cuda::GpuMat glints_template_;

        cv::Ptr<cv::cuda::TemplateMatching> template_matcher_{};

        cv::Mat template_crop_{};

        int camera_id_{};
    };
} // et

#endif //HDRMFS_EYE_TRACKER_IMAGEPREPROCESSOR_HPP
