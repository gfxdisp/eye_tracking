#ifndef HDRMFS_EYE_TRACKER_CAMERAIMAGEPREPROCESSOR_HPP
#define HDRMFS_EYE_TRACKER_CAMERAIMAGEPREPROCESSOR_HPP

#include "eye_tracker/image/preprocess/ImagePreprocessor.hpp"

namespace et
{

    class CameraImagePreprocessor : public ImagePreprocessor
    {
    public:
        CameraImagePreprocessor(int camera_id);

        void preprocess(const EyeImage &input, EyeImage &output) override;

    protected:
        // Template uploaded to the GPU used to detect glints with preprocessImage().
        cv::cuda::GpuMat glints_template_;
        // Template matching algorithm used to detect glints with preprocessImage().
        cv::Ptr<cv::cuda::TemplateMatching> template_matcher_{};
        // Affine warping matrix used to translate correlation matrix to align with
        // the original image.
        cv::Mat template_crop_{};
    };

} // et

#endif //HDRMFS_EYE_TRACKER_CAMERAIMAGEPREPROCESSOR_HPP
