#include "eye_tracker/image/preprocess/CameraImagePreprocessor.hpp"

#include <opencv2/cudaarithm.hpp>

namespace et
{
    CameraImagePreprocessor::CameraImagePreprocessor(int camera_id) : ImagePreprocessor(camera_id)
    {
        auto template_path = Settings::settings_folder_ / ("template_" + std::to_string(camera_id) + ".png");
        cv::Mat glints_template_cpu = cv::imread(template_path, cv::IMREAD_GRAYSCALE);
        glints_template_ = cv::cuda::GpuMat(glints_template_cpu.rows, glints_template_cpu.cols, CV_8UC1);
        glints_template_.upload(glints_template_cpu);
        template_matcher_ = cv::cuda::createTemplateMatching(CV_8UC1, cv::TM_CCOEFF);
        template_crop_ = (cv::Mat_<double>(2, 3) << 1, 0, glints_template_.cols / 2, 0, 1, glints_template_.rows / 2);

        close_filter_ = cv::cuda::createMorphologyFilter(cv::MORPH_CLOSE, CV_8UC1, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
        open_filter_ = cv::cuda::createMorphologyFilter(cv::MORPH_OPEN, CV_8UC1, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
    }

    void CameraImagePreprocessor::preprocess(const EyeImage &input, EyeImage &output)
    {
        gpu_image_.upload(input.pupil);

        cv::cuda::threshold(gpu_image_, pupil_thresholded_image_gpu_, *pupil_threshold_, 255, cv::THRESH_BINARY_INV);
//        open_filter_->apply(pupil_thresholded_image_gpu_, pupil_thresholded_image_gpu_);
//        close_filter_->apply(pupil_thresholded_image_gpu_, pupil_thresholded_image_gpu_);

        gpu_image_.upload(input.glints);

        // Finds the correlation of the glint template to every area in the image.
        template_matcher_->match(gpu_image_, glints_template_, glints_thresholded_image_gpu_);

        cv::cuda::threshold(glints_thresholded_image_gpu_, glints_thresholded_image_gpu_, *glint_threshold_ * 2e3, 255,
                            cv::THRESH_BINARY);
        glints_thresholded_image_gpu_.convertTo(glints_thresholded_image_gpu_, CV_8UC1);

        pupil_thresholded_image_gpu_.download(output.pupil);
        glints_thresholded_image_gpu_.download(output.glints);

        cv::Size2i image_size = cv::Size2i(input.pupil.cols, input.pupil.rows);

        // Moves the correlation map so that it is centered in the image.
//        cv::warpAffine(output.glints, output.glints, template_crop_, image_size);
    }
} // et