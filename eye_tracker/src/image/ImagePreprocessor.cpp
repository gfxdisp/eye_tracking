#include <eye_tracker/image/ImagePreprocessor.hpp>

#include <opencv2/cudaarithm.hpp>

namespace et {
    ImagePreprocessor::ImagePreprocessor(int camera_id) : camera_id_(camera_id) {
        gpu_image_.create(Settings::parameters.camera_params[camera_id].region_of_interest, CV_8UC1);
        pupil_thresholded_image_gpu_.create(Settings::parameters.camera_params[camera_id].region_of_interest, CV_8UC1);
        glints_thresholded_image_gpu_.create(Settings::parameters.camera_params[camera_id].region_of_interest, CV_8UC1);

        auto template_path = Settings::settings_folder_ / ("template_" + std::to_string(camera_id) + ".png");
        cv::Mat glints_template_cpu = cv::imread(template_path, cv::IMREAD_GRAYSCALE);
        glints_template_ = cv::cuda::GpuMat(glints_template_cpu.rows, glints_template_cpu.cols, CV_8UC1);
        glints_template_.upload(glints_template_cpu);
        template_matcher_ = cv::cuda::createTemplateMatching(CV_8UC1, cv::TM_CCOEFF);
        template_crop_ = (cv::Mat_<double>(2, 3) << 1, 0, glints_template_.cols / 2, 0, 1, glints_template_.rows / 2);
    }

    void ImagePreprocessor::preprocess(const EyeImage& eye_image, cv::Mat& thresholded_pupil, cv::Mat& thresholded_glints) {
        const int& pupil_threshold = Settings::parameters.user_params[camera_id_]->pupil_threshold;
        const int& glint_threshold = Settings::parameters.user_params[camera_id_]->glint_threshold;

        gpu_image_.upload(eye_image.frame);
        cv::cuda::threshold(gpu_image_, pupil_thresholded_image_gpu_, pupil_threshold, 255, cv::THRESH_BINARY_INV);

        gpu_image_.upload(eye_image.frame);
        template_matcher_->match(gpu_image_, glints_template_, glints_thresholded_image_gpu_);
        cv::cuda::threshold(glints_thresholded_image_gpu_, glints_thresholded_image_gpu_, glint_threshold * 2e3, 255, cv::THRESH_BINARY);
        glints_thresholded_image_gpu_.convertTo(glints_thresholded_image_gpu_, CV_8UC1);

        pupil_thresholded_image_gpu_.download(thresholded_pupil);
        glints_thresholded_image_gpu_.download(thresholded_glints);
    }
} // et
