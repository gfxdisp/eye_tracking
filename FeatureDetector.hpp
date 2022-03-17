#ifndef FEATURE_DETECTOR_H
#define FEATURE_DETECTOR_H

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/opencv.hpp>

#include <mutex>
#include <vector>

namespace et
{
	struct GlintCandidate
	{
		cv::Point2f location;
		float rating;
	};

	class FeatureDetector
	{
	public:
		FeatureDetector(std::string template_path);
		
		void initialize();
		
		void initializeKalmanFilters(cv::Size2i resolution, float framerate);
		
		bool findImageFeatures(cv::Mat image);
		
		cv::Point2f getPupil();
		
		void getPupil(cv::Point2f &pupil);
		
		float getPupilRadius();

		void getPupilRadius(float &pupil_radius);
		
		cv::Point2f *getLeds();
		
		void getLeds(cv::Point2f *leds_locations);
		
		cv::Mat getThresholdedPupilImage();

		cv::Mat getThresholdedGlintsImage();

		static constexpr int LED_COUNT = 2;

		int32_t pupil_threshold{5};

	private:
		bool findPupil();

		bool findGlints();

		cv::KalmanFilter makeKalmanFilter(cv::Size2i resolution, float framerate);

		std::mutex mtx_features_{};

		std::string template_path_{};
		float pupil_radius_{0};
		cv::Point2f pupil_location_{};
		cv::KalmanFilter pupil_kalman_{};
		cv::Point2f leds_locations_[LED_COUNT]{};
		cv::KalmanFilter led_kalmans_[LED_COUNT]{};

		cv::Mat cpu_image_{};
		cv::cuda::GpuMat gpu_image_{};
		cv::cuda::GpuMat pupil_thresholded_image_{};
		cv::cuda::GpuMat glints_thresholded_image_{};

		std::vector<std::vector<cv::Point>> contours_{};
        std::vector<cv::Vec4i> hierarchy_{}; // Unused output

        cv::cuda::GpuMat correlation_map_{};
        cv::Ptr<cv::cuda::TemplateMatching> spots_matcher_{};
        cv::cuda::GpuMat spots_{};
        cv::cuda::Stream stream_spots_{};

		int min_pupil_radius_{20};
		int max_pupil_radius_{90};

		int glints_threshold_{150};
		int max_glint_radius_{5};
		int min_vertical_distance{0};
		int max_vertical_distance{5};
		int min_horizontal_distance{30};
		int max_horizontal_distance{50};

		cv::Ptr<cv::cuda::Filter> dilate_filter_{};
		int dilation_size_{1};

		cv::Ptr<cv::cuda::Filter> erode_filter_{};
		int erosion_size_{5};

		cv::Ptr<cv::cuda::Filter> close_filter_{};
		cv::Ptr<cv::cuda::Filter> open_filter_{};
		int morphology_size{13};

		inline cv::Point2f toPoint(cv::Mat m) 
		{
        	return {(float)m.at<double>(0, 0), (float)m.at<double>(0, 1)};
    	}

    	inline float euclideanDistance(cv::Point2f& p, cv::Point2f& q) 
    	{
    		cv::Point2f diff = p - q;
    		return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
		}

	};
}

#endif