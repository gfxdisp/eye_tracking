/// Class containing all information about the IDS Camera used for eye-tracking. 
/// It serves as an interface for grabbing images and setting camera parameters.

#ifndef IDS_CAMERA_H
#define IDS_CAMERA_H

#include "ImageProvider.hpp"

#include <uEye.h>

#include <opencv2/opencv.hpp>

#include <mutex>
#include <string>
#include <thread>

namespace et 
{
	class IdsCamera : public ImageProvider
	{
	public:
		IdsCamera(int camera_index);
		virtual void initialize();
		virtual cv::Mat grabImage();
		virtual cv::Size2i getResolution();
		virtual void close();
		virtual void setExposure(double exposure);
		virtual void setGamma(float gamma);
		virtual void setFramerate(double framerate);

	private:
		void initializeCamera();
		void initializeImage();
		void imageGatheringThread();

		static constexpr uint32_t IMAGE_IN_QUEUE_COUNT = 10;

		std::mutex mtx_image_{};

		cv::Mat image_queue_[IMAGE_IN_QUEUE_COUNT]{};
		int32_t image_index_{-1};

		bool thread_running_{true};

		std::thread image_gatherer_{};

		int32_t camera_index_{};
		int32_t n_cameras_{};
		PUEYE_CAMERA_LIST camera_list_{};
		uint32_t camera_handle_{};
		SENSORINFO sensor_info_{};
		char* image_handle_{};
		int32_t image_id_{};
	};
}

#endif