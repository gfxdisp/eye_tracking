#ifndef INPUT_VIDEO_H
#define INPUT_VIDEO_H

#include "ImageProvider.hpp"

#include <opencv2/opencv.hpp>

namespace et
{
	class InputVideo : public ImageProvider
	{
	public:
		InputVideo(std::string &input_video_path);
		virtual void initialize();
		virtual cv::Mat grabImage();
		virtual cv::Size2i getResolution();
		virtual void close();
	private:
		std::string input_video_path_{};
		cv::VideoCapture video_{};
	};
}

#endif