#include "InputVideo.hpp"

namespace et
{
	InputVideo::InputVideo(std::string &input_video_path) : input_video_path_(std::move(input_video_path))
	{
	}

	void InputVideo::initialize()
	{
		assert(input_video_path_.length() > 0);
		
		video_.open(input_video_path_);
		assert(video_.isOpened());

		image_resolution_.width = video_.get(cv::CAP_PROP_FRAME_WIDTH);
		image_resolution_.height = video_.get(cv::CAP_PROP_FRAME_HEIGHT);
	}

	cv::Mat InputVideo::grabImage()
	{
		bool successful{};
		successful = video_.read(image_);
		if (!successful)
		{
			video_.set(cv::CAP_PROP_POS_FRAMES, 0);
			successful = video_.read(image_);
		}
		assert(successful);
		cv::cvtColor(image_, image_, cv::COLOR_BGR2GRAY);
		return image_;
	}

	cv::Size2i InputVideo::getResolution()
	{
		return image_resolution_;
	}

	void InputVideo::close()
	{
		video_.release();
	}
}