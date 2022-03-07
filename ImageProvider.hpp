#ifndef IMAGE_PROVIDER_H
#define IMAGE_PROVIDER_H

#include <opencv2/opencv.hpp>

namespace et
{
	class ImageProvider
	{
	public:
		virtual void initialize() = 0;
		virtual cv::Mat grabImage() = 0;
		virtual cv::Size2i getResolution() = 0;
		virtual void close() = 0;
		virtual double getPixelPitch();
		virtual void setExposure(double exposure);
		virtual void setGamma(float gamma);
		virtual void setFramerate(double framerate);
	protected:
		cv::Size2i image_resolution_{};
		cv::Mat image_{};
		double pixel_pitch_{0.0048};
	};
}

#endif