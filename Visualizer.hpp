#ifndef VISUALIZER_H
#define VISUALIZER_H

#include "EyeTracker.hpp"
#include "FeatureDetector.hpp"

#include <opencv2/opencv.hpp>

namespace et
{
	class Visualizer
	{
	public:
		Visualizer(FeatureDetector *feature_detector, EyeTracker *eye_tracker);
		void drawUi(cv::Mat image);
		void show();
		void calculateFramerate();
		void printFramerateInterval();
	private:
		static constexpr int FRAMES_FOR_FPS_MEASUREMENT{8};
		cv::Mat image_{};
		FeatureDetector *feature_detector_{};
		EyeTracker *eye_tracker_{};
		std::ostringstream fps_text_{};
		int frame_index_{};
		std::chrono::time_point<std::chrono::steady_clock> last_frame_time_{};
	};
}
#endif