#include "Visualizer.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>

using namespace std::chrono_literals;

namespace et
{
	Visualizer::Visualizer(FeatureDetector *feature_detector, EyeTracker *eye_tracker) : feature_detector_(feature_detector), eye_tracker_(eye_tracker)
	{
		last_frame_time_ = std::chrono::steady_clock::now();
	    fps_text_ << std::fixed << std::setprecision(2);
	}

	void Visualizer::drawUi(cv::Mat image)
	{
		cv::cvtColor(image, image_, cv::COLOR_GRAY2BGR);
		cv::Point2f* led_positions{feature_detector_->getLeds()};
		for (int i = 0; i < FeatureDetector::LED_COUNT; i++)
		{
            cv::circle(image_, led_positions[i], 5, cv::Scalar(0x00, 0x00, 0xFF), 2);
		}
        cv::circle(image_, feature_detector_->getPupil(), feature_detector_->getPupilRadius(), cv::Scalar(0xFF, 0x00, 0x00), 5);

		cv::circle(image_, eye_tracker_->getCorneaCurvaturePixelPosition(), 3, cv::Scalar(0x00, 0xFF, 0x00), 5);

        cv::putText(image_, fps_text_.str(), cv::Point2i(100, 100), cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(0x00, 0x00, 0xFF), 3);
	}

	void Visualizer::show()
	{
		cv::imshow("Output", image_);
	}

	void Visualizer::calculateFramerate()
	{
		if (++frame_index_ == FRAMES_FOR_FPS_MEASUREMENT) 
	 	{
            const std::chrono::duration<float> frame_time = std::chrono::steady_clock::now() - last_frame_time_;
            fps_text_.str(""); // Clear contents of fps_text
            fps_text_ << 1s / (frame_time / FRAMES_FOR_FPS_MEASUREMENT);
            frame_index_ = 0;
            last_frame_time_ = std::chrono::steady_clock::now();
        }
	}

	void Visualizer::printFramerateInterval()
	{
		static auto start{std::chrono::steady_clock::now()};
		auto current{std::chrono::steady_clock::now()};
		if (current - start > 1s)
		{
			start = current;
			std::clog << "Frames per second: " << fps_text_.str() << "\n";
		}
	}
}