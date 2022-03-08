#include "EyeTracker.hpp"
#include "FeatureDetector.hpp"
#include "IdsCamera.hpp"
#include "ImageProvider.hpp"
#include "InputVideo.hpp"
#include "SocketServer.hpp"
#include "Visualizer.hpp"

#include <opencv2/opencv.hpp>

#include <cassert>
#include <iostream>
#include <thread>

enum class VisualizationType
{
	DISABLED,
	STANDARD,
	THRESHOLD_PUPIL,
	THRESHOLD_GLINTS
};

int main(int argc, char *argv[])
{
	assert(argc >= 3 && argc <= 4);
	std::string input_type{argv[1]};

	et::ImageProvider *image_provider{};
	if (input_type == "ids")
	{
		int camera_index{std::stoi(std::string(argv[2]))};
		image_provider = new et::IdsCamera(camera_index);
	}
	else if (input_type == "file")
	{
		std::string input_file{argv[2]};
		image_provider = new et::InputVideo(input_file);
	}
	else {
		return EXIT_FAILURE;
	}

	float framerate{90.0f};

	image_provider->initialize();
	image_provider->setGamma(3.5f);
	image_provider->setFramerate(framerate);

	et::FeatureDetector feature_detector{"template.png"};
	feature_detector.initialize();
	feature_detector.initializeKalmanFilters(image_provider->getResolution(), framerate);

	et::SetupLayout setup_layout{};
	setup_layout.camera_lambda = 27.119;
	setup_layout.camera_nodal_point_position = {187.044, 129.307, 484.442};
	setup_layout.led_positions[0] = {232.464, 151.985, 747.823};
	setup_layout.led_positions[1] = {233.252, 193.252, 749.389};
	setup_layout.camera_eye_distance = 316.29;
	setup_layout.camera_eye_projection_factor = setup_layout.camera_eye_distance / setup_layout.camera_lambda;

	et::EyeTracker eye_tracker{setup_layout, image_provider};
	eye_tracker.initializeKalmanFilter(framerate);

	et::SocketServer socket_server{&eye_tracker, &feature_detector};
	socket_server.startServer();
	std::thread t{&et::SocketServer::openSocket, &socket_server};

	et::Visualizer visualizer(&feature_detector, &eye_tracker);
	VisualizationType visualization_type = VisualizationType::STANDARD;

	cv::VideoWriter video_output{};

	while(!socket_server.finished) 
	{
		cv::Mat image{image_provider->grabImage()};
		bool features_found{feature_detector.findImageFeatures(image)};
		et::EyePosition eye_position{};
		if (features_found)
		{
			eye_tracker.calculateEyePosition(feature_detector.getPupil(), feature_detector.getLeds());
		}

		visualizer.calculateFramerate();

		if (visualization_type != VisualizationType::DISABLED)
		{
			switch(visualization_type)
			{
				case VisualizationType::STANDARD:
					visualizer.drawUi(image);
					break;
				case VisualizationType::THRESHOLD_PUPIL:
					visualizer.drawUi(feature_detector.getThresholdedPupilImage());
					break;
				case VisualizationType::THRESHOLD_GLINTS:
					visualizer.drawUi(feature_detector.getThresholdedGlintsImage());
					break;
			}
			visualizer.show();
		}
		else
		{
			visualizer.printFramerateInterval();
		}

		if (video_output.isOpened())
		{
			video_output.write(image);
		}

		switch (cv::waitKey(7) & 0xFF) 
		{
			case 27: // Esc
				socket_server.finished = true;
				break;
			case 'w':
				if (input_type != "file")
				{
					video_output.open("recorded_output.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30, image_provider->getResolution(), false);
				}
                break;
			case 's':
				imwrite("fullFrame.png", image);
                break;
			case 'q':
				visualization_type = VisualizationType::DISABLED;
				break;
			case 'e':
				visualization_type = VisualizationType::STANDARD;
				break;
			case 't':
				visualization_type = VisualizationType::THRESHOLD_PUPIL;
				break;
			case 'r':
				visualization_type = VisualizationType::THRESHOLD_GLINTS;
				break;
		}
	}

    socket_server.finished = true;

    t.join();

	if (video_output.isOpened())
	{
		video_output.release();
	}

	image_provider->close();
	socket_server.closeSocket();
	cv::destroyAllWindows();

	return EXIT_SUCCESS;
}