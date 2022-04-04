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
#include <string>
#include <thread>

enum class VisualizationType { DISABLED, STANDARD, THRESHOLD_PUPIL, THRESHOLD_GLINTS };

std::string getCurrentTimeText() {
    std::time_t now{std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())};
    char buffer[80];

    std::strftime(buffer, 80, "%Y-%m-%d_%H-%M-%S", std::localtime(&now));
    std::string s{buffer};
    return s;
}

int main(int argc, char *argv[]) {
    assert(argc >= 3 && argc <= 4);
    std::string input_type{argv[1]};

    et::ImageProvider *image_provider;
    if (input_type == "ids") {
        int camera_index{std::stoi(std::string(argv[2]))};
        image_provider = new et::IdsCamera(camera_index);
    } else if (input_type == "file") {
        std::string input_file{argv[2]};
        image_provider = new et::InputVideo(input_file);
    } else {
        return EXIT_FAILURE;
    }

    float framerate{90.0f};

    image_provider->initialize();
    image_provider->setGamma(3.5f);
    image_provider->setFramerate(framerate);

    et::FeatureDetector feature_detector{};
    feature_detector.initializeKalmanFilters(image_provider->getResolution(), framerate);

    et::SetupLayout setup_layout{};
    setup_layout.camera_lambda = 27.119;
    setup_layout.camera_nodal_point_position = {206.023, 135.415, 507.786};
    setup_layout.led_positions[0] = {143.046, 223.2, 750.928};
    setup_layout.led_positions[1] = {171.429, 234.524, 749.077};
    setup_layout.camera_eye_distance = 328.39;
    setup_layout.camera_eye_projection_factor = setup_layout.camera_eye_distance / setup_layout.camera_lambda;
    double rotation_data[] = {-0.01263944363592692, -0.996010492298247, 0.08833653658808978, 0.9992362816964321, 
    	-0.01584811618383168, -0.0357168105366733, 0.03697428574107997, 0.08781763174459042, 0.995450132225974};
    setup_layout.rotation = cv::Mat(3, 3, CV_64F, rotation_data);

    et::EyeTracker eye_tracker{setup_layout, image_provider};
    eye_tracker.initializeKalmanFilter(framerate);

    et::SocketServer socket_server{&eye_tracker, &feature_detector};
    socket_server.startServer();
    std::thread t{&et::SocketServer::openSocket, &socket_server};

    et::Visualizer visualizer(&feature_detector, &eye_tracker);
    VisualizationType visualization_type = VisualizationType::STANDARD;

    cv::VideoWriter video_output{};

    while (!socket_server.finished) {
        cv::Mat image{image_provider->grabImage()};
        bool features_found{feature_detector.findImageFeatures(image)};
        et::EyePosition eye_position{};
        if (eye_tracker.isSetupUpdated() && features_found) {
        	// cv::Point2f pupil{321.4096, 175.0556};
        	// cv::Point2f reflections[]{{330.5021, 331.4708},{331.4708,195.9988}};
        	// float rad = 5.0f;
         //    eye_tracker.calculateJoined(pupil, reflections, rad);
         //    cv::Vec3d eye_position{};
         //    eye_tracker.getCorneaCurvaturePosition(eye_position);
         //    std::cout << eye_position << std::endl;

         //    break;
            eye_tracker.calculateJoined(feature_detector.getPupil(), feature_detector.getLeds(), feature_detector.getPupilRadius());
        }

        visualizer.calculateFramerate();

        switch (visualization_type) {
            case VisualizationType::STANDARD:
                visualizer.drawUi(image);
                visualizer.show();
                break;
            case VisualizationType::THRESHOLD_PUPIL:
                visualizer.drawUi(feature_detector.getThresholdedPupilImage());
                visualizer.show();
                break;
            case VisualizationType::THRESHOLD_GLINTS:
                visualizer.drawUi(feature_detector.getThresholdedGlintsImage());
                visualizer.show();
                break;
            case VisualizationType::DISABLED:
                visualizer.printFramerateInterval();
                break;
        }

        if (video_output.isOpened()) {
            video_output.write(image);
        }

        int key_pressed = cv::waitKey(7) & 0xFF;
        switch (key_pressed) {
            case 27:// Esc
                socket_server.finished = true;
                break;
            case 'w':
                if (input_type != "file") {
                    std::string filename{"videos/" + getCurrentTimeText() + ".mp4"};
                    std::clog << "Saving video to " << filename << "\n";
                    video_output.open(filename, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30,
                                      image_provider->getResolution(), false);
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
            case 171:
                feature_detector.pupil_threshold++;
                break;
            case 173:
                feature_detector.pupil_threshold--;
                break;
            default:
                break;
        }
    }

    socket_server.finished = true;

    t.join();

    if (video_output.isOpened()) {
        video_output.release();
    }

    image_provider->close();
    socket_server.closeSocket();
    cv::destroyAllWindows();

    return EXIT_SUCCESS;
}