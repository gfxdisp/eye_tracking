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
    feature_detector.initializeKalmanFilters(image_provider->getImageResolution(), framerate);

    et::SetupLayout setup_layout{};
    setup_layout.camera_lambda = 27.119;
    setup_layout.camera_nodal_point_position = {224.1558, 139.7573, 491.9391};
    setup_layout.led_positions[0] = {185.4109, 144.5800, 778.9492};
    setup_layout.led_positions[1] = {235.4109, 130.3241, 763.2100};
    setup_layout.camera_eye_distance = 324.9387;
    setup_layout.translation = {-59.1085, -25.8134, -554.6367};
    setup_layout.camera_eye_projection_factor = setup_layout.camera_eye_distance / setup_layout.camera_lambda;
    double rotation_data[] = {-0.0126, 0.9960, 0.0883, 0.9992, 0.0158, -0.0357, 0.0370, -0.0878, 0.9955};
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
        // {
        // 	cv::Point2f pupil{310.9498, 184.5560};
        // 	cv::Point2f reflections[]{{320.1631, 163.7441},{320.2516, 206.1573}};
        // 	float rad = 5.0f;
        //     eye_tracker.calculateJoined(pupil, reflections, rad);
        //     cv::Vec3d cornea{};
        //     eye_tracker.getCorneaCurvaturePosition(cornea);
        //     std::cout << cornea << std::endl;
        //     break;
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
                                      image_provider->getImageResolution(), false);
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

    if (video_output.isOpened()) {
        video_output.release();
    }

    t.join();

    image_provider->close();
    socket_server.closeSocket();
    cv::destroyAllWindows();

    return EXIT_SUCCESS;
}