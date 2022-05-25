#include "EyeTracker.hpp"
#include "FeatureDetector.hpp"
#include "IdsCamera.hpp"
#include "ImageProvider.hpp"
#include "InputVideo.hpp"
#include "Settings.hpp"
#include "SocketServer.hpp"
#include "Visualizer.hpp"

#include <opencv2/opencv.hpp>

#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

enum class VisualizationType {
    DISABLED,
    STANDARD,
    THRESHOLD_PUPIL,
    THRESHOLD_GLINTS
};

std::string getCurrentTimeText() {
    std::time_t now{
        std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())};
    char buffer[80];

    std::strftime(buffer, 80, "%Y-%m-%d_%H-%M-%S", std::localtime(&now));
    std::string s{buffer};
    return s;
}

int main(int argc, char *argv[]) {
    et::Settings settings("settings.json");
    //    setenv("DISPLAY", "10.248.97.27:0", true);
    assert(argc >= 2 && argc <= 4);

    int user_idx{};

    std::string input_type{argv[1]};
    et::ImageProvider *image_provider;
    if (input_type == "ids") {
        image_provider = new et::IdsCamera();
        user_idx = 2;
    } else if (input_type == "file") {
        std::string input_file{argv[2]};
        user_idx = 3;
        image_provider = new et::InputVideo(input_file);
    } else {
        return EXIT_FAILURE;
    }
    if (argc > user_idx) {
        std::string user{argv[user_idx]};
        if (!et::Settings::parameters.features_params.contains(user)) {
            et::Settings::parameters.features_params[user] =
                et::Settings::parameters.features_params["default"];
        }
        et::Settings::parameters.user_params =
            &et::Settings::parameters.features_params[user];
    } else {
    	et::Settings::parameters.user_params =
            &et::Settings::parameters.features_params["default"];
    }

    image_provider->initialize();
    image_provider->setGamma(et::Settings::parameters.camera_params.gamma);
    image_provider->setFramerate(
        et::Settings::parameters.camera_params.framerate);

    et::FeatureDetector feature_detector{};
    feature_detector.initializeKalmanFilters(
        et::Settings::parameters.camera_params.dimensions,
        et::Settings::parameters.camera_params.framerate);

    et::EyeTracker eye_tracker{image_provider};
    eye_tracker.initializeKalmanFilter(
        et::Settings::parameters.camera_params.framerate);

    et::SocketServer socket_server{&eye_tracker, &feature_detector};
    socket_server.startServer();
    std::thread t{&et::SocketServer::openSocket, &socket_server};

    et::Visualizer visualizer(&feature_detector, &eye_tracker);
    VisualizationType visualization_type = VisualizationType::STANDARD;

    cv::VideoWriter video_output{};
    cv::Point2f pupil{};
    cv::Point2f *glints{};

    while (!socket_server.finished) {
        cv::Mat image{image_provider->grabImage()};
        if (image.empty()) {
            break;
        }
        bool features_found{feature_detector.findImageFeatures(image)};
        et::EyePosition eye_position{};
        if (features_found) {
            eye_tracker.calculateJoined(feature_detector.getPupil(),
                                        *feature_detector.getGlints(),
                                        feature_detector.getPupilRadius());
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
                    std::string filename{"videos/" + getCurrentTimeText()
                                         + ".mp4"};
                    std::clog << "Saving video to " << filename << "\n";
                    video_output.open(
                        filename, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                        30, et::Settings::parameters.camera_params.region_of_interest,
                        false);
                }
                break;
            case 's': imwrite("fullFrame.png", image); break;
            case 'q': visualization_type = VisualizationType::DISABLED; break;
            case 'e': visualization_type = VisualizationType::STANDARD; break;
            case 'r':
                visualization_type = VisualizationType::THRESHOLD_GLINTS;
                break;
            case 't':
                visualization_type = VisualizationType::THRESHOLD_PUPIL;
                break;
            default: break;
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