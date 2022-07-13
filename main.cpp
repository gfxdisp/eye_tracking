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
#include <sstream>
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
    setenv("DISPLAY", "10.248.101.97:0", true);
    assert(argc >= 3 && argc <= 6);

    std::string settings_path(argv[1]);
    et::Settings settings(settings_path);

    int user_idx{};

    std::string input_type{argv[2]};
    et::ImageProvider *image_provider;
    if (input_type == "ids") {
        image_provider = new et::IdsCamera();
        user_idx = 3;
    } else if (input_type == "file") {
        std::string input_file{argv[3]};
        user_idx = 4;
        image_provider = new et::InputVideo(input_file);
    } else {
        return EXIT_FAILURE;
    }
    if (argc > user_idx) {
        std::string user{argv[user_idx]};
        std::cout << "Starting eye-tracking for user: \"" << user << "\"\n";
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
    image_provider->setExposure(
        et::Settings::parameters.camera_params.exposure);

    et::FeatureDetector feature_detector{};
    feature_detector.initialize(
        et::Settings::parameters.camera_params.region_of_interest,
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

    bool slow_mode{false};

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
            if (input_type == "file") {
                video_output.write(visualizer.getUiImage());
            } else {
                video_output.write(image);
            }
        }

        int key_pressed = cv::waitKey(slow_mode ? 120 : 1) & 0xFF;
        switch (key_pressed) {
        case 27: // Esc
            socket_server.finished = true;
            break;
        case 'w': {
            std::string filename{"videos/" + getCurrentTimeText() + ".mp4"};
            std::clog << "Saving video to " << filename << "\n";
            video_output.open(
                filename, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30,
                et::Settings::parameters.camera_params.region_of_interest,
                false);
            break;
        }
        case 's':
            slow_mode = !slow_mode;
            break;
        case 'p': {
            std::string filename{"images/" + getCurrentTimeText() + ".png"};
            imwrite(filename.c_str(), image);
            break;
        }
        case 'q':
            visualization_type = VisualizationType::DISABLED;
            break;
        case 'e':
            visualization_type = VisualizationType::STANDARD;
            break;
        case 'r':
            visualization_type = VisualizationType::THRESHOLD_GLINTS;
            break;
        case 't':
            visualization_type = VisualizationType::THRESHOLD_PUPIL;
            break;
        default:
            break;
        }
    }
    socket_server.finished = true;

    std::cout << "Average framerate: " << visualizer.getAvgFramerate()
              << " fps\n";

    if (video_output.isOpened()) {
        video_output.release();
    }

    t.join();

    image_provider->close();
    socket_server.closeSocket();
    cv::destroyAllWindows();
    settings.saveSettings(settings_path);

    return EXIT_SUCCESS;
}