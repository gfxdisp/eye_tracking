#include "IdsCamera.hpp"
#include "ImageProvider.hpp"
#include "InputImages.hpp"
#include "InputVideo.hpp"
#include "Settings.hpp"
#include "SocketServer.hpp"

#include <opencv2/opencv.hpp>

#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <string>
#include <thread>

namespace fs = std::filesystem;

int main(int argc, char *argv[]) {
    constexpr option options[]{
        {"settings-path", required_argument, nullptr, 's'},
        {"input-type", required_argument, nullptr, 't'},
        {"input-path", required_argument, nullptr, 'p'},
        {"user", required_argument, nullptr, 'u'},
        {"cameras", no_argument, nullptr, 'c'},
        {"glint-ellipse", no_argument, nullptr, 'e'},
        {"log", no_argument, nullptr, 'l'},
        {nullptr, no_argument, nullptr, 0}};

    int argument{0};
    std::string settings_path{"."};
    std::string input_type{"ids"};
    std::string input_path;
    std::string user{"default"};

    bool saving_log{false};
    bool ellipse_fitting[]{true, false};
    bool enabled_cameras[]{true, true};
    bool enabled_kalman[]{true, true};

    while (argument != -1) {
        argument = getopt_long(argc, argv, "s:t:p:u:c:e:lk:", options, nullptr);
        switch (argument) {
        case 's':
            settings_path = optarg;
            break;
        case 't':
            input_type = optarg;
            break;
        case 'p':
            input_path = optarg;
            break;
        case 'u':
            user = optarg;
            break;
        case 'c':
            enabled_cameras[0] = optarg[0] == '1';
            enabled_cameras[1] = optarg[1] == '1';
            break;
        case 'e':
            ellipse_fitting[0] = optarg[0] == '1';
            ellipse_fitting[1] = optarg[1] == '1';
            break;
        case 'l':
            saving_log = true;
            break;
        case 'k':
            enabled_kalman[0] = optarg[0] == '1';
            enabled_kalman[1] = optarg[1] == '1';
            break;
        default:
            break;
        }
    }

    et::Settings settings(fs::path(settings_path) / "parameters.json");
    et::ImageProvider *image_provider;
    if (input_type == "ids") {
        image_provider = new et::IdsCamera();
    } else if (input_type == "file") {
        image_provider =
            new et::InputVideo(fs::path(settings_path) / input_path);
    } else if (input_type == "folder") {
        image_provider =
            new et::InputImages(fs::path(settings_path) / input_path);
    } else {
        return EXIT_FAILURE;
    }
    std::cout << "Starting eye-tracking for user: \"" << user << "\"\n";
    if (!et::Settings::parameters.features_params[0].contains(user)) {
        et::Settings::parameters.features_params[0][user] =
            et::Settings::parameters.features_params[0]["default"];
        et::Settings::parameters.features_params[1][user] =
            et::Settings::parameters.features_params[1]["default"];
    }
    et::Settings::parameters.user_params[0] =
        &et::Settings::parameters.features_params[0][user];
    et::Settings::parameters.user_params[1] =
        &et::Settings::parameters.features_params[1][user];

    et::EyeTracker eye_tracker{};
    eye_tracker.initialize(image_provider, settings_path, enabled_cameras,
                           ellipse_fitting, enabled_kalman);

    et::SocketServer socket_server{&eye_tracker};
    socket_server.startServer();
    std::thread t{&et::SocketServer::openSocket, &socket_server};

    if (saving_log) {
        std::ofstream file{};
        file.open("log.txt");
        file.close();
    }

    int frame_counter{0};
    bool slow_mode{false};

    while (!socket_server.finished) {
        if (!eye_tracker.analyzeNextFrame()) {
            std::cout << "Empty image. Finishing.\n";
            socket_server.finished = true;
            break;
        }

        if (saving_log) {
            std::ofstream file{};
            file.open("log.txt", std::ios::app);
            eye_tracker.logDetectedFeatures(file);
            file.close();
        }

        eye_tracker.updateUi();

        int key_pressed;
        if (slow_mode) {
            key_pressed = cv::waitKey(1000) & 0xFFFF;
        } else {
            key_pressed = cv::pollKey() & 0xFFFF;
        }

        switch (key_pressed) {
        case 27: // Esc
            if (!socket_server.isClientConnected()) {
                socket_server.finished = true;
            }
            break;
        case 'v': {
            eye_tracker.switchVideoRecordingState();
            break;
        }
        case 's':
            slow_mode = !slow_mode;
            break;
        case 'p': {
            eye_tracker.captureCameraImage();
            break;
        }
        case 'q':
            eye_tracker.disableImageUpdate();
            break;
        case 'w':
            eye_tracker.switchToCameraImage();
            break;
        case 'e':
            eye_tracker.switchToPupilThreshImage();
            break;
        case 'r':
            eye_tracker.switchToGlintThreshImage();
            break;
        case 'i': // + left
            et::Settings::parameters.detection_params[0].pupil_search_radius++;
            break;
        case 'k': // - left
            et::Settings::parameters.detection_params[0].pupil_search_radius--;
            break;
        case 'g': // ← left
            et::Settings::parameters.detection_params[0]
                .pupil_search_centre.x--;
            break;
        case 'y': // ↑ left
            et::Settings::parameters.detection_params[0]
                .pupil_search_centre.y--;
            break;
        case 'j': // → left
            et::Settings::parameters.detection_params[0]
                .pupil_search_centre.x++;
            break;
        case 'h': // ↓ left
            et::Settings::parameters.detection_params[0]
                .pupil_search_centre.y++;
            break;
        case 'I': // + right
            et::Settings::parameters.detection_params[1].pupil_search_radius++;
            break;
        case 'K': // - right
            et::Settings::parameters.detection_params[1].pupil_search_radius--;
            break;
        case 'G': // ← right
            et::Settings::parameters.detection_params[1]
                .pupil_search_centre.x--;
            break;
        case 'Y': // ↑ right
            et::Settings::parameters.detection_params[1]
                .pupil_search_centre.y--;
            break;
        case 'J': // → right
            et::Settings::parameters.detection_params[1]
                .pupil_search_centre.x++;
            break;
        case 'H': // ↓ right
            et::Settings::parameters.detection_params[1]
                .pupil_search_centre.y++;
            break;
        default:
            break;
        }

        if (eye_tracker.shouldAppClose()) {
            socket_server.finished = true;
        }

        frame_counter++;
    }
    socket_server.finished = true;

    std::cout << "Average framerate: " << et::EyeTracker::getAvgFramerate()
              << " fps\n";

    eye_tracker.stopVideoRecording();

    image_provider->close();
    socket_server.closeSocket();
    cv::destroyAllWindows();
    et::Settings::saveSettings(settings_path);

    t.join();

    return EXIT_SUCCESS;
}