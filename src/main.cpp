#include "IdsCamera.hpp"
#include "ImageProvider.hpp"
#include "InputImages.hpp"
#include "InputVideo.hpp"
#include "Settings.hpp"
#include "SocketServer.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>

#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <string>
#include <thread>

namespace fs = std::filesystem;

int main(int argc, char *argv[]) {
    constexpr option options[]{{"settings-path", required_argument, nullptr, 's'},
                               {"feed-type", required_argument, nullptr, 'f'},
                               {"input-path", required_argument, nullptr, 'p'},
                               {"user", required_argument, nullptr, 'u'},
                               {"cameras", required_argument, nullptr, 'c'},
                               {"glint-ellipse", required_argument, nullptr, 'e'},
                               {"template", required_argument, nullptr, 't'},
                               {"log", no_argument, nullptr, 'l'},
                               {"distorted", required_argument, nullptr, 'd'},
                               {"kalman", required_argument, nullptr, 'k'},
                               {"headless", no_argument, nullptr, 'h'},
                               {nullptr, no_argument, nullptr, 0}};

    int argument{0};
    std::string settings_path{"."};
    std::string feed_type{"ids"};
    std::string input_path;
    std::string user{"default"};

    bool saving_log{false};
    bool ellipse_fitting[]{true, false};
    bool enabled_cameras[]{true, true};
    bool enabled_kalman[]{true, true};
    bool enabled_template_matching[]{true, false};
    bool distorted[]{true, true};
    bool headless{false};

    while (argument != -1) {
        argument = getopt_long(argc, argv, "s:f:p:u:c:e:ld:k:t:h", options, nullptr);
        switch (argument) {
        case 's':
            settings_path = optarg;
            break;
        case 'f':
            feed_type = optarg;
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
        case 't':
            enabled_template_matching[0] = optarg[0] == '1';
            enabled_template_matching[1] = optarg[1] == '1';
            break;
        case 'd':
            distorted[0] = optarg[0] == '1';
            distorted[1] = optarg[1] == '1';
            break;
        case 'h':
            headless = true;
            break;
        default:
            break;
        }
    }

    et::Settings settings{fs::path(settings_path) / "capture_params.json"};
    et::ImageProvider *image_provider;
    if (feed_type == "ids") {
        image_provider = new et::IdsCamera();
    } else if (feed_type == "file") {
        image_provider = new et::InputVideo(input_path);
    } else if (feed_type == "folder") {
        image_provider = new et::InputImages(input_path);
    } else {
        return EXIT_FAILURE;
    }
    std::cout << "Starting eye-tracking for user: \"" << user << "\"\n";
    if (!et::Settings::parameters.features_params[0].contains(user)) {
        et::Settings::parameters.features_params[0][user] = et::Settings::parameters.features_params[0]["default"];
        et::Settings::parameters.features_params[1][user] = et::Settings::parameters.features_params[1]["default"];
    }
    et::Settings::parameters.user_params[0] = &et::Settings::parameters.features_params[0][user];
    et::Settings::parameters.user_params[1] = &et::Settings::parameters.features_params[1][user];

    et::EyeTracker eye_tracker{};
    eye_tracker.initialize(image_provider, settings_path, input_path, enabled_cameras, ellipse_fitting, enabled_kalman,
                           distorted, enabled_template_matching, headless);

    et::SocketServer socket_server{&eye_tracker};
    socket_server.startServer();
    std::thread t{&et::SocketServer::openSocket, &socket_server};

    int frame_counter{0};
    bool slow_mode{false};

    if (saving_log) {
        for (int i = 0; i < 2; i++) {
            if (enabled_cameras[i]) {
                std::ofstream file{};
                file.open(fs::path(input_path) / ("image_features.csv"));
                file.close();
                file.open(fs::path(input_path) / ("eye_estimates.csv"));
                file.close();
            }
        }
    }

    while (!socket_server.finished) {
        if (!eye_tracker.analyzeNextFrame()) {
            std::cout << "Empty image. Finishing.\n";
            socket_server.finished = true;
            break;
        }

        if (saving_log) {
            for (int i = 0; i < 2; i++) {
                if (enabled_cameras[i]) {
                    std::ofstream file{};
                    file.open(fs::path(input_path) / ("image_features.csv"), std::ios::app);
                    eye_tracker.logDetectedFeatures(file, i);
                    file.close();

                    file.open(fs::path(input_path) / ("eye_estimates.csv"), std::ios::app);
                    eye_tracker.logEyePosition(file, i);
                    file.close();
                }
            }
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
            if (!headless) {
                eye_tracker.switchToCameraImage();
            }
            break;
        case 'e':
            if (!headless) {
                eye_tracker.switchToPupilThreshImage();
            }
            break;
        case 'r':
            if (!headless) {
                eye_tracker.switchToGlintThreshImage();
            }
            break;
        case 'i': // + left
            et::Settings::parameters.detection_params[0].pupil_search_radius++;
            break;
        case 'k': // - left
            et::Settings::parameters.detection_params[0].pupil_search_radius--;
            break;
        case 'g': // ← left
            et::Settings::parameters.detection_params[0].pupil_search_centre.x--;
            break;
        case 'y': // ↑ left
            et::Settings::parameters.detection_params[0].pupil_search_centre.y--;
            break;
        case 'j': // → left
            et::Settings::parameters.detection_params[0].pupil_search_centre.x++;
            break;
        case 'h': // ↓ left
            et::Settings::parameters.detection_params[0].pupil_search_centre.y++;
            break;
        case 'I': // + right
            et::Settings::parameters.detection_params[1].pupil_search_radius++;
            break;
        case 'K': // - right
            et::Settings::parameters.detection_params[1].pupil_search_radius--;
            break;
        case 'G': // ← right
            et::Settings::parameters.detection_params[1].pupil_search_centre.x--;
            break;
        case 'Y': // ↑ right
            et::Settings::parameters.detection_params[1].pupil_search_centre.y--;
            break;
        case 'J': // → right
            et::Settings::parameters.detection_params[1].pupil_search_centre.x++;
            break;
        case 'H': // ↓ right
            et::Settings::parameters.detection_params[1].pupil_search_centre.y++;
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

    std::cout << "Average framerate: " << et::EyeTracker::getAvgFramerate() << " fps\n";

    eye_tracker.stopVideoRecording();

    image_provider->close();
    socket_server.closeSocket();
    cv::destroyAllWindows();
    et::Settings::saveSettings(fs::path(settings_path) / "capture_params.json");

    t.join();

    return EXIT_SUCCESS;
}