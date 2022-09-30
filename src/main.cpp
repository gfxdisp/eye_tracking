#include "EyeTracker.hpp"
#include "FeatureDetector.hpp"
#include "IdsCamera.hpp"
#include "ImageProvider.hpp"
#include "InputImages.hpp"
#include "InputVideo.hpp"
#include "Settings.hpp"
#include "SocketServer.hpp"
#include "Visualizer.hpp"

#include <opencv2/opencv.hpp>

#include <cassert>
#include <chrono>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

enum class VisualizationType {
    DISABLED,
    PUPIL,
    GLINTS,
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
    constexpr option options[]{
        {"settings-path", required_argument, nullptr, 's'},
        {"input-type", required_argument, nullptr, 't'},
        {"input-path", required_argument, nullptr, 'p'},
        {"user", required_argument, nullptr, 'u'},
        {"glint-ellipse", no_argument, nullptr, 'g'},
        {"double-exposure", no_argument, nullptr, 'e'},
        {"log", no_argument, nullptr, 'l'},
        {nullptr, no_argument, nullptr, 0}};

    int argument{0};
    std::string settings_path{"settings.json"};
    std::string input_type{"ids"};
    std::string input_path;
    std::string user{"default"};

    bool saving_log{false};
    bool ellipse_fitting{false};
    bool double_exposure{false};

    while (argument != -1) {
        argument = getopt_long(argc, argv, "s:t:p:u:gel", options, nullptr);
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
        case 'g':
            ellipse_fitting = true;
            break;
        case 'e':
            double_exposure = true;
            break;
        case 'l':
            saving_log = true;
            break;
        default:
            break;
        }
    }

    et::Settings settings(settings_path);
    et::ImageProvider *image_provider;
    if (input_type == "ids") {
        image_provider = new et::IdsCamera();
    } else if (input_type == "file") {
        image_provider = new et::InputVideo(input_path);
    } else if (input_type == "folder") {
        image_provider = new et::InputImages(input_path);
    } else {
        return EXIT_FAILURE;
    }
    std::cout << "Starting eye-tracking for user: \"" << user << "\"\n";
    if (!et::Settings::parameters.features_params.contains(user)) {
        et::Settings::parameters.features_params[user] =
            et::Settings::parameters.features_params["default"];
    }
    et::Settings::parameters.user_params =
        &et::Settings::parameters.features_params[user];

    image_provider->initialize(double_exposure);

    et::FeatureDetector feature_detector{};
    feature_detector.initialize();

    et::EyeTracker eye_tracker{};
    eye_tracker.initialize();

    std::vector<int> camera_ids = image_provider->getCameraIds();

    et::SocketServer socket_server{&eye_tracker, &feature_detector};
    socket_server.startServer();
    std::thread t{&et::SocketServer::openSocket, &socket_server};

    et::Visualizer visualizer(&feature_detector, &eye_tracker);
    VisualizationType visualization_type = VisualizationType::PUPIL;

    cv::VideoWriter pupil_video_output[2]{};
    cv::VideoWriter glint_video_output[2]{};

    if (saving_log) {
        std::ofstream file{};
        file.open("log.txt");
        file.close();
    }

    int frame_counter{0};
    bool slow_mode{false};

    cv::Mat pupil_image[2], glint_image[2];
    while (!socket_server.finished) {
        for (int i : camera_ids) {

            pupil_image[i] = double_exposure ? image_provider->grabPupilImage(i)
                                             : image_provider->grabImage(i);
            glint_image[i] = double_exposure ? image_provider->grabGlintImage(i)
                                             : pupil_image[i];
            if (pupil_image[i].empty() || glint_image[i].empty()) {
                std::cout << "Empty image. Finishing.\n";
                socket_server.finished = true;
                break;
            }
            bool features_found{feature_detector.findPupil(pupil_image[i], i)};
            if (ellipse_fitting) {
                features_found &= feature_detector.findEllipse(
                    glint_image[i], feature_detector.getPupil(i), i);
            } else {
                features_found &=
                    feature_detector.findGlints(glint_image[i], i);
            }
            et::EyePosition eye_position{};
            if (saving_log && features_found) {
                std::ofstream file{};
                file.open("log.txt", std::ios::app);
                cv::Point2f pupil = feature_detector.getPupil(i);
                file << i << "," << frame_counter << "," << pupil.x << ","
                     << pupil.y;
                if (ellipse_fitting) {
                    cv::RotatedRect ellipse = feature_detector.getEllipse(i);
                    file << "," << ellipse.center.x << "," << ellipse.center.y
                         << ",";
                    file << ellipse.size.width << "," << ellipse.size.height
                         << ",";
                    file << ellipse.angle;
                } else {
                    std::vector<cv::Point2f> *glints =
                        feature_detector.getGlints(i);
                    for (int j = 0; j < glints->size(); j++) {
                        file << "," << (*glints)[j].x << "," << (*glints)[j].y;
                    }
                }
                file << "\n";
                file.close();
            } else if (features_found) {
                eye_tracker.calculateJoined(
                    feature_detector.getPupil(i), feature_detector.getGlints(i),
                    feature_detector.getPupilRadius(i), i);
            }

            visualizer.calculateFramerate();

            switch (visualization_type) {
            case VisualizationType::PUPIL:
                visualizer.drawUi(pupil_image[i], i);
                break;
            case VisualizationType::GLINTS:
                visualizer.drawUi(glint_image[i], i);
                break;
            case VisualizationType::THRESHOLD_PUPIL:
                visualizer.drawUi(feature_detector.getThresholdedPupilImage(i),
                                  i);
                break;
            case VisualizationType::THRESHOLD_GLINTS:
                visualizer.drawUi(feature_detector.getThresholdedGlintsImage(i),
                                  i);
                break;
            case VisualizationType::DISABLED:
                visualizer.printFramerateInterval();
                break;
            }

            if (pupil_video_output[i].isOpened()) {
                if (input_type == "file") {
                    pupil_video_output[i].write(visualizer.getUiImage(i));
                } else {
                    pupil_video_output[i].write(pupil_image[i]);
                }
            }

            if (glint_video_output[i].isOpened()) {
                glint_video_output[i].write(glint_image[i]);
            }
        }
        feature_detector.updateGazeBuffer();
        if (visualization_type != VisualizationType::DISABLED) {
            visualizer.show();
        }

        static bool first_it = false;
//        static bool first_it = true;
        if (first_it) {
            first_it = false;
            for (int i : camera_ids) {
                if (double_exposure) {
                    std::string filename{"videos/" + getCurrentTimeText() + "_"
                                         + std::to_string(i) + "_pupil.mp4"};
                    std::clog << "Saving video to " << filename << "\n";
                    pupil_video_output[i].open(
                        filename, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                        30,
                        et::Settings::parameters.camera_params[i]
                            .region_of_interest,
                        false);
                    filename = "videos/" + getCurrentTimeText() + "_glint.mp4";
                    std::clog << "Saving video to " << filename << "\n";
                    glint_video_output[i].open(
                        filename, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                        30,
                        et::Settings::parameters.camera_params[i]
                            .region_of_interest,
                        false);
                } else {
                    std::string filename{"videos/" + getCurrentTimeText() + "_"
                                         + std::to_string(i) + ".mp4"};
                    std::clog << "Saving video to " << filename << "\n";
                    pupil_video_output[i].open(
                        filename, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                        30,
                        et::Settings::parameters.camera_params[i]
                            .region_of_interest,
                        false);
                }
            }
        }

        int key_pressed = cv::pollKey() & 0xFFFF;
        switch (key_pressed) {
        case 27: // Esc
            if (!socket_server.isClientConnected()) {
                socket_server.finished = true;
            }
            break;
        case 'v': {
            for (int i : camera_ids) {
                if (double_exposure) {
                    std::string filename{"videos/" + getCurrentTimeText() + "_"
                                         + std::to_string(i) + "_pupil.mp4"};
                    std::clog << "Saving video to " << filename << "\n";
                    pupil_video_output[i].open(
                        filename, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                        30,
                        et::Settings::parameters.camera_params[i]
                            .region_of_interest,
                        false);
                    filename = "videos/" + getCurrentTimeText() + "_glint.mp4";
                    std::clog << "Saving video to " << filename << "\n";
                    glint_video_output[i].open(
                        filename, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                        30,
                        et::Settings::parameters.camera_params[i]
                            .region_of_interest,
                        false);
                } else {
                    std::string filename{"videos/" + getCurrentTimeText() + "_"
                                         + std::to_string(i) + ".mp4"};
                    std::clog << "Saving video to " << filename << "\n";
                    pupil_video_output[i].open(
                        filename, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                        30,
                        et::Settings::parameters.camera_params[i]
                            .region_of_interest,
                        false);
                }
            }
            break;
        }
        case 's':
            slow_mode = !slow_mode;
            break;
        case 'p': {
            for (int i : camera_ids) {
                if (double_exposure) {
                    std::string filename{"images/" + getCurrentTimeText() + "_"
                                         + std::to_string(i) + "_pupil.png"};
                    imwrite(filename, pupil_image[i]);
                    filename = "images/" + getCurrentTimeText() + "_glint.png";
                    imwrite(filename, glint_image[i]);
                } else {
                    std::string filename{"images/" + getCurrentTimeText() + "_"
                                         + std::to_string(i) + ".png"};
                    imwrite(filename, pupil_image[i]);
                }
            }
            break;
        }
        case 'q':
            visualization_type = VisualizationType::DISABLED;
            break;
        case 'w':
            visualization_type = VisualizationType::PUPIL;
            break;
        case 'e':
            visualization_type = VisualizationType::GLINTS;
            break;
        case 'r':
            visualization_type = VisualizationType::THRESHOLD_PUPIL;
            break;
        case 't':
            visualization_type = VisualizationType::THRESHOLD_GLINTS;
            break;
        case 'i': // + left
            et::Settings::parameters.detection_params.pupil_search_radius[0]++;
            break;
        case 'k': // - left
            et::Settings::parameters.detection_params.pupil_search_radius[0]--;
            break;
        case 'g': // ← left
            et::Settings::parameters.detection_params.pupil_search_centre[0]
                .x--;
            break;
        case 'y': // ↑ left
            et::Settings::parameters.detection_params.pupil_search_centre[0]
                .y--;
            break;
        case 'j': // → left
            et::Settings::parameters.detection_params.pupil_search_centre[0]
                .x++;
            break;
        case 'h': // ↓ left
            et::Settings::parameters.detection_params.pupil_search_centre[0]
                .y++;
            break;
        case 65451: // + right
            et::Settings::parameters.detection_params.pupil_search_radius[1]++;
            break;
        case 65453: // - right
            et::Settings::parameters.detection_params.pupil_search_radius[1]--;
            break;
        case 65361: // ← right
            et::Settings::parameters.detection_params.pupil_search_centre[1]
                .x--;
            break;
        case 65362: // ↑ right
            et::Settings::parameters.detection_params.pupil_search_centre[1]
                .y--;
            break;
        case 65363: // → right
            et::Settings::parameters.detection_params.pupil_search_centre[1]
                .x++;
            break;
        case 65364: // ↓ right
            et::Settings::parameters.detection_params.pupil_search_centre[1]
                .y++;
            break;
        default:
            break;
        }

        if (!visualizer.isWindowOpen()) {
            socket_server.finished = true;
        }

        frame_counter++;
    }
    socket_server.finished = true;

    std::cout << "Average framerate: " << visualizer.getAvgFramerate()
              << " fps\n";

    for (int i : camera_ids) {
        if (pupil_video_output[i].isOpened()) {
            pupil_video_output[i].release();
        }

        if (glint_video_output[i].isOpened()) {
            glint_video_output[i].release();
        }
    }

    image_provider->close();
    socket_server.closeSocket();
    cv::destroyAllWindows();
    settings.saveSettings(settings_path);

    t.join();

    return EXIT_SUCCESS;
}