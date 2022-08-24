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
    //setenv("DISPLAY", "10.248.101.97:0", true);
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

        std::cout << input_file + "_pupil.mp4" << " " << input_file + "_glint.mp4" << std::endl;
        image_provider = new et::InputVideo(input_file);
//        image_provider = new et::InputVideo(input_file + "_pupil.mp4", input_file + "_glint.mp4");
    } else if (input_type == "folder") {
        std::string input_folder{argv[3]};
        user_idx = 4;
        image_provider = new et::InputImages(input_folder);
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
    VisualizationType visualization_type = VisualizationType::PUPIL;

    cv::VideoWriter pupil_video_output{};
    cv::VideoWriter glint_video_output{};

    bool slow_mode{false};
    bool saving_log{false};
    bool ellipse_fitting{false};

    if (saving_log) {
        std::ofstream file{};
        file.open("log.txt");
        file.close();
    }

    int frame_counter{0};
    while (!socket_server.finished) {
        cv::Mat pupil_image{image_provider->grabPupilImage()};
        cv::Mat glint_image{image_provider->grabGlintImage()};
        if (pupil_image.empty() || glint_image.empty()) {
            break;
        }
        bool features_found{feature_detector.findPupil(pupil_image)};
        if (features_found) {
            if (ellipse_fitting) {
                features_found &= feature_detector.findEllipse(glint_image);
            } else {
                features_found &= feature_detector.findGlints(glint_image);
            }
        }
        feature_detector.updateGazeBuffer();
        et::EyePosition eye_position{};
        if (saving_log && features_found) {
            std::ofstream file{};
            file.open("log.txt", std::ios::app);
            cv::Point2f pupil = feature_detector.getPupil();
            file << frame_counter << "," << pupil.x << "," << pupil.y;
            if (ellipse_fitting) {
                cv::RotatedRect ellipse = feature_detector.getEllipse();
                file << "," << ellipse.center.x << "," << ellipse.center.y
                     << ",";
                file << ellipse.size.width << "," << ellipse.size.height << ",";
                file << ellipse.angle;
            } else {
                std::vector<cv::Point2f> *glints = feature_detector.getGlints();
                for (int i = 0; i < glints->size(); i++) {
                    file << "," << (*glints)[i].x << "," << (*glints)[i].y;
                }
            }
            file << "\n";
            file.close();
        } else if (features_found) {
            eye_tracker.calculateJoined(feature_detector.getPupil(),
                                        *feature_detector.getGlints(),
                                        feature_detector.getPupilRadius());
        }

        visualizer.calculateFramerate();

        switch (visualization_type) {
        case VisualizationType::PUPIL:
            visualizer.drawUi(pupil_image);
            visualizer.show();
            break;
        case VisualizationType::GLINTS:
            visualizer.drawUi(glint_image);
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

        if (pupil_video_output.isOpened()) {
            if (input_type == "file") {
                pupil_video_output.write(visualizer.getUiImage());
            } else {
                pupil_video_output.write(pupil_image);
            }
        }

        if (glint_video_output.isOpened() && input_type != "file") {
            glint_video_output.write(glint_image);
        }

        int key_pressed = cv::waitKey(slow_mode ? 120 : 1) & 0xFF;
        switch (key_pressed) {
        case 27: // Esc
            if (!socket_server.isClientConnected()) {
                socket_server.finished = true;
            }
            break;
        case 'v': {
            std::string filename{"videos/" + getCurrentTimeText()
                                 + "_pupil.mp4"};
            std::clog << "Saving video to " << filename << "\n";
            pupil_video_output.open(
                filename, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30,
                et::Settings::parameters.camera_params.region_of_interest,
                false);
            if (input_type != "file") {
                filename = "videos/" + getCurrentTimeText() + "_glint.mp4";
                std::clog << "Saving video to " << filename << "\n";
                glint_video_output.open(
                    filename, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30,
                    et::Settings::parameters.camera_params.region_of_interest,
                    false);
            }
            break;
        }
        case 's':
            slow_mode = !slow_mode;
            break;
        case 'p': {
            std::string filename{"images/" + getCurrentTimeText()
                                 + "_pupil.png"};
            imwrite(filename.c_str(), pupil_image);
            filename = "images/" + getCurrentTimeText() + "_glint.png";
            imwrite(filename.c_str(), glint_image);
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
        default:
            break;
        }
        frame_counter++;
    }
    socket_server.finished = true;

    std::cout << "Average framerate: " << visualizer.getAvgFramerate()
              << " fps\n";

    if (pupil_video_output.isOpened()) {
        pupil_video_output.release();
    }

    if (glint_video_output.isOpened()) {
        glint_video_output.release();
    }

    t.join();

    image_provider->close();
    socket_server.closeSocket();
    cv::destroyAllWindows();
    settings.saveSettings(settings_path);

    return EXIT_SUCCESS;
}