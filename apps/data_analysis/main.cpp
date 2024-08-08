#include <eye_tracker/Settings.hpp>
#include "eye_tracker/frameworks/Framework.hpp"
#include "eye_tracker/image/temporal_filter/ContinuousTemporalFilterer.hpp"
#include "eye_tracker/SocketServer.hpp"
#include "eye_tracker/frameworks/OnlineCameraFramework.hpp"
#include "eye_tracker/frameworks/VideoCameraFramework.hpp"
#include "eye_tracker/Utils.hpp"

#include <getopt.h>
#include <string>
#include <memory>

int main(int argc, char *argv[])
{
    constexpr option options[] = {{"settings-path", required_argument, nullptr, 's'},
                                  {"user",          required_argument, nullptr, 'u'},
                                  {"headless",      no_argument,       nullptr, 'h'},
                                  {nullptr,         no_argument,       nullptr, 0}};

    int argument{0};
    std::string settings_path{"."};
    std::string user{"default"};
    bool headless{false};

    while (argument != -1)
    {
        argument = getopt_long(argc, argv, "s:u:h", options, nullptr);
        switch (argument)
        {
            case 's':
                settings_path = optarg;
                break;
            case 'u':
                user = optarg;
                break;
            case 'h':
                headless = true;
                break;
            default:
                break;
        }
    }

    int n_cameras = 1;



    user = "jgm45";
    auto settings = std::make_shared<et::Settings>(settings_path);
    std::shared_ptr<et::Framework> frameworks[2];
    for (int i = 0; i < n_cameras; i++)
    {
        if (!et::Settings::parameters.features_params[i].contains(user))
        {
            et::Settings::parameters.features_params[i][user] = et::Settings::parameters.features_params[i]["default"];
        }
        et::Settings::parameters.user_params[i] = &et::Settings::parameters.features_params[i][user];
    }


    auto meta_model = std::make_shared<et::MetaModel>(0);

    bool *variables_to_disable[] = {&et::ContinuousTemporalFilterer::ransac, &et::ModelEyeEstimator::test_all_glints, &et::MetaModel::calibration_enabled};
    et::EyeEstimator::moving_average = false;

    for (int j = 3; j <= 3; j++) {
        for (int i = 0; i < 3; i++) {
            *variables_to_disable[i] = i != j;
        }

        if (user == "dh706") {
            et::Settings::parameters.user_params[0]->glint_threshold = 98;
        }

        meta_model->findMetaModelFromFile("/mnt/d/Downloads/results/" + user + "_position_calib.mp4", "/mnt/d/Downloads/results/" + user + "_position_calib.csv", true);
        for (int i = 1; i <= 3; i++) {
            std::string video_path = "/mnt/d/Downloads/results/" + user + "_session_" + std::to_string(i) + ".mp4";
            std::string csv_path = "/mnt/d/Downloads/results/" + user + "_session_" + std::to_string(i) + ".csv";
            auto errors = meta_model->getEstimationsAtFrames(video_path, csv_path);
            et::Utils::writeFloatCsv(errors, "/mnt/d/Downloads/results/errors/" + user + "_session_" + std::to_string(i) + "_errors_mod" + std::to_string(j) + ".csv", false);
        }

        for (int i = 1; i <= 4; i++) {
            std::string video_path = "/mnt/d/Downloads/results/" + user + "_gaze_" + std::to_string(i) + ".mp4";
            std::string csv_path = "/mnt/d/Downloads/results/" + user + "_gaze_" + std::to_string(i) + ".csv";

            if (user == "dh706") {
                if (i < 3) {
                    et::Settings::parameters.user_params[0]->glint_threshold = 98;
                } else {
                    et::Settings::parameters.user_params[0]->glint_threshold = 119;
                }
            }

            auto errors = meta_model->findMetaModelFromFile(video_path, csv_path, false);
            et::Utils::writeFloatCsv(errors, "/mnt/d/Downloads/results/errors/" + user + "_gaze_errors_mod" + std::to_string(j) + ".csv", i != 1, "offset_error,angle_error,pc_error");
        }
    }

//    meta_model->findMetaModelFromFile("/mnt/d/Downloads/results/" + user + "_position_calib.mp4", "/mnt/d/Downloads/results/" + user + "_position_calib.csv", true);

    frameworks[0] = std::make_shared<et::VideoCameraFramework>(0, headless, "/mnt/d/Downloads/results/jgm45_gaze_1.mp4", false);
    auto socket_server = std::make_shared<et::SocketServer>(frameworks[0], frameworks[1]);
    socket_server->startServer();

//    frameworks[0]->startRecording();
    while (!socket_server->finished)
    {
        int key_pressed = cv::pollKey() & 0xFFFF;

        for (int i = 0; i < n_cameras; i++)
        {
            if (!frameworks[i]->analyzeNextFrame())
            {
                std::cout << "Empty image. Finishing.\n";
                socket_server->finished = true;
                break;
            }

            frameworks[i]->updateUi();
            switch (key_pressed)
            {
                case 27: // Esc
                    if (!socket_server->isClientConnected())
                    {
                        socket_server->finished = true;
                    }
                    break;
                case 'v':
                {
                    frameworks[i]->startRecording();
                    break;
                }
                case 'p':
                {
                    frameworks[i]->captureCameraImage();
                    break;
                }
                case 'q':
                    frameworks[i]->disableImageUpdate();
                    break;
                case 'w':
                    if (!headless)
                    {
                        frameworks[i]->switchToCameraImage();
                    }
                    break;
                case 'e':
                    if (!headless)
                    {
                        frameworks[i]->switchToPupilThreshImage();
                    }
                    break;
                case 'r':
                    if (!headless)
                    {
                        frameworks[i]->switchToGlintThreshImage();
                    }
                    break;
                default:
                    break;
            }

            if (frameworks[i]->shouldAppClose())
            {
                socket_server->finished = true;
                break;
            }
        }
    }
//   frameworks[0]->stopRecording();

    socket_server->closeSocket();
    cv::destroyAllWindows();
    et::Settings::saveSettings();


    return 0;
}
