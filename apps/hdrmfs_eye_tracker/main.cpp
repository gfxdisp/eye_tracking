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



    auto settings = std::make_shared<et::Settings>(settings_path);
    std::shared_ptr<et::Framework> frameworks[2];
    for (int i = 0; i < n_cameras; i++)
    {
        if (!et::Settings::parameters.features_params[i].contains(user))
        {
            et::Settings::parameters.features_params[i][user] = et::Settings::parameters.features_params[i]["default"];
        }
        et::Settings::parameters.user_params[i] = &et::Settings::parameters.features_params[i][user];
        frameworks[i] = std::make_shared<et::OnlineCameraFramework>(i, headless);
        //frameworks[i] = std::make_shared<et::VideoCameraFramework>(i, headless, "/home/jgm45/Documents/hdr_display/eye_tracker/experiments/videos/test_video2_0.mp4", true);
    }

//    std::cout << "Extrinsic matrix: " << et::Settings::parameters.camera_params[0].extrinsic_matrix << std::endl;
//    std::cout << "Intrinsic matrix: " << et::Settings::parameters.camera_params[0].intrinsic_matrix << std::endl;

//    frameworks[0] = std::make_shared<et::VideoCameraFramework>(0, headless, "/mnt/c/Users/Marek/Documents/Projects/hdr_display/eye_tracker/experiments/videos/spiral_0.mp4", false);
//    frameworks[1] = std::make_shared<et::VideoCameraFramework>(1, headless, "/mnt/d/Downloads/et_videos_2024-04-20_1/2024-04-20_16-22-52_1.mp4", true);

    auto socket_server = std::make_shared<et::SocketServer>(frameworks[0], frameworks[1]);
    socket_server->startServer();

//    std::vector<std::vector<double>> data;
//    std::shared_ptr<et::MetaModel> meta_model = std::make_shared<et::MetaModel>(0);
//    meta_model->findOnlineMetaModel("/mnt/d/Downloads/input_data.csv", "/mnt/d/Downloads/output_data.csv", true);
//    frameworks[0]->eye_estimator_->updateFineTuning();

    et::EyeDataToSend eye_data;
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

            frameworks[i]->getEyeDataPackage(eye_data);
//            data.push_back({eye_data.eye_centre.x, eye_data.eye_centre.y, eye_data.eye_centre.z, eye_data.cornea_centre.x, eye_data.cornea_centre.y, eye_data.cornea_centre.z});

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
//    et::Utils::writeFloatCsv(data, "/mnt/c/Users/Marek/Documents/Projects/hdr_display/eye_tracker/experiments/data.csv");

    socket_server->closeSocket();
    cv::destroyAllWindows();
    et::Settings::saveSettings();


    return 0;
}
