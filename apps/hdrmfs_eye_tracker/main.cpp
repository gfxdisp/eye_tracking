#include <eye_tracker/Settings.hpp>
#include "eye_tracker/frameworks/Framework.hpp"
#include "eye_tracker/image/temporal_filter/ContinuousTemporalFilterer.hpp"
#include "eye_tracker/SocketServer.hpp"
#include "eye_tracker/frameworks/OnlineCameraFramework.hpp"
#include "eye_tracker/frameworks/VideoCameraFramework.hpp"

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
//        frameworks[i] = std::make_shared<et::VideoCameraFramework>(i, headless, "/mnt/d/Downloads/et_test/2024-02-28_17-38-58_0.mp4", true);
    }


//    frameworks[0] = std::make_shared<et::VideoCameraFramework>(0, headless, "/mnt/d/Downloads/et_videos_2024-04-20_1/2024-04-20_15-56-59_0.mp4", true);
//    frameworks[1] = std::make_shared<et::VideoCameraFramework>(1, headless, "/mnt/d/Downloads/et_videos_2024-04-20_1/2024-04-20_16-22-52_1.mp4", true);

    auto socket_server = std::make_shared<et::SocketServer>(frameworks[0], frameworks[1]);
    socket_server->startServer();

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

    socket_server->closeSocket();
    cv::destroyAllWindows();
    et::Settings::saveSettings();


    return 0;
}