#include "eye_tracker/Settings.hpp"
#include "eye_tracker/frameworks/Framework.hpp"
#include "eye_tracker/frameworks/RandomImagesFramework.hpp"
#include "eye_tracker/input/InputImages.hpp"
#include "eye_tracker/image/BlenderDiscreteFeatureAnalyser.hpp"
#include "eye_tracker/Utils.hpp"
#include "eye_tracker/optimizers/PolynomialFit.hpp"
#include "eye_tracker/eye/PolynomialEyeEstimator.hpp"
#include "eye_tracker/optimizers/AggregatedPolynomialOptimizer.hpp"

#include <getopt.h>

#include <string>
#include <memory>
#include <thread>
#include <future>


std::mutex mutex;

void polynomialCalculator(std::string model_path, int model_num, int camera_id,
                          std::vector<std::vector<double>> &setups_params,
                          std::unordered_map<int, et::PolynomialParams> *polynomial_params, bool headless);

int main(int argc, char *argv[])
{
    constexpr option options[] = {{"settings-path", required_argument, nullptr, 's'},
                                  {"models-path",   required_argument, nullptr, 'm'},
                                  {"headless",      no_argument,       nullptr, 'h'},
                                  {"threads",       required_argument, nullptr, 't'},
                                  {"begin",         required_argument, nullptr, 'b'},
                                  {"end",           required_argument, nullptr, 'e'},
                                  {nullptr,         no_argument,       nullptr, 0}};

    int argument{0};
    std::string settings_path{"."};
    std::string models_path{"."};
    bool headless{false};
    int num_threads{1};
    int begin = -1;
    int end = -1;

    while (argument != -1)
    {
        argument = getopt_long(argc, argv, "s:m:ht:b:e:", options, nullptr);
        switch (argument)
        {
            case 's':
                settings_path = optarg;
                break;
            case 'm':
                models_path = optarg;
                break;
            case 'h':
                headless = true;
                break;
            case 't':
                num_threads = std::stoi(optarg);
                if (num_threads > 1)
                {
                    headless = true;
                }
                break;
            case 'b':
                begin = std::stoi(optarg);
                break;
            case 'e':
                end = std::stoi(optarg);
                break;
            default:
                break;
        }
    }

    std::vector<std::thread> threads;

    auto settings = std::make_shared<et::Settings>(settings_path);
    std::string camera_names[] = {"left", "right"};
    for (int camera_id = 0; camera_id < 2; camera_id++)
    {
        et::Settings::parameters.user_params[camera_id] = &et::Settings::parameters.features_params[camera_id]["blender"];
        auto current_eye_path = std::filesystem::path(models_path) / ("setups_" + camera_names[camera_id]);

        if (!std::filesystem::exists(current_eye_path))
        {
            continue;
        }

        auto polynomial_params = &settings->parameters.polynomial_params[camera_id];
        auto setups_params = et::Utils::readFloatRowsCsv(std::filesystem::path(current_eye_path) / "setups_params.csv",
                                                         true);
        // Loop over all folders in models_path
        for (const auto &entry: std::filesystem::directory_iterator(current_eye_path))
        {
            // If the folder name is a number, then it is a model
            if (!entry.is_directory())
            {
                continue;
            }
            std::string model_name = entry.path().filename().string();
            if (!std::all_of(model_name.begin(), model_name.end(), ::isdigit))
            {
                continue;
            }
            int model_num = std::stoi(model_name);

            if (begin != -1 && begin > model_num)
            {
                continue;
            }
            if (end != -1 && end < model_num)
            {
                continue;
            }


            while (threads.size() == num_threads)
            {
                for (int j = 0; j < threads.size(); j++)
                {
                    auto future = std::async(std::launch::async, &std::thread::join, &threads[j]);
                    if (future.wait_for(std::chrono::seconds(1)) != std::future_status::timeout)
                    {
                        threads.erase(threads.begin() + j);
                        break;
                    }
                }
            }

            threads.emplace_back(polynomialCalculator, entry.path().string(), model_num, camera_id,
                                 std::ref(setups_params), polynomial_params, headless);
        }

        while (threads.size() != 0)
        {
            for (int i = 0; i < threads.size(); i++)
            {
                auto future = std::async(std::launch::async, &std::thread::join, &threads[i]);
                if (future.wait_for(std::chrono::seconds(5)) != std::future_status::timeout)
                {
                    threads.erase(threads.begin() + i);
                    break;
                }
            }
        }
    }

    return 0;
}

void polynomialCalculator(std::string model_path, int model_num, int camera_id,
                          std::vector<std::vector<double>> &setups_params,
                          std::unordered_map<int, et::PolynomialParams> *polynomial_params, bool headless)
{
    std::clog << "Loading model " << model_num << " for camera " << camera_id << std::endl;
    // Find row with first column equal to model_num

    int setup_row = 0;
    while (setup_row < setups_params.size() && setups_params[setup_row][0] != model_num)
    {
        setup_row++;
    }

    if (setup_row == setups_params.size())
    {
        std::cerr << "Error: model " << model_num << " not found in setups_params.csv" << std::endl;
        return;
    }

    auto setup_params = &setups_params[setup_row];

    et::SetupVariables setup_variables{setup_params->at(1), setup_params->at(2), setup_params->at(3),
                                       setup_params->at(4), setup_params->at(5)};

    et::EyeDataToSend eye_data_package{};
    std::vector<cv::Point2d> pupils{};
    std::vector<cv::RotatedRect> ellipses{};

    auto eye_features_path = std::filesystem::path(model_path) / "eye_features.csv";
    auto eye_features = et::Utils::readFloatColumnsCsv(eye_features_path);
    int current_row = 0;
    std::vector<cv::Point3d> eye_centres{};
    std::vector<cv::Point3d> nodal_points{};
    std::vector<cv::Vec3d> visual_axes{};

    auto framework = std::make_shared<et::RandomImagesFramework>(camera_id, headless, model_path);

    while (framework->analyzeNextFrame())
    {
        framework->updateUi();
        if (framework->wereFeaturesFound())
        {
            framework->getEyeDataPackage(eye_data_package);

            while (eye_data_package.frame_num > eye_features[0][current_row])
            {
                current_row++;
            }

            if (eye_data_package.frame_num != eye_features[0][current_row])
            {
                continue;
            }

            pupils.push_back(eye_data_package.pupil);
            ellipses.push_back(eye_data_package.ellipse);

            cv::Point3d nodal_point = {eye_features[1][current_row], eye_features[2][current_row],
                                       eye_features[3][current_row]};
            cv::Point3d eye_centre = {eye_features[4][current_row], eye_features[5][current_row],
                                      eye_features[6][current_row]};

            cv::Vec3d optical_axis = nodal_point - eye_centre;
            optical_axis = optical_axis / cv::norm(optical_axis);
            cv::Vec3d visual_axis = et::Utils::opticalToVisualAxis(optical_axis, setup_variables.alpha,
                                                                   setup_variables.beta);

            nodal_points.push_back(nodal_point);
            eye_centres.push_back(eye_centre);
            visual_axes.push_back(visual_axis);
        }

        if (!headless)
        {
            int key_pressed = cv::waitKey() & 0xFFFF;
            switch (key_pressed)
            {
                case 'w':
                    framework->switchToCameraImage();
                    break;
                case 'e':
                    framework->switchToPupilThreshImage();
                    break;
                case 'r':
                    framework->switchToGlintThreshImage();
                    break;
                default:
                    break;
            }
        }
    }

    if (nodal_points.size() < 1000)
    {
        std::cerr << "Error: not enough data for model " << model_num << " for camera " << camera_id << std::endl;
        return;
    }

    std::shared_ptr<et::PolynomialEyeEstimator> polynomial_eye_estimator = std::make_shared<et::PolynomialEyeEstimator>(
            camera_id);

    bool result = polynomial_eye_estimator->fitModel(pupils, ellipses, eye_centres, nodal_points, visual_axes);

    if (!result)
    {
        std::cerr << "Error: polynomial fit failed for model " << model_num << " for camera " << camera_id << std::endl;
        return;
    }

    mutex.lock();
    et::Settings::loadSettings();
    if (polynomial_params->contains(model_num))
    {
        polynomial_params->at(model_num).coefficients = polynomial_eye_estimator->getCoefficients();
        polynomial_params->at(model_num).setup_variables = setup_variables;
    }
    else
    {
        polynomial_params->emplace(model_num,
                                   et::PolynomialParams{polynomial_eye_estimator->getCoefficients(), setup_variables});
    }
    et::Settings::saveSettings();
    mutex.unlock();
}