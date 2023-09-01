#include "eye_tracker/Settings.hpp"
#include "eye_tracker/frameworks/Framework.hpp"
#include "eye_tracker/frameworks/RandomImagesFramework.hpp"
#include "eye_tracker/input/InputImages.hpp"
#include "eye_tracker/image/BlenderDiscreteFeatureAnalyser.hpp"
#include "eye_tracker/Utils.hpp"
#include "eye_tracker/optimizers/PolynomialFit.hpp"

#include <getopt.h>

#include <string>
#include <memory>
#include <thread>
#include <future>


std::mutex mutex;

void
polynomialCalculator(std::string model_path, int model_num, int camera_id, std::vector<std::vector<float>> &setups_params,
                     std::unordered_map<int, et::PolynomialParams> *polynomial_params, bool headless);

int main(int argc, char *argv[])
{
    constexpr option options[] = {{"settings-path", required_argument, nullptr, 's'},
                                  {"models-path",   required_argument, nullptr, 'm'},
                                  {"headless",      no_argument,       nullptr, 'h'},
                                  {"threads",       required_argument, nullptr, 't'},
                                  {nullptr,         no_argument,       nullptr, 0}};

    int argument{0};
    std::string settings_path{"."};
    std::string models_path{"."};
    bool headless{false};
    int num_threads{1};

    while (argument != -1)
    {
        argument = getopt_long(argc, argv, "s:m:ht:", options, nullptr);
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
            default:
                break;
        }
    }

    std::vector<std::thread> threads;

    auto settings = std::make_shared<et::Settings>(settings_path);
    std::string camera_names[] = {"left", "right"};
    for (int camera_id = 0; camera_id < 2; camera_id++)
    {
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

            threads.emplace_back(polynomialCalculator, entry.path().string(), model_num, camera_id, std::ref(setups_params),
                                 polynomial_params, headless);
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

void
polynomialCalculator(std::string model_path, int model_num, int camera_id, std::vector<std::vector<float>> &setups_params,
                     std::unordered_map<int, et::PolynomialParams> *polynomial_params, bool headless)
{
    std::clog << "Loading model " << model_num << " for camera " << camera_id << std::endl;
    // Find row with first column equal to model_num

    int setup_row = 0;
    while (setup_row < setups_params[0].size() && setups_params[setup_row][0] != model_num)
    {
        setup_row++;
    }

    if (setup_row == setups_params[0].size())
    {
        std::cerr << "Error: model " << model_num << " not found in setups_params.csv" << std::endl;
        return;
    }

    auto setup_params = &setups_params[setup_row];

    et::EyeDataPackage eye_data_package{};
    std::vector<float> pupil_x{};
    std::vector<float> pupil_y{};
    std::vector<float> ellipse_x{};
    std::vector<float> ellipse_y{};
    std::vector<float> ellipse_width{};
    std::vector<float> ellipse_height{};
    std::vector<float> ellipse_angle{};

    auto eye_features_path = std::filesystem::path(model_path) / "eye_features.csv";
    auto eye_features = et::Utils::readFloatColumnsCsv(eye_features_path);
    int current_row = 0;
    std::vector<float> eye_centre_x{};
    std::vector<float> eye_centre_y{};
    std::vector<float> eye_centre_z{};
    std::vector<float> visual_axis_x{};
    std::vector<float> visual_axis_y{};
    std::vector<float> visual_axis_z{};

    et::PolynomialFit eye_centre_x_fit{5, 3};
    et::PolynomialFit eye_centre_y_fit{5, 3};
    et::PolynomialFit eye_centre_z_fit{5, 3};
    et::PolynomialFit visual_axis_x_fit{5, 3};
    et::PolynomialFit visual_axis_y_fit{5, 3};
    et::PolynomialFit visual_axis_z_fit{5, 3};

    auto framework = std::make_shared<et::RandomImagesFramework>(camera_id, headless, model_path);

    while (framework->analyzeNextFrame())
    {
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

            pupil_x.push_back(eye_data_package.pupil.x);
            pupil_y.push_back(eye_data_package.pupil.y);
            ellipse_x.push_back(eye_data_package.ellipse.center.x);
            ellipse_y.push_back(eye_data_package.ellipse.center.y);
            ellipse_width.push_back(eye_data_package.ellipse.size.width);
            ellipse_height.push_back(eye_data_package.ellipse.size.height);
            ellipse_angle.push_back(eye_data_package.ellipse.angle);

            visual_axis_x.push_back(eye_features[1][current_row]);
            visual_axis_y.push_back(eye_features[2][current_row]);
            visual_axis_z.push_back(eye_features[3][current_row]);
            eye_centre_x.push_back(eye_features[4][current_row]);
            eye_centre_y.push_back(eye_features[5][current_row]);
            eye_centre_z.push_back(eye_features[6][current_row]);
        }
    }

    bool result{true};
    std::vector<std::vector<float> *> input_vars{5};
    input_vars[0] = &pupil_x;
    input_vars[1] = &ellipse_x;
    input_vars[2] = &ellipse_width;
    input_vars[3] = &ellipse_height;
    input_vars[4] = &ellipse_angle;
    result &= eye_centre_x_fit.fit(input_vars, &eye_centre_x);
    result &= visual_axis_x_fit.fit(input_vars, &visual_axis_x);

    input_vars[0] = &pupil_y;
    input_vars[1] = &ellipse_y;
    result &= eye_centre_y_fit.fit(input_vars, &eye_centre_y);
    result &= visual_axis_y_fit.fit(input_vars, &visual_axis_y);

    input_vars[0] = &ellipse_x;
    result &= eye_centre_z_fit.fit(input_vars, &eye_centre_z);
    result &= visual_axis_z_fit.fit(input_vars, &visual_axis_z);

    if (!result)
    {
        return;
    }


    mutex.lock();
    if (polynomial_params->contains(model_num))
    {
        polynomial_params->at(model_num).coefficients = et::Coefficients{eye_centre_x_fit.getCoefficients(),
                                                                         eye_centre_y_fit.getCoefficients(),
                                                                         eye_centre_z_fit.getCoefficients(),
                                                                         visual_axis_x_fit.getCoefficients(),
                                                                         visual_axis_y_fit.getCoefficients(),
                                                                         visual_axis_z_fit.getCoefficients()};
        polynomial_params->at(model_num).setup_variables = et::SetupVariables{
                cv::Mat{4, 4, CV_32F, setup_params->data() + 1}, cv::Mat{3, 3, CV_32F, setup_params->data() + 17},
                setup_params->at(26), setup_params->at(27), setup_params->at(28), setup_params->at(29),
                setup_params->at(30)};
    }
    else
    {
        polynomial_params->emplace(model_num, et::PolynomialParams{
                et::Coefficients{eye_centre_x_fit.getCoefficients(), eye_centre_y_fit.getCoefficients(),
                                 eye_centre_z_fit.getCoefficients(), visual_axis_x_fit.getCoefficients(),
                                 visual_axis_y_fit.getCoefficients(), visual_axis_z_fit.getCoefficients()},
                et::SetupVariables{cv::Mat{4, 4, CV_32F, setup_params->data() + 1},
                                   cv::Mat{3, 3, CV_32F, setup_params->data() + 17}, setup_params->at(26),
                                   setup_params->at(27), setup_params->at(28), setup_params->at(29),
                                   setup_params->at(30)}});
    }
    et::Settings::saveSettings();
    mutex.unlock();
}