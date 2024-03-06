#include "eye_tracker/Settings.hpp"
#include "eye_tracker/frameworks/Framework.hpp"
#include "eye_tracker/Utils.hpp"
#include "eye_tracker/optimizers/PolynomialFit.hpp"
#include "eye_tracker/eye/PolynomialEyeEstimator.hpp"
#include "eye_tracker/eye/ModelEyeEstimator.hpp"

#include <getopt.h>

#include <string>
#include <memory>
#include <eye_tracker/optimizers/CorneaOptimizer.hpp>
#include <eye_tracker/optimizers/EyeForPixelOptimizer.hpp>

int main(int argc, char *argv[])
{
    constexpr option options[] = {
            {"settings-path", required_argument, nullptr, 's'},
            {"alpha",         required_argument, nullptr, 'a'},
            {"beta",          required_argument, nullptr, 'b'},
            {"eye",           required_argument, nullptr, 'e'},
            {"id",            required_argument, nullptr, 'i'},
            {nullptr,         no_argument,       nullptr, 0}
    };

    int argument{0};
    std::string settings_path{"."};
    double alpha{0};
    double beta{0};
    int camera_id{0};
    std::string user_id{""};

    while (argument != -1) {
        argument = getopt_long(argc, argv, "s:a:b:e:i:", options, nullptr);
        switch (argument) {
            case 's':
                settings_path = optarg;
                break;
            case 'a':
                alpha = std::stod(optarg);
                break;
            case 'b':
                beta = std::stod(optarg);
                break;
            case 'e':
                if (strcmp(optarg, "left") == 0) {
                    camera_id = 0;
                } else if (strcmp(optarg, "right") == 0) {
                    camera_id = 1;
                } else {
                    std::cerr << "Error: invalid eye" << std::endl;
                    return 1;
                }
                break;
            case 'i':
                user_id = optarg;
                break;
            default:
                break;
        }
    }

    et::EyeInfo eye_info{};
    cv::Point3d marker_pos{};
    cv::Point3d visual_axis{};
    cv::Point3d optical_axis{};
    cv::Point3d nodal_point{};
    cv::Point3d eye_centre{};
    cv::Vec2d angle{};

    auto settings = std::make_shared<et::Settings>(settings_path);

    et::PolynomialEyeEstimator polynomial_eye_estimator(camera_id);
    std::vector<cv::Point2d> pupils;
    std::vector<cv::RotatedRect> ellipses;
    std::vector<cv::Point3d> eye_centres;
    std::vector<cv::Vec2d> angles;
    cv::RotatedRect ellipse;

    auto cornea_optimizer = new et::CorneaOptimizer();
    auto cornea_minimizer_function = cv::Ptr<cv::DownhillSolver::Function>(cornea_optimizer);
    auto cornea_solver = cv::DownhillSolver::create();

    cornea_solver->setFunction(cornea_minimizer_function);
    cornea_solver->setTermCriteria(
            cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 1000, std::numeric_limits<float>::min()));

    et::EyeMeasurements eye_measurements = {
            .cornea_centre_distance = 5.3, .cornea_curvature_radius = 7.8, .cornea_refraction_index = 1.3375,
            .pupil_cornea_distance = 4.2, .alpha = alpha, .beta = beta
    };

    et::ModelEyeEstimator model_eye_estimator(camera_id);


    cv::Vec3d min_eye_pos = {100.0, 120.0, 830.0};
    cv::Vec3d max_eye_pos = {160.0, 180.0, 1030.0};

    cv::Vec2d min_rotation = {-15, -15};
    cv::Vec2d max_rotation = {15, 15};

    double width = et::Settings::parameters.camera_params[camera_id].dimensions.width;
    double height = et::Settings::parameters.camera_params[camera_id].dimensions.height;

    cv::Point2d min_pixel_pos = {0, 0};
    cv::Point2d max_pixel_pos = {width, height};

    auto eye_for_pixel_optimizer = new et::EyeForPixelOptimizer();
    auto eye_for_pixel_minimizer_function = cv::Ptr<cv::DownhillSolver::Function>(eye_for_pixel_optimizer);
    auto eye_for_pixel_solver = cv::DownhillSolver::create();
    eye_for_pixel_solver->setFunction(eye_for_pixel_minimizer_function);
    eye_for_pixel_solver->setTermCriteria(
            cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 1000, std::numeric_limits<float>::min()));

    cv::Mat x = (cv::Mat_<double>(1, 2) << min_eye_pos[0], max_eye_pos[1]);
    cv::Mat step = cv::Mat::ones(x.rows, x.cols, CV_64F) * 0.1;
    cv::Point2d pixel_pos;
    double depth;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            for (int k = 0; k < 5; k++) {
                x = (cv::Mat_<double>(1, 2) << min_eye_pos[0], min_eye_pos[1]);
                pixel_pos.x = (min_pixel_pos.x + (max_pixel_pos.x - min_pixel_pos.x) / 5.0 * i);
                pixel_pos.y = (min_pixel_pos.y + (max_pixel_pos.y - min_pixel_pos.y) / 5.0 * j);
                depth = min_eye_pos[2] + (max_eye_pos[2] - min_eye_pos[2]) / 5.0 * k;
                eye_for_pixel_optimizer->setParameters(std::make_shared<et::ModelEyeEstimator>(camera_id), pixel_pos,
                                                       et::Settings::parameters.polynomial_params[camera_id].eye_measurements,
                                                       depth);
                eye_for_pixel_solver->setInitStep(step);
                eye_for_pixel_solver->minimize(x);
                if (i == 0 && j == 0 && k == 0) {
                    min_eye_pos[0] = x.at<double>(0, 0);
                    max_eye_pos[0] = x.at<double>(0, 0);
                    min_eye_pos[1] = x.at<double>(0, 1);
                    max_eye_pos[1] = x.at<double>(0, 1);
                } else {
                    min_eye_pos[0] = std::min(x.at<double>(0, 0), min_eye_pos[0]);
                    max_eye_pos[0] = std::max(x.at<double>(0, 0), max_eye_pos[0]);
                    min_eye_pos[1] = std::min(x.at<double>(0, 1), min_eye_pos[1]);
                    max_eye_pos[1] = std::max(x.at<double>(0, 1), max_eye_pos[1]);
                }
            }
        }
    }
    min_eye_pos[0] -= 50;
    max_eye_pos[0] += 50;
    min_eye_pos[1] -= 50;
    max_eye_pos[1] += 50;

    std::cout << "Eye poses range: " << min_eye_pos << " - " << max_eye_pos << std::endl;


    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis_eye_x(min_eye_pos[0], max_eye_pos[0]);
    std::uniform_real_distribution<> dis_eye_y(min_eye_pos[1], max_eye_pos[1]);
    std::uniform_real_distribution<> dis_eye_z(min_eye_pos[2], max_eye_pos[2]);

    std::uniform_real_distribution<> dis_rotation_hor(min_rotation[0], max_rotation[0]);
    std::uniform_real_distribution<> dis_rotation_vert(min_rotation[1], max_rotation[1]);



    std::vector<cv::Point2f> glints{};

    cv::Point2d min_pupil;
    cv::Point2d max_pupil;
    cv::RotatedRect min_ellipse;
    cv::RotatedRect max_ellipse;

    int total_images = 100'000;
    for (int j = 0; j < total_images; j++) {
        angle = {dis_rotation_hor(gen), dis_rotation_vert(gen)};
        eye_centre = {dis_eye_x(gen), dis_eye_y(gen), dis_eye_z(gen)};

        double vert = 100 * std::tan(angle[1] * M_PI / 180.0);
        double hor = 100 * std::tan(angle[0] * M_PI / 180.0);
        marker_pos = eye_centre + cv::Point3d(hor, vert, -100);

        cornea_optimizer->setParameters(eye_measurements, eye_centre, marker_pos);


        x = (cv::Mat_<double>(1, 2) << 0.0, 0.0);
        step = cv::Mat::ones(x.rows, x.cols, CV_64F) * 0.1;
        cornea_solver->setInitStep(step);
        cornea_solver->minimize(x);

        nodal_point.x = eye_centre.x -
                        eye_measurements.cornea_curvature_radius * sin(x.at<double>(0, 0)) * cos(x.at<double>(0, 1));
        nodal_point.y = eye_centre.y + eye_measurements.cornea_curvature_radius * sin(x.at<double>(0, 1));
        nodal_point.z = eye_centre.z -
                        eye_measurements.cornea_curvature_radius * cos(x.at<double>(0, 0)) * cos(x.at<double>(0, 1));

        model_eye_estimator.invertDetectEye(eye_info, nodal_point, eye_centre, eye_measurements);
        visual_axis = marker_pos - nodal_point;
        visual_axis = visual_axis / cv::norm(visual_axis);
        et::Utils::vectorToAngles(visual_axis, angle);

        glints.clear();
        for (const auto &glint: eye_info.glints) {
            glints.push_back(glint);
        }

        ellipse = cv::fitEllipse(glints);
        if (ellipse.size.width < 1 || ellipse.size.height < 1) {
            j--;
            continue;
        }
        if (ellipse.size.width > 200 || ellipse.size.height > 200) {
            j--;
            continue;
        }

        pupils.push_back(eye_info.pupil);
        ellipses.push_back(ellipse);
        eye_centres.push_back(eye_centre);
        angles.push_back(angle);
        if (j % 100 == 0) {
            std::cout << j << " / " << total_images << std::endl;
        }
        if (j == 0) {
            min_pupil = eye_info.pupil;
            max_pupil = eye_info.pupil;
            min_ellipse = ellipse;
            max_ellipse = ellipse;
        } else {
            if (eye_info.pupil.x < min_pupil.x) {
                min_pupil.x = eye_info.pupil.x;
            }
            if (eye_info.pupil.y < min_pupil.y) {
                min_pupil.y = eye_info.pupil.y;
            }
            if (eye_info.pupil.x > max_pupil.x) {
                max_pupil.x = eye_info.pupil.x;
            }
            if (eye_info.pupil.y > max_pupil.y) {
                max_pupil.y = eye_info.pupil.y;
            }
            if (ellipse.size.width < min_ellipse.size.width) {
                min_ellipse.size.width = ellipse.size.width;
            }
            if (ellipse.size.height < min_ellipse.size.height) {
                min_ellipse.size.height = ellipse.size.height;
            }
            if (ellipse.size.width > max_ellipse.size.width) {
                max_ellipse.size.width = ellipse.size.width;
            }
            if (ellipse.size.height > max_ellipse.size.height) {
                max_ellipse.size.height = ellipse.size.height;
            }
            if (ellipse.center.x < min_ellipse.center.x) {
                min_ellipse.center.x = ellipse.center.x;
            }
            if (ellipse.center.y < min_ellipse.center.y) {
                min_ellipse.center.y = ellipse.center.y;
            }
            if (ellipse.center.x > max_ellipse.center.x) {
                max_ellipse.center.x = ellipse.center.x;
            }
            if (ellipse.center.y > max_ellipse.center.y) {
                max_ellipse.center.y = ellipse.center.y;
            }
        }
    }
    std::cout << "Pupil range: " << min_pupil << " - " << max_pupil << std::endl;
    std::cout << "Ellipse size: " << min_ellipse.size << " - " << max_ellipse.size << std::endl;
    std::cout << "Ellipse center: " << min_ellipse.center << " - " << max_ellipse.center << std::endl;
    polynomial_eye_estimator.fitModel(pupils, ellipses, eye_centres, angles);

    auto polynomial_params = &et::Settings::parameters.polynomial_params[camera_id];

    polynomial_params->coefficients = polynomial_eye_estimator.getCoefficients();
    polynomial_params->eye_measurements = eye_measurements;
    et::Settings::saveSettings();

    return 0;
}
