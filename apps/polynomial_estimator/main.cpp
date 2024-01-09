#include "eye_tracker/Settings.hpp"
#include "eye_tracker/frameworks/Framework.hpp"
#include "eye_tracker/Utils.hpp"
#include "eye_tracker/optimizers/PolynomialFit.hpp"
#include "eye_tracker/eye/PolynomialEyeEstimator.hpp"
#include "eye_tracker/eye/ModelEyeEstimator.hpp"

#include <getopt.h>

#include <string>
#include <memory>

int main(int argc, char *argv[])
{
    constexpr option options[] = {{"settings-path", required_argument, nullptr, 's'},
                                  {"alpha",         required_argument, nullptr, 'a'},
                                  {"beta",          required_argument, nullptr, 'b'},
                                  {"eye",           required_argument, nullptr, 'e'},
                                  {"id",            required_argument, nullptr, 'i'},
                                  {nullptr,         no_argument,       nullptr, 0}};

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

    cv::Vec3d min_eye_pos = {170.0, 95.0, 800.0};
    cv::Vec3d max_eye_pos = {190.0, 125.0, 850.0};

    cv::Vec3d min_marker_pos = {170.0, 90.0, -100.0};
    cv::Vec3d max_marker_pos = {290.0, 210.0, 200.0};

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis_eye_x(min_eye_pos[0], max_eye_pos[0]);
    std::uniform_real_distribution<> dis_eye_y(min_eye_pos[1], max_eye_pos[1]);
    std::uniform_real_distribution<> dis_eye_z(min_eye_pos[2], max_eye_pos[2]);

    std::uniform_real_distribution<> dis_marker_x(min_marker_pos[0], max_marker_pos[0]);
    std::uniform_real_distribution<> dis_marker_y(min_marker_pos[1], max_marker_pos[1]);
    std::uniform_real_distribution<> dis_marker_z(min_marker_pos[2], max_marker_pos[2]);

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

    auto optical_axis_optimizer = new et::OpticalAxisOptimizer();
    auto optical_axis_minimizer_function = cv::Ptr<cv::DownhillSolver::Function>(optical_axis_optimizer);
    auto optical_axis_solver = cv::DownhillSolver::create();
    optical_axis_solver->setFunction(optical_axis_minimizer_function);

    et::EyeMeasurements eye_measurements = {
            .eye_cornea_dist = 5.3, .pupil_cornea_dist = 4.2,
            .cornea_radius = 7.8, .refraction_index = 1.3375,
    };

    et::ModelEyeEstimator model_eye_estimator(camera_id);
    std::vector<cv::Point2f> glints{};

    for (int j = 0; j < 25000; j++) {
        marker_pos = {dis_marker_x(gen), dis_marker_y(gen), dis_marker_z(gen)};
        eye_centre = {dis_eye_x(gen), dis_eye_y(gen), dis_eye_z(gen)};

        if (j == 0) {
            eye_centre = {179.7799, 106.3562, 834.0974};
            marker_pos = {180.0, 100.0, 100.0};
        }

        optical_axis_optimizer->setParameters(alpha, beta,
                                              eye_measurements.eye_cornea_dist, eye_centre,
                                              marker_pos);

        cv::Vec3d gaze_direction = marker_pos - eye_centre;
        gaze_direction = gaze_direction / cv::norm(gaze_direction);

        et::Utils::vectorToAngles(gaze_direction, angle);

        cv::Mat x = (cv::Mat_<double>(1, 3) << gaze_direction[0], gaze_direction[1], gaze_direction[2]);
        cv::Mat step = cv::Mat::ones(x.rows, x.cols, CV_64F) * 0.1;
        optical_axis_solver->setInitStep(step);
        optical_axis_solver->minimize(x);

        optical_axis = cv::Vec3d(x.at<double>(0, 0), x.at<double>(0, 1), x.at<double>(0, 2));
        optical_axis = optical_axis / cv::norm(optical_axis);

        nodal_point = eye_centre + eye_measurements.eye_cornea_dist * optical_axis;

        visual_axis = marker_pos - nodal_point;
        visual_axis = visual_axis / cv::norm(visual_axis);

        model_eye_estimator.invertDetectEye(eye_info, nodal_point, eye_centre, eye_measurements);

        glints.clear();
        for (const auto &glint: eye_info.glints) {
            glints.push_back(glint);
        }

        ellipse = cv::fitEllipse(glints);

        pupils.push_back(eye_info.pupil);
        ellipses.push_back(ellipse);
        eye_centres.push_back(eye_centre);
        angles.push_back(angle);
        std::cout << j << " / 25000" << std::endl;
    }
    polynomial_eye_estimator.fitModel(pupils, ellipses, eye_centres, angles);

    et::SetupVariables setup_variables = {
            .cornea_centre_distance = eye_measurements.eye_cornea_dist,
            .cornea_curvature_radius = eye_measurements.cornea_radius,
            .cornea_refraction_index = eye_measurements.refraction_index,
            .alpha = alpha,
            .beta = beta,
    };

    auto polynomial_params = &et::Settings::parameters.polynomial_params[camera_id];

    if (polynomial_params->contains(user_id)) {
        polynomial_params->at(user_id).coefficients = polynomial_eye_estimator.getCoefficients();
        polynomial_params->at(user_id).setup_variables = setup_variables;
    } else {
        polynomial_params->emplace(user_id,
                                   et::PolynomialParams{polynomial_eye_estimator.getCoefficients(), setup_variables});
    }
    et::Settings::saveSettings();

    return 0;
}
