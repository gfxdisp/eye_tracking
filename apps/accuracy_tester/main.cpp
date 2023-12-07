#include "eye_tracker/Settings.hpp"
#include "eye_tracker/frameworks/Framework.hpp"
#include "eye_tracker/Utils.hpp"
#include "eye_tracker/optimizers/PolynomialFit.hpp"
#include "eye_tracker/eye/PolynomialEyeEstimator.hpp"

#include <getopt.h>

#include <string>
#include <memory>

int main(int argc, char *argv[])
{
    constexpr option options[] = {{"settings-path", required_argument, nullptr, 's'},
                                  {"angle-error",   required_argument, nullptr, 'a'},
                                  {"detect-error",  required_argument, nullptr, 'd'},
                                  {"type",          required_argument, nullptr, 't'},
                                  {nullptr,         no_argument,       nullptr, 0}};

    int argument{0};
    std::string settings_path{"."};
    double angle_error{0};
    double detect_error{0};
    bool estimating_using_model = false;

    while (argument != -1) {
        argument = getopt_long(argc, argv, "s:a:d:t:", options, nullptr);
        switch (argument) {
            case 's':
                settings_path = optarg;
                break;
            case 'a':
                angle_error = std::stod(optarg);
                break;
            case 'd':
                detect_error = std::stod(optarg);
                break;
            case 't':
                if (strcmp(optarg, "model") == 0) {
                    estimating_using_model = true;
                } else if (strcmp(optarg, "poly") == 0) {
                    estimating_using_model = false;
                } else {
                    std::cerr << "Error: invalid type. Should be either \"model\" or \"poly\"." << std::endl;
                    return 1;
                }
                break;
            default:
                break;
        }
    }

    auto settings = std::make_shared<et::Settings>(settings_path);

    cv::Vec3d min_eye_pos = {190.0, 135.0, 815.0};
    cv::Vec3d max_eye_pos = {204.0, 151.0, 830.0};

    cv::Vec3d min_marker_pos = {170.0, 90.0, -100.0};
    cv::Vec3d max_marker_pos = {290.0, 210.0, 100.0};

    double min_alpha = -1.0;
    double max_alpha = 11.0;

    double min_beta = -4.5;
    double max_beta = 7.5;

    double min_eye_cornea_dist = 5.0;
    double max_eye_cornea_dist = 5.6;

    min_eye_cornea_dist = max_eye_cornea_dist = 5.3;

    double min_pupil_cornea_dist = 3.9;
    double max_pupil_cornea_dist = 4.5;

    min_pupil_cornea_dist = max_pupil_cornea_dist = 4.2;

    double min_cornea_radius = 7.5;
    double max_cornea_radius = 8.1;

    min_cornea_radius = max_cornea_radius = 7.8;

    double min_refraction_index = 1.3275;
    double max_refraction_index = 1.3475;

    min_refraction_index = max_refraction_index = 1.3375;

    // Init randomizer
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis_eye_x(min_eye_pos[0], max_eye_pos[0]);
    std::uniform_real_distribution<> dis_eye_y(min_eye_pos[1], max_eye_pos[1]);
    std::uniform_real_distribution<> dis_eye_z(min_eye_pos[2], max_eye_pos[2]);

    std::uniform_real_distribution<> dis_marker_x(min_marker_pos[0], max_marker_pos[0]);
    std::uniform_real_distribution<> dis_marker_y(min_marker_pos[1], max_marker_pos[1]);
    std::uniform_real_distribution<> dis_marker_z(min_marker_pos[2], max_marker_pos[2]);

    std::uniform_real_distribution<> dis_alpha(min_alpha, max_alpha);
    std::uniform_real_distribution<> dis_beta(min_beta, max_beta);

    std::uniform_real_distribution<> dis_eye_cornea_dist(min_eye_cornea_dist, max_eye_cornea_dist);
    std::uniform_real_distribution<> dis_pupil_cornea_dist(min_pupil_cornea_dist, max_pupil_cornea_dist);
    std::uniform_real_distribution<> dis_cornea_radius(min_cornea_radius, max_cornea_radius);
    std::uniform_real_distribution<> dis_refraction_index(min_refraction_index, max_refraction_index);

    std::normal_distribution<double> dis_pixel_error_angles(0.0, angle_error);
    std::normal_distribution<double> dis_pixel_error(0.0, detect_error);

    std::vector<double> alpha_errors{};
    std::vector<double> beta_errors{};
    std::vector<double> horizontal_errors{};
    std::vector<double> vertical_errors{};
    std::vector<double> eye_pos_errors{};

    et::EyeInfo eye_info{};
    cv::Point3d marker_pos{};
    cv::Point3d visual_axis{};
    cv::Point3d optical_axis{};
    cv::Point3d nodal_point{};
    cv::Point3d estimated_eye_centre, estimated_nodal_point, estimated_visual_axis;
    cv::Vec3d estimated_optical_axis{};
    double estimated_alpha, estimated_beta;

    auto optical_axis_optimizer = new et::OpticalAxisOptimizer();
    auto optical_axis_minimizer_function = cv::Ptr<cv::DownhillSolver::Function>(optical_axis_optimizer);
    auto optical_axis_solver = cv::DownhillSolver::create();
    optical_axis_solver->setFunction(optical_axis_minimizer_function);

    et::ModelEyeEstimator model_eye_estimator(0);
    et::PolynomialEyeEstimator polynomial_eye_estimator(0);
    std::vector<cv::Point3d> marker_positions;
    int markers_per_row = 5;
    int markers_per_col = 3;
    double depth = 15.0;


    for (int i = 0; i < markers_per_row; i++) {
        for (int j = 0; j < markers_per_col; j++) {
            cv::Point3d marker{};
            marker.x = min_marker_pos[0] + i * (max_marker_pos[0] - min_marker_pos[0]) / (markers_per_row - 1.0);
            marker.y = min_marker_pos[1] + j * (max_marker_pos[1] - min_marker_pos[1]) / (markers_per_col - 1.0);
            marker.z = depth;
            marker_positions.push_back(marker);
        }
    }

    std::cout << std::setprecision(3) << std::fixed;

    for (int i = 0; i < 10; i++) {
        cv::Point3d eye_centre = {dis_eye_x(gen), dis_eye_y(gen), dis_eye_z(gen)};

        double alpha = dis_alpha(gen);
        double beta = dis_beta(gen);

        et::EyeMeasurements eye_measurements = {
                .eye_cornea_dist = dis_eye_cornea_dist(gen), .pupil_cornea_dist = dis_pupil_cornea_dist(gen),
                .cornea_radius = dis_cornea_radius(gen), .refraction_index = dis_refraction_index(gen),
        };

        std::vector<double> estimated_alphas{};
        std::vector<double> estimated_betas{};
        std::vector<cv::Point2f> glints;
        for (int j = 0; j < marker_positions.size(); j++) {
            marker_pos = marker_positions[j];

            optical_axis_optimizer->setParameters(alpha, beta, eye_measurements.eye_cornea_dist, eye_centre,
                                                  marker_pos);

            cv::Vec3d init_guess = marker_pos - eye_centre;
            init_guess = init_guess / cv::norm(init_guess);

            cv::Mat x = (cv::Mat_<double>(1, 3) << init_guess[0], init_guess[1], init_guess[2]);
            cv::Mat step = cv::Mat::ones(x.rows, x.cols, CV_64F) * 0.1;
            optical_axis_solver->setInitStep(step);
            optical_axis_solver->minimize(x);

            optical_axis = cv::Vec3d(x.at<double>(0, 0), x.at<double>(0, 1), x.at<double>(0, 2));
            optical_axis = optical_axis / cv::norm(optical_axis);

            nodal_point = eye_centre + eye_measurements.eye_cornea_dist * optical_axis;

            model_eye_estimator.invertDetectEye(eye_info, nodal_point, eye_centre, eye_measurements);

            eye_info.pupil.x += dis_pixel_error_angles(rd);
            eye_info.pupil.y += dis_pixel_error_angles(rd);

            for (int k = 0; k < eye_info.glints.size(); k++) {
                eye_info.glints[k].x += dis_pixel_error_angles(rd);
                eye_info.glints[k].y += dis_pixel_error_angles(rd);
            }

            eye_info.glints.erase(eye_info.glints.begin() + 1, eye_info.glints.end() - 1);

            model_eye_estimator.detectEye(eye_info, estimated_nodal_point, estimated_eye_centre,
                                          estimated_visual_axis);

            estimated_optical_axis = estimated_nodal_point - estimated_eye_centre;
            estimated_optical_axis = estimated_optical_axis / cv::norm(estimated_optical_axis);

            estimated_visual_axis = marker_pos - estimated_nodal_point;
            estimated_visual_axis = estimated_visual_axis / cv::norm(estimated_visual_axis);

            et::Utils::getAnglesBetweenVectors(estimated_optical_axis, estimated_visual_axis, estimated_alpha,
                                               estimated_beta);
            estimated_alphas.push_back(estimated_alpha);
            estimated_betas.push_back(estimated_beta);
        }

        double std_alpha = et::Utils::getStdDev(estimated_alphas);
        double std_beta = et::Utils::getStdDev(estimated_betas);
        double mean_alpha =
                std::accumulate(estimated_alphas.begin(), estimated_alphas.end(), 0.0) / estimated_alphas.
                        size();
        double mean_beta = std::accumulate(estimated_betas.begin(), estimated_betas.end(), 0.0) / estimated_betas.
                size();

        estimated_alphas.erase(std::remove_if(estimated_alphas.begin(), estimated_alphas.end(),
                                              [mean_alpha, std_alpha](double x)
                                              {
                                                  return std::abs(x - mean_alpha) > 3.0 * std_alpha;
                                              }), estimated_alphas.end());

        estimated_betas.erase(std::remove_if(estimated_betas.begin(), estimated_betas.end(),
                                             [mean_beta, std_beta](double x)
                                             {
                                                 return std::abs(x - mean_beta) > 3.0 * std_beta;
                                             }), estimated_betas.end());

        estimated_alpha = std::accumulate(estimated_alphas.begin(), estimated_alphas.end(), 0.0) / estimated_alphas.
                size();

        estimated_beta = std::accumulate(estimated_betas.begin(), estimated_betas.end(), 0.0) / estimated_betas.
                size();

        alpha_errors.push_back(std::abs(estimated_alpha - alpha));
        beta_errors.push_back(std::abs(estimated_beta - beta));

        std::vector<cv::Point2d> pupils;
        std::vector<cv::RotatedRect> ellipses;
        std::vector<cv::Point3d> eye_centres;
        std::vector<cv::Point3d> nodal_points;
        std::vector<cv::Vec3d> visual_axes;
        cv::RotatedRect ellipse;

        if (!estimating_using_model) {
            for (int j = 0; j < 10000; j++) {
                marker_pos = {dis_marker_x(gen), dis_marker_y(gen), dis_marker_z(gen)};
                eye_centre = {dis_eye_x(gen), dis_eye_y(gen), dis_eye_z(gen)};

                optical_axis_optimizer->setParameters(estimated_alpha, estimated_beta,
                                                      eye_measurements.eye_cornea_dist, eye_centre,
                                                      marker_pos);

                cv::Vec3d init_guess = marker_pos - eye_centre;
                init_guess = init_guess / cv::norm(init_guess);

                cv::Mat x = (cv::Mat_<double>(1, 3) << init_guess[0], init_guess[1], init_guess[2]);
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
                nodal_points.push_back(nodal_point);
                visual_axes.push_back(visual_axis);
                std::cout << j << " / 10000" << std::endl;
            }
            polynomial_eye_estimator.fitModel(pupils, ellipses, eye_centres, nodal_points, visual_axes);
        }

        // Error test
        for (int j = 0; j < 10000; j++) {
            marker_pos = {dis_marker_x(gen), dis_marker_y(gen), dis_marker_z(gen)};
            nodal_point = {dis_eye_x(gen), dis_eye_y(gen), dis_eye_z(gen)};
            visual_axis = marker_pos - nodal_point;
            visual_axis = visual_axis / cv::norm(visual_axis);

            optical_axis = et::Utils::visualToOpticalAxis(visual_axis, alpha, beta);
            eye_centre = nodal_point - optical_axis * eye_measurements.eye_cornea_dist;
            model_eye_estimator.invertDetectEye(eye_info, nodal_point, eye_centre, eye_measurements);

            eye_info.pupil.x += dis_pixel_error(rd);
            eye_info.pupil.y += dis_pixel_error(rd);

            for (int k = 0; k < eye_info.glints.size(); k++) {
                eye_info.glints[k].x += dis_pixel_error(rd);
                eye_info.glints[k].y += dis_pixel_error(rd);
            }

            if (estimating_using_model) {
                eye_info.glints.erase(eye_info.glints.begin() + 1, eye_info.glints.end() - 1);
                model_eye_estimator.detectEye(eye_info, estimated_nodal_point, estimated_eye_centre,
                                              estimated_visual_axis);
            } else {
                glints.clear();
                for (const auto &glint: eye_info.glints) {
                    glints.push_back(glint);
                }
                eye_info.ellipse = cv::fitEllipse(glints);
                polynomial_eye_estimator.detectEye(eye_info, estimated_nodal_point, estimated_eye_centre,
                                                   estimated_visual_axis, estimated_alpha,
                                                   estimated_beta);
            }

            estimated_optical_axis = estimated_nodal_point - estimated_eye_centre;
            estimated_optical_axis = estimated_optical_axis / cv::norm(estimated_optical_axis);

            estimated_visual_axis = et::Utils::opticalToVisualAxis(estimated_optical_axis, estimated_alpha,
                                                                   estimated_beta);
            estimated_visual_axis = estimated_visual_axis / cv::norm(estimated_visual_axis);

            double k = (marker_pos.z - estimated_nodal_point.z) / estimated_visual_axis.z;
            cv::Point3d estimated_marker_pos = estimated_nodal_point + k * estimated_visual_axis;

            cv::Vec3d v1 = marker_pos - nodal_point;
            v1 = v1 / cv::norm(v1);
            cv::Vec3d v2 = estimated_marker_pos - nodal_point;
            v2 = v2 / cv::norm(v2);

            double phi, theta;
            et::Utils::getAnglesBetweenVectors(v1, v2, phi, theta);

            horizontal_errors.push_back(std::abs(phi));
            vertical_errors.push_back(std::abs(theta));
            eye_pos_errors.push_back(cv::norm(estimated_eye_centre - eye_centre));
        }
        std::cout << i << " / 1000" << std::endl;
    }

    std::cout << "Alpha error: " << std::accumulate(alpha_errors.begin(), alpha_errors.end(), 0.0) / alpha_errors.
            size() << " ± " << et::Utils::getStdDev(alpha_errors) << " ("
              << et::Utils::getPercentile(alpha_errors, 0.99)
              << ") " << std::endl;

    std::cout << "Beta error: " << std::accumulate(beta_errors.begin(), beta_errors.end(), 0.0) / beta_errors.size()
              << " ± " << et::Utils::getStdDev(beta_errors) << " (" << et::Utils::getPercentile(beta_errors, 0.99)
              << ") "
              << std::endl;

    std::cout << "Horizontal error: " << std::accumulate(horizontal_errors.begin(), horizontal_errors.end(), 0.0) /
                                         horizontal_errors.size() << " ± " << et::Utils::getStdDev(horizontal_errors)
              << " (" << et::Utils::getPercentile(horizontal_errors, 0.99) << ") " << std::endl;

    std::cout << "Vertical error: " << std::accumulate(vertical_errors.begin(), vertical_errors.end(), 0.0) /
                                       vertical_errors.size() << " ± " << et::Utils::getStdDev(vertical_errors) << " ("
              << et::Utils::getPercentile(vertical_errors, 0.99) << ") " << std::endl;

    std::cout << "Eye pos error: " << std::accumulate(eye_pos_errors.begin(), eye_pos_errors.end(), 0.0) /
                                      eye_pos_errors.size() << " ± " << et::Utils::getStdDev(eye_pos_errors) << " ("
              << et::Utils::getPercentile(eye_pos_errors, 0.99) << ") " << std::endl;

    return 0;
}
