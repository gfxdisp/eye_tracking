#include <getopt.h>
#include <string>
#include <memory>
#include "eye_tracker/Settings.hpp"
#include "eye_tracker/optimizers/OpticalAxisOptimizer.hpp"
#include "eye_tracker/Utils.hpp"
#include "eye_tracker/eye/EyeEstimator.hpp"
#include "eye_tracker/eye/ModelEyeEstimator.hpp"
#include "eye_tracker/eye/PolynomialEyeEstimator.hpp"

int main(int argc, char *argv[])
{
    constexpr option options[] = {{"settings-path", required_argument, nullptr, 's'},
                                  {nullptr,         no_argument,       nullptr, 0}};

    int argument{0};
    std::string settings_path{"."};
    std::string user{"default"};

    while (argument != -1) {
        argument = getopt_long(argc, argv, "s:", options, nullptr);
        switch (argument) {
            case 's':
                settings_path = optarg;
                break;
            default:
                break;
        }
    }

    int camera_id = 0;

    auto settings = std::make_shared<et::Settings>(settings_path);
    auto optical_axis_optimizer = new et::OpticalAxisOptimizer();
    auto optical_axis_minimizer_function = cv::Ptr<cv::DownhillSolver::Function>(optical_axis_optimizer);
    auto optical_axis_solver = cv::DownhillSolver::create();
    optical_axis_solver->setFunction(optical_axis_minimizer_function);

    et::EyeInfo eye_info{};
    cv::Point3d marker_pos{};
    cv::Point3d visual_axis{};
    cv::Point3d optical_axis{};
    cv::Point3d nodal_point{};
    cv::Point3d eye_centre{};
    cv::Vec2d angle{};

    std::vector<cv::Point3d> cross_positions = {
            {180.0, 100.0, 100.0},
            {180.0, 150.0, 100.0},
            {180.0, 200.0, 100.0},
            {205.0, 100.0, 100.0},
            {205.0, 150.0, 100.0},
            {205.0, 200.0, 100.0},
            {230.0, 100.0, 100.0},
            {230.0, 150.0, 100.0},
            {230.0, 200.0, 100.0},
            {255.0, 100.0, 100.0},
            {255.0, 150.0, 100.0},
            {255.0, 200.0, 100.0},
            {280.0, 100.0, 100.0},
            {280.0, 150.0, 100.0},
            {280.0, 200.0, 100.0},
    };

    eye_centre = {179.7799, 106.3562, 834.0974};
    double alpha = 0.0;
    double beta = 0.0;

    et::EyeMeasurements eye_measurements = {
            .eye_cornea_dist = 5.3, .pupil_cornea_dist = 4.2,
            .cornea_radius = 7.8, .refraction_index = 1.3375,
    };

    et::ModelEyeEstimator model_eye_estimator(camera_id);
    et::PolynomialEyeEstimator polynomial_eye_estimator(camera_id);
    polynomial_eye_estimator.setModel("default");

    std::vector<cv::Point3d> eye_centres{};
    std::vector<cv::Point3d> mod_eye_centres{};
    std::vector<cv::Vec2d> angles{};
    std::vector<cv::Vec2d> mod_angles{};

    std::vector<cv::Point3d> estimated_eye_centres{};
    std::vector<cv::Vec2d> estimated_angles{};

    std::random_device rd;
    std::mt19937 gen(rd());

    std::normal_distribution<double> dis_pixel_error(0.0, 0.0);

    std::normal_distribution<double> dis_eye_centre_error(0.0, 0.0);
    std::normal_distribution<double> dis_marker_error(0.0, 0.0);

    for (int i = 0; i < cross_positions.size(); i++) {
        auto cross_position = cross_positions[i];
        cv::Point3d mod_cross_position = cross_position + cv::Point3d(dis_marker_error(gen), dis_marker_error(gen), 0.0);
        cv::Point3d mod_eye_centre = eye_centre + cv::Point3d(dis_eye_centre_error(gen), dis_eye_centre_error(gen), 0.0);

        optical_axis_optimizer->setParameters(alpha, beta, eye_measurements.eye_cornea_dist, mod_eye_centre, mod_cross_position);
        cv::Vec3d gaze_direction = mod_cross_position - mod_eye_centre;
        gaze_direction = gaze_direction / cv::norm(gaze_direction);
        et::Utils::vectorToAngles(gaze_direction, angle);
        mod_angles.push_back(angle);

        cv::Mat x = (cv::Mat_<double>(1, 3) << gaze_direction[0], gaze_direction[1], gaze_direction[2]);
        cv::Mat step = cv::Mat::ones(x.rows, x.cols, CV_64F) * 0.1;
        optical_axis_solver->setInitStep(step);
        optical_axis_solver->minimize(x);

        gaze_direction = cross_position - eye_centre;
        gaze_direction = gaze_direction / cv::norm(gaze_direction);
        et::Utils::vectorToAngles(gaze_direction, angle);
        angles.push_back(angle);

        optical_axis = cv::Vec3d(x.at<double>(0, 0), x.at<double>(0, 1), x.at<double>(0, 2));
        optical_axis = optical_axis / cv::norm(optical_axis);

        nodal_point = mod_eye_centre + eye_measurements.eye_cornea_dist * optical_axis;

        model_eye_estimator.invertDetectEye(eye_info, nodal_point, mod_eye_centre, eye_measurements);
        std::vector<cv::Point2f> glints{};
        for (int j = 0; j < eye_info.glints.size(); j++) {
            glints.push_back(eye_info.glints[j] + cv::Point2d(dis_pixel_error(gen), dis_pixel_error(gen)));
        }
        eye_info.ellipse = cv::fitEllipse(glints);

        eye_centres.push_back(eye_centre);
        mod_eye_centres.push_back(mod_eye_centre);

        eye_info.pupil.x += dis_pixel_error(gen);
        eye_info.pupil.y += dis_pixel_error(gen);
        eye_info.ellipse.center = {eye_info.ellipse.center.x, eye_info.ellipse.center.y};
        eye_info.ellipse.size = {eye_info.ellipse.size.width, eye_info.ellipse.size.height};
        eye_info.ellipse.angle = eye_info.ellipse.angle;

        cv::Point3d estimated_eye_centre;
        cv::Vec2d estimated_angle{};
        polynomial_eye_estimator.detectEye(eye_info, estimated_eye_centre, estimated_angle);
        while (estimated_angle[0] < -M_PI / 2) {
            estimated_angle[0] += 2 * M_PI;
        }
        while (estimated_angle[1] < -M_PI / 2) {
            estimated_angle[1] += 2 * M_PI;
        }
        estimated_eye_centres.push_back(estimated_eye_centre);
        estimated_angles.push_back(estimated_angle);
    }

    cv::Point3d eye_centre_offset{};
    cv::Vec2d angle_offset{};

    double train_fraction = 1.0;

    int train_start = 0;
    int train_end = (int)(estimated_eye_centres.size() * train_fraction);
    int test_start = train_end;
    int test_end = estimated_eye_centres.size();
    if (train_fraction == 1.0) {
        test_start = 0;
    }

    for (int i = train_start; i < train_end; i++) {
        eye_centre_offset += estimated_eye_centres[i] - eye_centres[i];
        angle_offset += estimated_angles[i] - angles[i];
    }
    eye_centre_offset.x /= (int)(estimated_eye_centres.size() * train_fraction);
    eye_centre_offset.y /= (int)(estimated_eye_centres.size() * train_fraction);
    eye_centre_offset.z /= (int)(estimated_eye_centres.size() * train_fraction);
    angle_offset[0] /= (int)(estimated_eye_centres.size() * train_fraction);
    angle_offset[1] /= (int)(estimated_eye_centres.size() * train_fraction);
    std::cout << "Eye centre offset: " << eye_centre_offset << "\n";
    std::cout << "Angles offset: " << angle_offset * 180 / CV_PI << "\n";

    std::vector<double> errors_x{};
    std::vector<double> errors_y{};
    std::vector<double> errors_z{};
    std::vector<double> errors_theta{};
    std::vector<double> errors_phi{};


    for (int i = test_start; i < test_end; i++) {
        errors_x.push_back(std::abs(estimated_eye_centres[i].x - mod_eye_centres[i].x - eye_centre_offset.x));
        errors_y.push_back(std::abs(estimated_eye_centres[i].y - mod_eye_centres[i].y - eye_centre_offset.y));
        errors_z.push_back(std::abs(estimated_eye_centres[i].z - mod_eye_centres[i].z - eye_centre_offset.z));
        errors_theta.push_back(std::abs(estimated_angles[i][0] - mod_angles[i][0] - angle_offset[0]));
        errors_phi.push_back(std::abs(estimated_angles[i][1] - mod_angles[i][1] - angle_offset[1]));
    }

    std::cout << std::fixed << std::setprecision(3);

    std::cout << "Error eye X: " << std::accumulate(errors_x.begin(), errors_x.end(), 0.0) / errors_x.size() << " ± " << et::Utils::getStdDev(errors_x) << "\n";
    std::cout << "Error eye Y: " << std::accumulate(errors_y.begin(), errors_y.end(), 0.0) / errors_y.size() << " ± " << et::Utils::getStdDev(errors_y) << "\n";
    std::cout << "Error eye Z: " << std::accumulate(errors_z.begin(), errors_z.end(), 0.0) / errors_z.size() << " ± " << et::Utils::getStdDev(errors_z) << "\n";

    std::cout << "Error theta: " << std::accumulate(errors_theta.begin(), errors_theta.end(), 0.0) / errors_theta.size() * 180 / CV_PI
              << " ± " << et::Utils::getStdDev(errors_theta) * 180 / CV_PI << "\n";

    std::cout << "Error phi: " << std::accumulate(errors_phi.begin(), errors_phi.end(), 0.0) / errors_phi.size() * 180 / CV_PI
              << " ± " << et::Utils::getStdDev(errors_phi) * 180 / CV_PI << std::endl;

    return 0;
}