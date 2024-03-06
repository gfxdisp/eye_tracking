#include "eye_tracker/optimizers/MetaModel.hpp"
#include "eye_tracker/image/CameraFeatureAnalyser.hpp"
#include "eye_tracker/Utils.hpp"
#include "eye_tracker/eye/EyeEstimator.hpp"
#include "eye_tracker/eye/PolynomialEyeEstimator.hpp"
#include "eye_tracker/frameworks/Framework.hpp"
#include "eye_tracker/eye/ModelEyeEstimator.hpp"

#include <fstream>

namespace et
{
    MetaModel::MetaModel(int camera_id) : camera_id_(camera_id)
    {
        visual_angles_optimizer_ = new VisualAnglesOptimizer();
        visual_angles_minimizer_function_ = cv::Ptr<cv::DownhillSolver::Function>(visual_angles_optimizer_);
        visual_angles_solver_ = cv::DownhillSolver::create();
        visual_angles_solver_->setFunction(visual_angles_minimizer_function_);

        optical_axis_optimizer_ = new OpticalAxisOptimizer();
        optical_axis_minimizer_function_ = cv::Ptr<cv::DownhillSolver::Function>(optical_axis_optimizer_);
        optical_axis_solver_ = cv::DownhillSolver::create();
        optical_axis_solver_->setFunction(optical_axis_minimizer_function_);

        eye_angles_optimizer_ = new EyeAnglesOptimizer();
        eye_angles_minimizer_function_ = cv::Ptr<cv::DownhillSolver::Function>(eye_angles_optimizer_);
        eye_angles_solver_ = cv::DownhillSolver::create();
        eye_angles_solver_->setFunction(eye_angles_minimizer_function_);

        eye_centre_optimizer_ = new EyeCentreOptimizer();
        eye_centre_minimizer_function_ = cv::Ptr<cv::DownhillSolver::Function>(eye_centre_optimizer_);
        eye_centre_solver_ = cv::DownhillSolver::create();
        eye_centre_solver_->setFunction(eye_centre_minimizer_function_);

        cornea_optimizer_ = new CorneaOptimizer();
        cornea_minimizer_function_ = cv::Ptr<cv::DownhillSolver::Function>(cornea_optimizer_);
        cornea_solver_ = cv::DownhillSolver::create();
        cornea_solver_->setFunction(cornea_minimizer_function_);
        cornea_solver_->setTermCriteria(cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 1000, std::numeric_limits<float>::min()));
    }

    void MetaModel::findMetaModel(const CalibrationData& calibration_data)
    {
        et::Settings::parameters.user_params[camera_id_]->poly_eye_centre_offset = {0.0, 0.0, 0.0};
        et::Settings::parameters.user_params[camera_id_]->model_eye_centre_offset = {0.0, 0.0, 0.0};
        et::Settings::parameters.user_params[camera_id_]->poly_angles_offset = {0.0, 0.0};
        et::Settings::parameters.user_params[camera_id_]->model_angles_offset = {0.0, 0.0};

        std::vector<std::vector<double>> data{};

        bool one_sample_per_marker = true;

        auto polynomial_estimator = std::make_shared<PolynomialEyeEstimator>(camera_id_);
        auto model_estimator = std::make_shared<ModelEyeEstimator>(camera_id_);
        std::vector<cv::Point3d> poly_estimated_eye_positions{};
        std::vector<cv::Point3d> model_estimated_eye_positions{};
        std::vector<cv::Vec2d> poly_estimated_angles{};
        std::vector<cv::Vec2d> model_estimated_angles{};
        std::vector<cv::Vec2d> real_angles{};

        auto eye_measurements = Settings::parameters.polynomial_params[camera_id_].eye_measurements;

        double current_timer = 0;
        EyeInfo eye_info{};
        std::vector<EyeInfo> eye_infos{};
        current_timer = calibration_data.timestamps[0];
        for (int i = 0; i <= calibration_data.marker_positions.size(); i++)
        {
            if (i < calibration_data.marker_positions.size() && calibration_data.timestamps[i] - current_timer < 1.0)
            {
                continue;
            }

            if (i != 0 && !eye_infos.empty() && (!one_sample_per_marker || i == calibration_data.marker_positions.size() || calibration_data.marker_positions[i - 1] != calibration_data.marker_positions[i]))
            {
                EyeInfo mean_eye_info{};
                mean_eye_info.pupil = std::accumulate(eye_infos.begin(), eye_infos.end(), cv::Point2d(0, 0),
                                                      [](cv::Point2d a, EyeInfo b) { return a + b.pupil; }) / static_cast<int>(eye_infos.size());
                mean_eye_info.ellipse.center = std::accumulate(eye_infos.begin(), eye_infos.end(), cv::Point2f(0, 0),
                                                               [](cv::Point2f a, EyeInfo b) { return a + b.ellipse.center; }) / static_cast<int>(eye_infos.size());
                mean_eye_info.ellipse.size = cv::Point2f(std::accumulate(eye_infos.begin(), eye_infos.end(), cv::Size2f(0, 0),
                                                                         [](cv::Size2f a, EyeInfo b) { return a + b.ellipse.size; })) / static_cast<int>(eye_infos.size());
                cv::Point2d mean_top_left_glint = std::accumulate(eye_infos.begin(), eye_infos.end(), cv::Point2d(0, 0),
                                                                  [](cv::Point2d a, EyeInfo b) { return a + b.glints[0]; }) / static_cast<int>(eye_infos.size());
                cv::Point2d mean_bottom_right_glint = std::accumulate(eye_infos.begin(), eye_infos.end(), cv::Point2d(0, 0),
                                                                      [](cv::Point2d a, EyeInfo b) { return a + b.glints[1]; }) / static_cast<int>(eye_infos.size());
                mean_eye_info.glints = {mean_top_left_glint, mean_bottom_right_glint};
                cv::Point3d estimated_eye_position{};
                cv::Vec2d estimated_angle{};
                polynomial_estimator->detectEye(mean_eye_info, estimated_eye_position, estimated_angle);
                poly_estimated_eye_positions.push_back(estimated_eye_position);
                poly_estimated_angles.push_back(estimated_angle);

                model_estimator->detectEye(mean_eye_info, estimated_eye_position, estimated_angle);
                model_estimated_eye_positions.push_back(estimated_eye_position);
                model_estimated_angles.push_back(estimated_angle);

                cv::Mat x = (cv::Mat_<double>(1, 2) << 0.0, 0.0);
                cv::Mat step = cv::Mat::ones(x.rows, x.cols, CV_64F) * 0.1;
                cv::Point3d nodal_point{};
                cornea_solver_->setInitStep(step);
                cornea_solver_->minimize(x);
                nodal_point.x = calibration_data.eye_position.x -
                                eye_measurements.cornea_curvature_radius * sin(x.at<double>(0, 0)) * cos(x.at<double>(0, 1));
                nodal_point.y = calibration_data.eye_position.y + eye_measurements.cornea_curvature_radius * sin(x.at<double>(0, 1));
                nodal_point.z = calibration_data.eye_position.z -
                                eye_measurements.cornea_curvature_radius * cos(x.at<double>(0, 0)) * cos(x.at<double>(0, 1));

                cv::Vec3d visual_axis = calibration_data.marker_positions[i] - nodal_point;
                cv::normalize(visual_axis, visual_axis);
                cv::Vec2d angle{};
                Utils::vectorToAngles(visual_axis, angle);

                real_angles.push_back(angle);

                if (i == calibration_data.marker_positions.size())
                {
                    break;
                }
                eye_infos.clear();

                data.push_back({calibration_data.eye_position.x, calibration_data.eye_position.y, calibration_data.eye_position.z,
                                calibration_data.marker_positions[i].x, calibration_data.marker_positions[i].y, calibration_data.marker_positions[i].z,
                                poly_estimated_eye_positions.back().x, poly_estimated_eye_positions.back().y, poly_estimated_eye_positions.back().z,
                                model_estimated_eye_positions.back().x, model_estimated_eye_positions.back().y, model_estimated_eye_positions.back().z,
                                poly_estimated_angles.back()[0], poly_estimated_angles.back()[1], model_estimated_angles.back()[0], model_estimated_angles.back()[1],
                                real_angles.back()[0], real_angles.back()[1]});
            }

            if (i != 0 && calibration_data.marker_positions[i - 1] != calibration_data.marker_positions[i]) {
                current_timer = calibration_data.timestamps[i];
                continue;
            }

            eye_info.pupil = calibration_data.pupil_positions[i];
            eye_info.ellipse = calibration_data.glint_ellipses[i];
            eye_info.glints = {calibration_data.top_left_glints[i], calibration_data.bottom_right_glints[i]};
            eye_infos.push_back(eye_info);
        }
        Utils::writeFloatCsv(data, "data.csv");

        std::cout << "Total samples: " << poly_estimated_eye_positions.size() << "\n";

        int data_points_num = poly_estimated_eye_positions.size();
        int cross_folds = 5;
        std::vector<int> indices(data_points_num);
        Utils::getCrossValidationIndices(indices, data_points_num, 5);
        std::vector<double> poly_errors_x{};
        std::vector<double> model_errors_x{};
        std::vector<double> poly_errors_y{};
        std::vector<double> model_errors_y{};
        std::vector<double> poly_errors_z{};
        std::vector<double> model_errors_z{};
        std::vector<double> poly_errors_theta{};
        std::vector<double> model_errors_theta{};
        std::vector<double> poly_errors_phi{};
        std::vector<double> model_errors_phi{};
        cv::Point3d poly_eye_centre_offset{};
        cv::Point3d model_eye_centre_offset{};
        cv::Vec2d poly_angle_offset{};
        cv::Vec2d model_angle_offset{};
        for (int i = 0; i <= cross_folds; i++)
        {
            poly_eye_centre_offset = {0, 0, 0};
            model_eye_centre_offset = {0, 0, 0};
            poly_angle_offset = {0, 0};
            model_angle_offset = {0, 0};
            int counter = 0;
            for (int j = 0; j < poly_estimated_eye_positions.size(); j++)
            {
                if (indices[j] == i)
                {
                    continue;
                }
                poly_eye_centre_offset += poly_estimated_eye_positions[j] - calibration_data.eye_position;
                model_eye_centre_offset += model_estimated_eye_positions[j] - calibration_data.eye_position;
                poly_angle_offset += poly_estimated_angles[j] - real_angles[j];
                model_angle_offset += model_estimated_angles[j] - real_angles[j];
                counter++;
            }
            poly_eye_centre_offset = poly_eye_centre_offset / counter;
            model_eye_centre_offset = model_eye_centre_offset / counter;
            poly_angle_offset = poly_angle_offset / counter;
            model_angle_offset = model_angle_offset / counter;

            for (int j = 0; j < poly_estimated_eye_positions.size(); j++)
            {
                if (indices[j] != i && i != cross_folds)
                {
                    continue;
                }
                poly_errors_x.push_back(
                    cv::norm(poly_estimated_eye_positions[j].x - calibration_data.eye_position.x - poly_eye_centre_offset.x));
                model_errors_x.push_back(
                        cv::norm(model_estimated_eye_positions[j].x - calibration_data.eye_position.x - model_eye_centre_offset.x));

                poly_errors_y.push_back(
                    cv::norm(poly_estimated_eye_positions[j].y - calibration_data.eye_position.y - poly_eye_centre_offset.y));
                model_errors_y.push_back(
                        cv::norm(model_estimated_eye_positions[j].y - calibration_data.eye_position.y - model_eye_centre_offset.y));

                poly_errors_z.push_back(
                    cv::norm(poly_estimated_eye_positions[j].z - calibration_data.eye_position.z - poly_eye_centre_offset.z));
                model_errors_z.push_back(
                        cv::norm(model_estimated_eye_positions[j].z - calibration_data.eye_position.z - model_eye_centre_offset.z));

                poly_errors_theta.push_back(std::abs(poly_estimated_angles[j][0] - real_angles[j][0] - poly_angle_offset[0]));
                model_errors_theta.push_back(std::abs(model_estimated_angles[j][0] - real_angles[j][0] - model_angle_offset[0]));

                poly_errors_phi.push_back(std::abs(poly_estimated_angles[j][1] - real_angles[j][1] - poly_angle_offset[1]));
                model_errors_phi.push_back(std::abs(model_estimated_angles[j][1] - real_angles[j][1] - model_angle_offset[1]));
            }

            if (i >= cross_folds - 1) {
                if (i == cross_folds - 1) {
                    std::cout << "Cross validation results:\n";
                } else {
                    std::cout << "Full data results:\n";
                }

                std::cout << "Polynomial mean error X: "
                          << std::accumulate(poly_errors_x.begin(), poly_errors_x.end(), 0.0) / poly_errors_x.size() <<
                          " ± " << Utils::getStdDev(poly_errors_x) << "\n";
                std::cout << "Model mean error X: "
                          << std::accumulate(model_errors_x.begin(), model_errors_x.end(), 0.0) / model_errors_x.size() <<
                          " ± " << Utils::getStdDev(model_errors_x) << "\n";

                std::cout << "Polynomial mean error Y: "
                          << std::accumulate(poly_errors_y.begin(), poly_errors_y.end(), 0.0) / poly_errors_y.size() <<
                          " ± " << Utils::getStdDev(poly_errors_y) << "\n";
                std::cout << "Model mean error Y: "
                          << std::accumulate(model_errors_y.begin(), model_errors_y.end(), 0.0) / model_errors_y.size() <<
                          " ± " << Utils::getStdDev(model_errors_y) << "\n";

                std::cout << "Polynomial mean error Z: "
                          << std::accumulate(poly_errors_z.begin(), poly_errors_z.end(), 0.0) / poly_errors_z.size() <<
                          " ± " << Utils::getStdDev(poly_errors_z) << "\n";
                std::cout << "Model mean error Z: "
                          << std::accumulate(model_errors_z.begin(), model_errors_z.end(), 0.0) / model_errors_z.size() <<
                          " ± " << Utils::getStdDev(model_errors_z) << "\n";

                std::cout << "Polynomial error theta: "
                          << std::accumulate(poly_errors_theta.begin(), poly_errors_theta.end(), 0.0) / poly_errors_theta.
                                  size() << " ± " << Utils::getStdDev(poly_errors_theta) << "\n";
                std::cout << "Model error theta: "
                          << std::accumulate(model_errors_theta.begin(), model_errors_theta.end(), 0.0) / model_errors_theta.
                                  size() << " ± " << Utils::getStdDev(model_errors_theta) << "\n";

                std::cout << "Polynomial error phi: "
                          << std::accumulate(poly_errors_phi.begin(), poly_errors_phi.end(), 0.0) / poly_errors_phi.size() << " ± "
                          << Utils::getStdDev(poly_errors_phi) << std::endl;
                std::cout << "Model error phi: "
                          << std::accumulate(model_errors_phi.begin(), model_errors_phi.end(), 0.0) / model_errors_phi.size() << " ± "
                          << Utils::getStdDev(model_errors_phi) << std::endl;

                poly_errors_x.clear();
                model_errors_x.clear();

                poly_errors_y.clear();
                model_errors_y.clear();

                poly_errors_z.clear();
                model_errors_z.clear();

                poly_errors_theta.clear();
                model_errors_theta.clear();

                poly_errors_phi.clear();
                model_errors_phi.clear();
            }
        }

        Framework::mutex.lock();
        et::Settings::parameters.user_params[camera_id_]->poly_eye_centre_offset = poly_eye_centre_offset;
        et::Settings::parameters.user_params[camera_id_]->model_eye_centre_offset = model_eye_centre_offset;
        et::Settings::parameters.user_params[camera_id_]->poly_angles_offset = poly_angle_offset;
        et::Settings::parameters.user_params[camera_id_]->model_angles_offset = model_angle_offset;
        et::Settings::saveSettings();
        Framework::mutex.unlock();
    }
} // et
