#include "eye_tracker/optimizers/MetaModel.hpp"
#include "eye_tracker/Utils.hpp"
#include "eye_tracker/eye/EyeEstimator.hpp"

#include <fstream>
#include <random>

namespace et {
    MetaModel::MetaModel(int camera_id) : camera_id_(camera_id) {
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

        eye_estimator_ = std::make_shared<ModelEyeEstimator>(camera_id);
    }

    void MetaModel::findMetaModel(const std::vector<CalibrationSample>& calibration_data, bool from_scratch) {
        auto eye_measurements = Settings::parameters.polynomial_params[camera_id_].eye_measurements;

        std::vector<int> indices{};
        bool ransac = true;

        std::vector<cv::Point3d> position_errors{};
        std::vector<cv::Point2d> angle_errors_model_offset{};
        std::vector<cv::Point2d> angle_errors_model_poly_fit{};
        std::vector<cv::Point2d> angle_errors_vog{};

        std::vector<std::vector<double>> full_data{};

        std::vector<cv::Point3d> positions_offsets{};
        std::vector<cv::Point2d> angles_offsets{};

        std::vector<cv::Point3d> real_positions{};
        std::vector<cv::Point2d> real_angles{};

        std::vector<cv::Point3d> predicted_positions{};
        std::vector<cv::Point2d> predicted_angles{};

        std::shared_ptr<PolynomialFit> polynomial_fit_x = std::make_shared<PolynomialFit>(2, 2);
        std::shared_ptr<PolynomialFit> polynomial_fit_y = std::make_shared<PolynomialFit>(2, 2);

        std::shared_ptr<PolynomialFit> polynomial_fit_theta = std::make_shared<PolynomialFit>(2, 2);
        std::shared_ptr<PolynomialFit> polynomial_fit_phi = std::make_shared<PolynomialFit>(2, 2);

        std::vector<std::vector<double>> input_x_y{};
        std::vector<std::vector<double>> input_theta_phi{};

        std::vector<double> output_x{};
        std::vector<double> output_y{};

        std::vector<double> output_theta{};
        std::vector<double> output_phi{};

        cv::Point3d position_offset{};
        cv::Point2d angle_offset{};

        for (auto const& sample: calibration_data) {
            if (!sample.detected || sample.marker_time < 1.0) {
                continue;
            }

            EyeInfo eye_info;
            cv::Point3d predicted_eye_position{};
            cv::Vec2d predicted_angle{};
            eye_info.glints = {sample.top_left_glint, sample.bottom_right_glint};
            eye_info.pupil = sample.pupil_position;
            eye_info.ellipse = sample.glint_ellipse;
            eye_estimator_->detectEye(eye_info, predicted_eye_position, predicted_angle);

            cv::Mat x = (cv::Mat_<double>(1, 2) << 0.0, 0.0);
            cv::Mat step = cv::Mat::ones(x.rows, x.cols, CV_64F) * 0.1;
            cv::Point3d nodal_point{};
            cornea_optimizer_->setParameters(eye_measurements, sample.eye_position, sample.marker_position);
            cornea_solver_->setInitStep(step);
            cornea_solver_->minimize(x);
            nodal_point.x = sample.eye_position.x - eye_measurements.cornea_curvature_radius * sin(x.at<double>(0, 0)) * cos(x.at<double>(0, 1));
            nodal_point.y = sample.eye_position.y + eye_measurements.cornea_curvature_radius * sin(x.at<double>(0, 1));
            nodal_point.z = sample.eye_position.z - eye_measurements.cornea_curvature_radius * cos(x.at<double>(0, 0)) * cos(x.at<double>(0, 1));

            cv::Vec3d visual_axis = sample.marker_position - nodal_point;
            cv::normalize(visual_axis, visual_axis);
            cv::Vec2d real_angle{};
            Utils::vectorToAngles(visual_axis, real_angle);

            positions_offsets.push_back(sample.eye_position - predicted_eye_position);
            angles_offsets.emplace_back(real_angle - predicted_angle);

            input_x_y.push_back({sample.pupil_position.x - sample.glint_ellipse.center.x, sample.pupil_position.y - sample.glint_ellipse.center.y});
            input_theta_phi.push_back({predicted_angle[0], predicted_angle[1]});

            output_x.push_back(sample.marker_position.x);
            output_y.push_back(sample.marker_position.y);

            output_theta.push_back(real_angle[0]);
            output_phi.push_back(real_angle[1]);

            predicted_positions.push_back(predicted_eye_position);
            real_positions.push_back(sample.eye_position);

            predicted_angles.emplace_back(predicted_angle);
            real_angles.emplace_back(real_angle);
        }

        std::vector<bool> best_x_y_samples{};
        std::vector<bool> best_theta_phi_samples{};

        position_offset = Utils::getMean<cv::Point3d>(positions_offsets);
        angle_offset = Utils::getMean<cv::Point2d>(angles_offsets);

        std::cout << "Position offset: " << position_offset << std::endl;
        std::cout << "Angle offset: " << angle_offset << std::endl;

        if (from_scratch) {
            int min_fitting_size = polynomial_fit_x->getCoefficients().size();
            int trials_num = ransac ? 100'000 : 0;
            std::vector<std::vector<int>> trials{};
            for (int i = 0; i < input_x_y.size(); i++) {
                indices.push_back(i);
            }
            for (int i = 0; i < trials_num; i++) {
                std::shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device()()));
                trials.emplace_back(indices.begin(), indices.begin() + min_fitting_size);
            }

            double threshold_theta_phi = 1.5;
            double threshold_x_y = 20.0;

            int best_x_y = 0;
            int best_theta_phi = 0;

            std::shared_ptr<PolynomialFit> best_polynomial_fit_x = std::make_shared<PolynomialFit>(2, 2);
            std::shared_ptr<PolynomialFit> best_polynomial_fit_y = std::make_shared<PolynomialFit>(2, 2);

            std::shared_ptr<PolynomialFit> best_polynomial_fit_theta = std::make_shared<PolynomialFit>(2, 2);
            std::shared_ptr<PolynomialFit> best_polynomial_fit_phi = std::make_shared<PolynomialFit>(2, 2);

            for (int i = 0; i < trials_num; i++) {
                int current_x_y = 0;
                int current_theta_phi = 0;

                std::vector<std::vector<double>> input_x_y_sample{};
                std::vector<std::vector<double>> input_theta_phi_sample{};
                std::vector<double> output_x_sample{};
                std::vector<double> output_y_sample{};
                std::vector<double> output_theta_sample{};
                std::vector<double> output_phi_sample{};
                std::vector<bool> x_y_samples{};
                std::vector<bool> theta_phi_samples{};
                for (int j = 0; j < min_fitting_size; j++) {
                    input_x_y_sample.push_back(input_x_y[trials[i][j]]);
                    input_theta_phi_sample.push_back(input_theta_phi[trials[i][j]]);
                    output_x_sample.push_back(output_x[trials[i][j]]);
                    output_y_sample.push_back(output_y[trials[i][j]]);
                    output_theta_sample.push_back(output_theta[trials[i][j]]);
                    output_phi_sample.push_back(output_phi[trials[i][j]]);
                }
                polynomial_fit_x->fit(input_x_y_sample, &output_x_sample);
                polynomial_fit_y->fit(input_x_y_sample, &output_y_sample);
                polynomial_fit_theta->fit(input_theta_phi_sample, &output_theta_sample);
                polynomial_fit_phi->fit(input_theta_phi_sample, &output_phi_sample);

                input_x_y_sample.clear();
                input_theta_phi_sample.clear();
                output_x_sample.clear();
                output_y_sample.clear();
                output_theta_sample.clear();
                output_phi_sample.clear();
                for (int j = 0; j < input_x_y.size(); j++) {
                    auto predicted_x = polynomial_fit_x->getEstimation(input_x_y[j]);
                    auto predicted_y = polynomial_fit_y->getEstimation(input_x_y[j]);
                    auto predicted_theta = polynomial_fit_theta->getEstimation(input_theta_phi[j]);
                    auto predicted_phi = polynomial_fit_phi->getEstimation(input_theta_phi[j]);
                    auto real_x = output_x[j];
                    auto real_y = output_y[j];
                    auto real_theta = output_theta[j];
                    auto real_phi = output_phi[j];

                    if (std::abs(predicted_x - real_x) < threshold_x_y && std::abs(predicted_y - real_y) < threshold_x_y) {
                        current_x_y++;
                        input_x_y_sample.push_back(input_x_y[j]);
                        output_x_sample.push_back(output_x[j]);
                        output_y_sample.push_back(output_y[j]);
                        x_y_samples.push_back(true);
                    } else {
                        x_y_samples.push_back(false);
                    }
                    if (std::abs(predicted_theta - real_theta) < threshold_theta_phi && std::abs(predicted_phi - real_phi) < threshold_theta_phi) {
                        current_theta_phi++;
                        input_theta_phi_sample.push_back(input_theta_phi[j]);
                        output_theta_sample.push_back(output_theta[j]);
                        output_phi_sample.push_back(output_phi[j]);
                        theta_phi_samples.push_back(true);
                    } else {
                        theta_phi_samples.push_back(false);
                    }
                }
                if (current_x_y > best_x_y) {
                    best_x_y = current_x_y;
                    best_polynomial_fit_x->fit(input_x_y_sample, &output_x_sample);
                    best_polynomial_fit_y->fit(input_x_y_sample, &output_y_sample);
                    best_x_y_samples = x_y_samples;
                }
                if (current_theta_phi > best_theta_phi) {
                    best_theta_phi = current_theta_phi;
                    best_polynomial_fit_theta->fit(input_theta_phi_sample, &output_theta_sample);
                    best_polynomial_fit_phi->fit(input_theta_phi_sample, &output_phi_sample);
                    best_theta_phi_samples = theta_phi_samples;
                }
            }

            if (ransac) {
                polynomial_fit_x->setCoefficients(best_polynomial_fit_x->getCoefficients());
                polynomial_fit_y->setCoefficients(best_polynomial_fit_y->getCoefficients());
                polynomial_fit_theta->setCoefficients(best_polynomial_fit_theta->getCoefficients());
                polynomial_fit_phi->setCoefficients(best_polynomial_fit_phi->getCoefficients());
            } else {
                polynomial_fit_x->fit(input_x_y, &output_x);
                polynomial_fit_y->fit(input_x_y, &output_y);
                polynomial_fit_theta->fit(input_theta_phi, &output_theta);
                polynomial_fit_phi->fit(input_theta_phi, &output_phi);
                best_x_y_samples = std::vector<bool>(input_x_y.size(), true);
                best_theta_phi_samples = std::vector<bool>(input_theta_phi.size(), true);
            }
            et::Settings::parameters.user_params[camera_id_]->position_offset = position_offset;
            et::Settings::parameters.user_params[camera_id_]->angle_offset = angle_offset;
            et::Settings::parameters.user_params[camera_id_]->polynomial_x = polynomial_fit_x->getCoefficients();
            et::Settings::parameters.user_params[camera_id_]->polynomial_y = polynomial_fit_y->getCoefficients();
            et::Settings::parameters.user_params[camera_id_]->polynomial_theta = polynomial_fit_theta->getCoefficients();
            et::Settings::parameters.user_params[camera_id_]->polynomial_phi = polynomial_fit_phi->getCoefficients();
            et::Settings::saveSettings();
        } else {
            position_offset = et::Settings::parameters.user_params[camera_id_]->position_offset;
            angle_offset = et::Settings::parameters.user_params[camera_id_]->angle_offset;
            polynomial_fit_x->setCoefficients(et::Settings::parameters.user_params[camera_id_]->polynomial_x);
            polynomial_fit_y->setCoefficients(et::Settings::parameters.user_params[camera_id_]->polynomial_y);
            polynomial_fit_theta->setCoefficients(et::Settings::parameters.user_params[camera_id_]->polynomial_theta);
            polynomial_fit_phi->setCoefficients(et::Settings::parameters.user_params[camera_id_]->polynomial_phi);
        }

        for (int j = 0; j < predicted_positions.size(); j++) {
//            if (!best_x_y_samples[j]) {
//                continue;
//            }

//            if (!best_theta_phi_samples[j]) {
//                continue;
//            }

            std::vector<double> data_point{};
            cv::Point3d marker_position = {polynomial_fit_x->getEstimation(input_x_y[j]), polynomial_fit_y->getEstimation(input_x_y[j]), 180.0};

            cv::Mat x = (cv::Mat_<double>(1, 2) << 0.0, 0.0);
            cv::Mat step = cv::Mat::ones(x.rows, x.cols, CV_64F) * 0.1;
            cv::Point3d nodal_point{};
            cornea_optimizer_->setParameters(eye_measurements, predicted_positions[j], marker_position);
            cornea_solver_->setInitStep(step);
            cornea_solver_->minimize(x);
            nodal_point.x = predicted_positions[j].x - eye_measurements.cornea_curvature_radius * sin(x.at<double>(0, 0)) * cos(x.at<double>(0, 1));
            nodal_point.y = predicted_positions[j].y + eye_measurements.cornea_curvature_radius * sin(x.at<double>(0, 1));
            nodal_point.z = predicted_positions[j].z - eye_measurements.cornea_curvature_radius * cos(x.at<double>(0, 0)) * cos(x.at<double>(0, 1));
            cv::Vec3d visual_axis = marker_position - nodal_point;
            cv::normalize(visual_axis, visual_axis);
            cv::Vec2d predicted_angle{};
            Utils::vectorToAngles(visual_axis, predicted_angle);

            data_point.push_back(real_positions[j].x);
            data_point.push_back(real_positions[j].y);
            data_point.push_back(real_positions[j].z);
            data_point.push_back(predicted_positions[j].x + position_offset.x);
            data_point.push_back(predicted_positions[j].y + position_offset.y);
            data_point.push_back(predicted_positions[j].z + position_offset.z);
            data_point.push_back(real_angles[j].x);
            data_point.push_back(real_angles[j].y);

            position_errors.push_back(predicted_positions[j] - real_positions[j] + position_offset);
            position_errors.back().x = std::abs(position_errors.back().x);
            position_errors.back().y = std::abs(position_errors.back().y);
            position_errors.back().z = std::abs(position_errors.back().z);

            angle_errors_vog.emplace_back((cv::Point2d) predicted_angle - real_angles[j]);
            angle_errors_vog.back().x = std::abs(angle_errors_vog.back().x);
            angle_errors_vog.back().y = std::abs(angle_errors_vog.back().y);
            data_point.push_back(predicted_angle[0]);
            data_point.push_back(predicted_angle[1]);

            predicted_angle = {polynomial_fit_theta->getEstimation(input_theta_phi[j]), polynomial_fit_phi->getEstimation(input_theta_phi[j])};
            angle_errors_model_poly_fit.emplace_back((cv::Point2d) predicted_angle - real_angles[j]);
            angle_errors_model_poly_fit.back().x = std::abs(angle_errors_model_poly_fit.back().x);
            angle_errors_model_poly_fit.back().y = std::abs(angle_errors_model_poly_fit.back().y);
            data_point.push_back(predicted_angle[0]);
            data_point.push_back(predicted_angle[1]);

            predicted_angle = predicted_angles[j] + angle_offset;
            angle_errors_model_offset.emplace_back((cv::Point2d) predicted_angle - real_angles[j]);
            angle_errors_model_offset.back().x = std::abs(angle_errors_model_offset.back().x);
            angle_errors_model_offset.back().y = std::abs(angle_errors_model_offset.back().y);
            data_point.push_back(predicted_angle[0]);
            data_point.push_back(predicted_angle[1]);

            full_data.push_back(data_point);
        }

        Utils::writeFloatCsv(full_data, "meta_model_data_full.csv");

        cv::Point3d mean_position_error = Utils::getMean<cv::Point3d>(position_errors);
        cv::Point3d std_position_error = Utils::getStdDev(position_errors);

        std::cout << "x: " << mean_position_error.x << " ± " << std_position_error.x << std::endl;
        std::cout << "y: " << mean_position_error.y << " ± " << std_position_error.y << std::endl;
        std::cout << "z: " << mean_position_error.z << " ± " << std_position_error.z << std::endl;

        cv::Point2d mean_angle_error = Utils::getMean<cv::Point2d>(angle_errors_model_offset);
        cv::Point2d std_angle_error = Utils::getStdDev(angle_errors_model_offset);

        std::cout << "Offset error:" << std::endl;
        std::cout << "theta: " << mean_angle_error.x << " ± " << std_angle_error.x << std::endl;
        std::cout << "phi: " << mean_angle_error.y << " ± " << std_angle_error.y << std::endl;

        mean_angle_error = Utils::getMean<cv::Point2d>(angle_errors_model_poly_fit);
        std_angle_error = Utils::getStdDev(angle_errors_model_poly_fit);

        std::cout << "Polynomial fit error:" << std::endl;
        std::cout << "theta: " << mean_angle_error.x << " ± " << std_angle_error.x << std::endl;
        std::cout << "phi: " << mean_angle_error.y << " ± " << std_angle_error.y << std::endl;

        mean_angle_error = Utils::getMean<cv::Point2d>(angle_errors_vog);
        std_angle_error = Utils::getStdDev(angle_errors_vog);

        std::cout << "Glint-pupil error:" << std::endl;
        std::cout << "theta: " << mean_angle_error.x << " ± " << std_angle_error.x << std::endl;
        std::cout << "phi: " << mean_angle_error.y << " ± " << std_angle_error.y << std::endl;
    }
} // et
