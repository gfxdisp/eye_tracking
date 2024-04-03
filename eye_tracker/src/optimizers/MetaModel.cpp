#include "eye_tracker/optimizers/MetaModel.hpp"
#include "eye_tracker/Utils.hpp"
#include "eye_tracker/eye/EyeEstimator.hpp"

#include <fstream>

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
        int marker_count = calibration_data.back().marker_id + 1;
        Utils::getCrossValidationIndices(indices, marker_count, 5);

        std::vector<cv::Point3d> position_errors{};
        std::vector<cv::Point2d> angle_errors_model_offset{};
        std::vector<cv::Point2d> angle_errors_model_poly_fit{};
        std::vector<cv::Point2d> angle_errors_vog{};

        std::vector<std::vector<double>> full_data{};

        for (int i = 0; i < 1; i++) {
            std::vector<bool> train_markers(marker_count);
            std::vector<bool> test_markers(marker_count);
            for (int j = 0; j < marker_count; j++) {
//                train_markers[j] = (indices[j] != i);
                train_markers[j] = true;
//                test_markers[j] = (indices[j] == i);
                test_markers[j] = true;
            }

            std::vector<cv::Point3d> positions_offsets{};
            std::vector<cv::Point2d> angles_offsets{};

            std::vector<cv::Point3d> real_positions{};
            std::vector<cv::Point2d> real_angles{};

            std::vector<cv::Point3d> predicted_positions{};
            std::vector<cv::Point2d> predicted_angles{};

            std::shared_ptr<PolynomialFit> polynomial_fit_x = std::make_shared<PolynomialFit>(2, 1);
            std::shared_ptr<PolynomialFit> polynomial_fit_y = std::make_shared<PolynomialFit>(2, 1);

            std::shared_ptr<PolynomialFit> polynomial_fit_theta = std::make_shared<PolynomialFit>(2, 1);
            std::shared_ptr<PolynomialFit> polynomial_fit_phi = std::make_shared<PolynomialFit>(2, 1);

            std::vector<std::vector<double>> train_input_x_y{};
            std::vector<std::vector<double>> test_input_x_y{};

            std::vector<std::vector<double>> train_input_theta_phi{};
            std::vector<std::vector<double>> test_input_theta_phi{};

            std::vector<double> output_x{};
            std::vector<double> output_y{};

            std::vector<double> output_theta{};
            std::vector<double> output_phi{};

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

                if (train_markers[sample.marker_id]) {
                    positions_offsets.push_back(sample.eye_position - predicted_eye_position);
                    angles_offsets.emplace_back(real_angle - predicted_angle);

                    train_input_x_y.push_back({sample.pupil_position.x - sample.glint_ellipse.center.x, sample.pupil_position.y - sample.glint_ellipse.center.y});
                    output_x.push_back(sample.marker_position.x);
                    output_y.push_back(sample.marker_position.y);

                    train_input_theta_phi.push_back({predicted_angle[0], predicted_angle[1]});
                    output_theta.push_back(real_angle[0]);
                    output_phi.push_back(real_angle[1]);
                }
                if (test_markers[sample.marker_id]) {
                    predicted_positions.push_back(predicted_eye_position);
                    real_positions.push_back(sample.eye_position);

                    test_input_x_y.push_back({sample.pupil_position.x - sample.glint_ellipse.center.x, sample.pupil_position.y - sample.glint_ellipse.center.y});
                    test_input_theta_phi.push_back({predicted_angle[0], predicted_angle[1]});

                    predicted_angles.push_back(predicted_angle);
                    real_angles.emplace_back(real_angle);
                }
            }

            cv::Point3d position_offset = Utils::getMean<cv::Point3d>(positions_offsets);
            polynomial_fit_x->fit(train_input_x_y, &output_x);
            polynomial_fit_y->fit(train_input_x_y, &output_y);

            cv::Point2d angle_offset = Utils::getMean<cv::Point2d>(angles_offsets);
            polynomial_fit_theta->fit(train_input_theta_phi, &output_theta);
            polynomial_fit_phi->fit(train_input_theta_phi, &output_phi);

            std::cout << "Position offset: " << position_offset << std::endl;
            std::cout << "Angle offset: " << angle_offset << std::endl;

            if (from_scratch) {
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
                std::vector<double> data_point{};
                cv::Point3d marker_position = {polynomial_fit_x->getEstimation(test_input_x_y[j]), polynomial_fit_y->getEstimation(test_input_x_y[j]), 180.0};

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

                predicted_angle = {polynomial_fit_theta->getEstimation(test_input_theta_phi[j]), polynomial_fit_phi->getEstimation(test_input_theta_phi[j])};
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
        }

        Utils::writeFloatCsv(full_data, "meta_model_data_old_intrinsic.csv");

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


/*        Framework::mutex.lock();
        et::Settings::parameters.user_params[camera_id_]->poly_eye_centre_offset = poly_eye_centre_offset;
        et::Settings::parameters.user_params[camera_id_]->model_eye_centre_offset = model_eye_centre_offset;
        et::Settings::parameters.user_params[camera_id_]->poly_angles_offset = poly_angle_offset;
        et::Settings::parameters.user_params[camera_id_]->model_angles_offset = model_angle_offset;
        et::Settings::saveSettings();
        Framework::mutex.unlock();*/
    }
} // et
