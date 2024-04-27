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
        std::vector<double> angle_errors_model_offset{};
        std::vector<double> angle_errors_model_poly_fit{};
        std::vector<double> angle_errors_pcr{};

        std::vector<std::vector<double>> full_data{};

        std::vector<cv::Point3d> positions_offsets{};
        std::vector<cv::Point2d> angles_offsets{};

        std::vector<cv::Point3d> real_eye_positions{};
        std::vector<cv::Point3d> real_nodal_points{};
        std::vector<cv::Point2d> real_angles{};
        std::vector<cv::Point3d> real_marker_positions{};

        std::vector<cv::Point3d> predicted_eye_positions{};
        std::vector<cv::Point3d> predicted_nodal_points{};
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
        std::vector<double> timestamps{};

        cv::Point3d eye_position_offset{};
        cv::Point2d angle_offset{};

        double marker_depth = calibration_data[0].marker_position.z; // Assuming that all markers are at the same depth

        for (auto const& sample: calibration_data) {
            if (!sample.detected) {
                continue;
            }

            if (sample.marker_time < 1.0) {
                continue;
            }

            EyeInfo eye_info;
            cv::Point3d predicted_eye_position{};
            cv::Point3d predicted_nodal_point{};
            cv::Vec2d predicted_angle{};
            eye_info.glints = sample.glints;
            eye_info.glints_validity = sample.glints_validity;
            eye_info.pupil = sample.pupil_position;
            eye_info.ellipse = sample.glint_ellipse;
            eye_estimator_->detectEye(eye_info, predicted_eye_position, predicted_nodal_point, predicted_angle);

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

            predicted_eye_positions.push_back(predicted_eye_position);
            real_nodal_points.push_back(nodal_point);
            predicted_nodal_points.push_back(predicted_nodal_point);
            real_eye_positions.push_back(sample.eye_position);
            timestamps.push_back(sample.timestamp);

            predicted_angles.emplace_back(predicted_angle);
            real_angles.emplace_back(real_angle);
            real_marker_positions.push_back(sample.marker_position);
        }

/*        std::vector<CalibrationInput> calibration_input;
        CalibrationOutput calibration_output;

        for (int i = 0; i < predicted_eye_positions.size(); i++) {
            CalibrationInput input{};
            input.eye_position = predicted_eye_positions[i];
            input.cornea_position = predicted_nodal_points[i];
            input.pcr_distance = {input_x_y[i][0], input_x_y[i][1]};
            input.angles = {input_theta_phi[i][0], input_theta_phi[i][1]};
            input.timestamp = timestamps[i];
            calibration_input.push_back(input);
        }
        calibration_output.eye_position = real_eye_positions[0];
        for (int i = 0; i < real_marker_positions.size(); i++) {
            if (i == 0 || calibration_output.marker_positions.back() != real_marker_positions[i]) {
                calibration_output.marker_positions.push_back(real_marker_positions[i]);
                calibration_output.timestamps.push_back((static_cast<int>(calibration_output.timestamps.size() + 1)) * 3);
            }
        }*/

        std::vector<int> position_outliers = Utils::getOutliers(positions_offsets, 2.0);
        std::vector<int> angle_outliers = Utils::getOutliers(angles_offsets, 2.0);
        std::vector<int> all_outliers{};
        all_outliers.insert(all_outliers.end(), position_outliers.begin(), position_outliers.end());
//        all_outliers.insert(all_outliers.end(), angle_outliers.begin(), angle_outliers.end());
        std::sort(all_outliers.begin(), all_outliers.end());
        all_outliers.erase(std::unique(all_outliers.begin(), all_outliers.end()), all_outliers.end());

        for (int i = 0; i < all_outliers.size(); i++) {
            int index = all_outliers[i] - i; // -i to correct for the fact that the vector is shrinking
            positions_offsets.erase(positions_offsets.begin() + index);
            angles_offsets.erase(angles_offsets.begin() + index);
            input_x_y.erase(input_x_y.begin() + index);
            input_theta_phi.erase(input_theta_phi.begin() + index);
            output_x.erase(output_x.begin() + index);
            output_y.erase(output_y.begin() + index);
            output_theta.erase(output_theta.begin() + index);
            output_phi.erase(output_phi.begin() + index);
            predicted_eye_positions.erase(predicted_eye_positions.begin() + index);
            real_nodal_points.erase(real_nodal_points.begin() + index);
            predicted_nodal_points.erase(predicted_nodal_points.begin() + index);
            real_eye_positions.erase(real_eye_positions.begin() + index);
            predicted_angles.erase(predicted_angles.begin() + index);
            real_angles.erase(real_angles.begin() + index);
            real_marker_positions.erase(real_marker_positions.begin() + index);
        }

        std::vector<bool> best_x_y_samples{};
        std::vector<bool> best_theta_phi_samples{};

        eye_position_offset = Utils::getMean<cv::Point3d>(positions_offsets);
        angle_offset = Utils::getMean<cv::Point2d>(angles_offsets);

        cv::Point2d angle_offset_deg = angle_offset * 180.0 / CV_PI;


        std::cout << "Position offset: " << eye_position_offset << std::endl;
        std::cout << "Angle offset: " << angle_offset_deg << std::endl;

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

            double threshold_theta_phi = 1.5 * CV_PI / 180.0;
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
            et::Settings::parameters.user_params[camera_id_]->position_offset = eye_position_offset;
            et::Settings::parameters.user_params[camera_id_]->angle_offset = angle_offset;
            et::Settings::parameters.user_params[camera_id_]->polynomial_x = polynomial_fit_x->getCoefficients();
            et::Settings::parameters.user_params[camera_id_]->polynomial_y = polynomial_fit_y->getCoefficients();
            et::Settings::parameters.user_params[camera_id_]->polynomial_theta = polynomial_fit_theta->getCoefficients();
            et::Settings::parameters.user_params[camera_id_]->polynomial_phi = polynomial_fit_phi->getCoefficients();
            et::Settings::saveSettings();
        } else {
            eye_position_offset = et::Settings::parameters.user_params[camera_id_]->position_offset;
            angle_offset = et::Settings::parameters.user_params[camera_id_]->angle_offset;
            polynomial_fit_x->setCoefficients(et::Settings::parameters.user_params[camera_id_]->polynomial_x);
            polynomial_fit_y->setCoefficients(et::Settings::parameters.user_params[camera_id_]->polynomial_y);
            polynomial_fit_theta->setCoefficients(et::Settings::parameters.user_params[camera_id_]->polynomial_theta);
            polynomial_fit_phi->setCoefficients(et::Settings::parameters.user_params[camera_id_]->polynomial_phi);
        }

        for (int j = 0; j < predicted_eye_positions.size(); j++) {
//            if (from_scratch && !best_x_y_samples[j]) {
//                continue;
//            }
//
//            if (from_scratch && !best_theta_phi_samples[j]) {
//                continue;
//            }

            std::vector<double> data_point{};

            data_point.push_back(real_eye_positions[j].x);
            data_point.push_back(real_eye_positions[j].y);
            data_point.push_back(real_eye_positions[j].z);
            data_point.push_back(predicted_eye_positions[j].x + eye_position_offset.x);
            data_point.push_back(predicted_eye_positions[j].y + eye_position_offset.y);
            data_point.push_back(predicted_eye_positions[j].z + eye_position_offset.z);
            data_point.push_back(real_angles[j].x);
            data_point.push_back(real_angles[j].y);

            cv::Vec3d real_visual_axis = real_marker_positions[j] - real_nodal_points[j];
            cv::normalize(real_visual_axis, real_visual_axis);

            position_errors.push_back(predicted_eye_positions[j] - real_eye_positions[j] + eye_position_offset);
            position_errors.back().x = std::abs(position_errors.back().x);
            position_errors.back().y = std::abs(position_errors.back().y);
            position_errors.back().z = std::abs(position_errors.back().z);

            cv::Point3d predicted_marker_position = {polynomial_fit_x->getEstimation(input_x_y[j]), polynomial_fit_y->getEstimation(input_x_y[j]), marker_depth};
            cv::Vec3d visual_axis = predicted_marker_position - (predicted_nodal_points[j] + eye_position_offset);
            cv::normalize(visual_axis, visual_axis);
            cv::Vec2d predicted_angle{};
            Utils::vectorToAngles(visual_axis, predicted_angle);
            cv::Vec3d vec1 = real_marker_positions[j] - real_eye_positions[j];
            cv::Vec3d vec2 = predicted_marker_position - real_eye_positions[j];
            double angle_error = std::acos(vec1.dot(vec2) / (cv::norm(vec1) * cv::norm(vec2))) * 180.0 / CV_PI;
            angle_errors_pcr.push_back(angle_error);
            data_point.push_back(predicted_angle[0]);
            data_point.push_back(predicted_angle[1]);

            predicted_angle = {polynomial_fit_theta->getEstimation(input_theta_phi[j]), polynomial_fit_phi->getEstimation(input_theta_phi[j])};
            Utils::anglesToVector(predicted_angle, visual_axis);
            double k = (marker_depth - predicted_nodal_points[j].z - eye_position_offset.z) / visual_axis[2];
            predicted_marker_position = predicted_nodal_points[j] + eye_position_offset + (cv::Point3d) (k * visual_axis);
            vec1 = real_marker_positions[j] - real_eye_positions[j];
            vec2 = predicted_marker_position - real_eye_positions[j];
            angle_error = std::acos(vec1.dot(vec2) / (cv::norm(vec1) * cv::norm(vec2))) * 180.0 / CV_PI;
            angle_errors_model_poly_fit.push_back(angle_error);
            data_point.push_back(predicted_angle[0]);
            data_point.push_back(predicted_angle[1]);

            predicted_angle = predicted_angles[j] + angle_offset;
            Utils::anglesToVector(predicted_angle, visual_axis);
            k = (marker_depth - predicted_nodal_points[j].z - eye_position_offset.z) / visual_axis[2];
            predicted_marker_position = predicted_nodal_points[j] + eye_position_offset + (cv::Point3d) (k * visual_axis);
            vec1 = real_marker_positions[j] - real_eye_positions[j];
            vec2 = predicted_marker_position - real_eye_positions[j];
            angle_error = std::acos(vec1.dot(vec2) / (cv::norm(vec1) * cv::norm(vec2))) * 180.0 / CV_PI;
            angle_errors_model_offset.push_back(angle_error);
            data_point.push_back(predicted_angle[0]);
            data_point.push_back(predicted_angle[1]);

            full_data.push_back(data_point);
        }

        Utils::writeFloatCsv(full_data, "meta_model_data_full.csv");

        std::cout << std::setprecision(3) << std::fixed;

        auto mean_position_error = Utils::getMean<cv::Point3d>(position_errors);
        cv::Point3d std_position_error = Utils::getStdDev(position_errors);

        std::cout << "x: " << mean_position_error.x << " ± " << std_position_error.x << std::endl;
        std::cout << "y: " << mean_position_error.y << " ± " << std_position_error.y << std::endl;
        std::cout << "z: " << mean_position_error.z << " ± " << std_position_error.z << std::endl;
        std::cout << "total xy: " << cv::norm(cv::Point2d(mean_position_error.x, mean_position_error.y)) << " ± " << cv::norm(cv::Point2d(std_position_error.x, std_position_error.y)) << std::endl;
        std::cout << "total: " << cv::norm(mean_position_error) << " ± " << cv::norm(std_position_error) << std::endl;

        auto mean_angle_error = Utils::getMean<double>(angle_errors_model_offset);
        auto std_angle_error = Utils::getStdDev<double>(angle_errors_model_offset);
        std::cout << "Offset error: " << mean_angle_error << " ± " << std_angle_error << std::endl;

        mean_angle_error = Utils::getMean<double>(angle_errors_model_poly_fit);
        std_angle_error = Utils::getStdDev<double>(angle_errors_model_poly_fit);
        std::cout << "Polynomial fit error: " << mean_angle_error << " ± " << std_angle_error << std::endl;

        mean_angle_error = Utils::getMean<double>(angle_errors_pcr);
        std_angle_error = Utils::getStdDev<double>(angle_errors_pcr);
        std::cout << "Glint-pupil error: " << mean_angle_error << " ± " << std_angle_error << std::endl;
    }

    void MetaModel::findOnlineMetaModel(std::vector<CalibrationInput> const& calibration_input, CalibrationOutput const& calibration_output, bool from_scratch) {
        auto eye_measurements = Settings::parameters.polynomial_params[camera_id_].eye_measurements;

        double marker_depth = calibration_output.marker_positions[0].z; // Assuming that all markers are at the same depth

        int total_markers = 0;

        MetaModelData meta_model_data{};
        meta_model_data.real_eye_position = calibration_output.eye_position;

        std::vector<int> cum_samples_per_marker{};
        cum_samples_per_marker.push_back(0);
        cum_samples_per_marker.push_back(0);

        double start_timestamp = 0;

        for (const auto & sample : calibration_input) {
            if (sample.timestamp > calibration_output.timestamps[calibration_output.timestamps.size() - 1]) {
                break;
            }

            // We started a new marker
            if (sample.timestamp >= calibration_output.timestamps[total_markers]) {
                start_timestamp = calibration_output.timestamps[total_markers];
                total_markers++;
                cum_samples_per_marker.push_back(cum_samples_per_marker[total_markers]);
            }

            // Current sample was captured less than 1 second after the marker was shown
            if (sample.timestamp - start_timestamp < 1.0) {
                continue;
            }

            cv::Mat x = (cv::Mat_<double>(1, 2) << 0.0, 0.0);
            cv::Mat step = cv::Mat::ones(x.rows, x.cols, CV_64F) * 0.1;
            cornea_optimizer_->setParameters(eye_measurements, calibration_output.eye_position, calibration_output.marker_positions[total_markers]);
            cornea_solver_->setInitStep(step);
            cornea_solver_->minimize(x);
            cv::Point3d real_cornea_position{};
            real_cornea_position.x = meta_model_data.real_eye_position.x - eye_measurements.cornea_curvature_radius * sin(x.at<double>(0, 0)) * cos(x.at<double>(0, 1));
            real_cornea_position.y = meta_model_data.real_eye_position.y + eye_measurements.cornea_curvature_radius * sin(x.at<double>(0, 1));
            real_cornea_position.z = meta_model_data.real_eye_position.z - eye_measurements.cornea_curvature_radius * cos(x.at<double>(0, 0)) * cos(x.at<double>(0, 1));

            meta_model_data.real_cornea_positions.push_back(real_cornea_position);

            cv::Vec3d real_visual_axis = calibration_output.marker_positions[total_markers] - real_cornea_position;
            cv::normalize(real_visual_axis, real_visual_axis);
            cv::Vec2d real_angles{};
            Utils::vectorToAngles(real_visual_axis, real_angles);
            meta_model_data.real_marker_positions.push_back(calibration_output.marker_positions[total_markers]);
            meta_model_data.real_angles_theta.push_back(real_angles[0]);
            meta_model_data.real_angles_phi.push_back(real_angles[1]);

            meta_model_data.estimated_eye_positions.push_back(sample.eye_position);
            meta_model_data.estimated_cornea_positions.push_back(sample.cornea_position);
            meta_model_data.estimated_angles_theta.push_back(sample.angles[0]);
            meta_model_data.estimated_angles_phi.push_back(sample.angles[1]);
            meta_model_data.pcr_distances_x.push_back(sample.pcr_distance[0]);
            meta_model_data.pcr_distances_y.push_back(sample.pcr_distance[1]);

            cum_samples_per_marker[total_markers + 1]++;
        }
        total_markers++;


        std::vector<int> outliers = Utils::getOutliers(meta_model_data.estimated_eye_positions, 2.0);
        for (int i = 0; i < outliers.size(); i++) {
            int index = outliers[i] - i; // -i to correct for the fact that the vector is shrinking
            meta_model_data.estimated_eye_positions.erase(meta_model_data.estimated_eye_positions.begin() + index);
            meta_model_data.estimated_cornea_positions.erase(meta_model_data.estimated_cornea_positions.begin() + index);
            meta_model_data.estimated_angles_theta.erase(meta_model_data.estimated_angles_theta.begin() + index);
            meta_model_data.estimated_angles_phi.erase(meta_model_data.estimated_angles_phi.begin() + index);
            meta_model_data.real_cornea_positions.erase(meta_model_data.real_cornea_positions.begin() + index);
            meta_model_data.real_marker_positions.erase(meta_model_data.real_marker_positions.begin() + index);
            meta_model_data.real_angles_theta.erase(meta_model_data.real_angles_theta.begin() + index);
            meta_model_data.real_angles_phi.erase(meta_model_data.real_angles_phi.begin() + index);
            meta_model_data.pcr_distances_x.erase(meta_model_data.pcr_distances_x.begin() + index);
            meta_model_data.pcr_distances_y.erase(meta_model_data.pcr_distances_y.begin() + index);
        }

        int total_samples = static_cast<int>(meta_model_data.estimated_eye_positions.size());

        std::vector<bool> best_x_y_samples{};
        std::vector<bool> best_theta_phi_samples{};

        auto mean_estimated_eye_position = Utils::getTrimmmedMean(meta_model_data.estimated_eye_positions, 0.0);
        auto eye_position_offset = meta_model_data.real_eye_position - mean_estimated_eye_position;

        std::cout << "Position offset: " << eye_position_offset << std::endl;

        std::shared_ptr<PolynomialFit> polynomial_fit_pcr_x = std::make_shared<PolynomialFit>(2, 2);
        std::shared_ptr<PolynomialFit> polynomial_fit_pcr_y = std::make_shared<PolynomialFit>(2, 2);

        std::shared_ptr<PolynomialFit> polynomial_fit_theta = std::make_shared<PolynomialFit>(2, 2);
        std::shared_ptr<PolynomialFit> polynomial_fit_phi = std::make_shared<PolynomialFit>(2, 2);

        std::shared_ptr<PolynomialFit> best_polynomial_fit_pcr_x = std::make_shared<PolynomialFit>(2, 2);
        std::shared_ptr<PolynomialFit> best_polynomial_fit_pcr_y = std::make_shared<PolynomialFit>(2, 2);

        std::shared_ptr<PolynomialFit> best_polynomial_fit_theta = std::make_shared<PolynomialFit>(2, 2);
        std::shared_ptr<PolynomialFit> best_polynomial_fit_phi = std::make_shared<PolynomialFit>(2, 2);

        if (from_scratch) {

            std::random_device random_device;
            std::mt19937 generator(random_device());

            int min_fitting_size = static_cast<int>(polynomial_fit_pcr_x->getCoefficients().size());
            int trials_num = 100'000;
            std::vector<std::vector<int>> trials{};
            std::vector<int> indices{};
            for (int i = 0; i < total_markers || i < min_fitting_size; i++) {
                indices.push_back(i % total_markers);
            }
            for (int i = 0; i < trials_num; i++) {
                std::shuffle(indices.begin(), indices.end(), generator);
                trials.emplace_back(indices.begin(), indices.begin() + min_fitting_size);
                for (int j = 0; j < min_fitting_size; j++) {
                    int marker_num = trials[i][j];
                    int start_index = cum_samples_per_marker[marker_num];
                    int end_index = cum_samples_per_marker[marker_num + 1] - 1;
                    auto distribution = std::uniform_int_distribution<int>(start_index, end_index);
                    int sample_index = distribution(generator);
                    trials[i][j] = sample_index;
                }
            }

            double threshold_theta_phi = 0.5 * CV_PI / 180.0;
            double threshold_x_y = 6.0;

            int best_x_y = 0;
            int best_theta_phi = 0;

            std::vector<std::vector<double>> input_x_y_sample{};
            std::vector<std::vector<double>> input_theta_phi_sample{};
            std::vector<double> output_x_sample{};
            std::vector<double> output_y_sample{};
            std::vector<double> output_theta_sample{};
            std::vector<double> output_phi_sample{};

            for (int i = 0; i < trials_num; i++) {

                input_x_y_sample.clear();
                input_theta_phi_sample.clear();
                output_x_sample.clear();
                output_y_sample.clear();
                output_theta_sample.clear();
                output_phi_sample.clear();
                for (int j = 0; j < min_fitting_size; j++) {
                    int sample_index = trials[i][j];
                    input_x_y_sample.push_back({meta_model_data.pcr_distances_x[sample_index], meta_model_data.pcr_distances_y[sample_index]});
                    input_theta_phi_sample.push_back({meta_model_data.estimated_angles_theta[sample_index], meta_model_data.estimated_angles_phi[sample_index]});
                    output_x_sample.push_back(meta_model_data.real_marker_positions[sample_index].x);
                    output_y_sample.push_back(meta_model_data.real_marker_positions[sample_index].y);
                    output_theta_sample.push_back(meta_model_data.real_angles_theta[sample_index]);
                    output_phi_sample.push_back(meta_model_data.real_angles_phi[sample_index]);
                }
                polynomial_fit_pcr_x->fit(input_x_y_sample, &output_x_sample);
                polynomial_fit_pcr_y->fit(input_x_y_sample, &output_y_sample);
                polynomial_fit_theta->fit(input_theta_phi_sample, &output_theta_sample);
                polynomial_fit_phi->fit(input_theta_phi_sample, &output_phi_sample);

                int current_x_y = 0;
                int current_theta_phi = 0;
                std::vector<bool> x_y_samples{};
                std::vector<bool> theta_phi_samples{};
                for (int j = 0; j < total_samples; j++) {
                    auto predicted_x = polynomial_fit_pcr_x->getEstimation({meta_model_data.pcr_distances_x[j], meta_model_data.pcr_distances_y[j]});
                    auto predicted_y = polynomial_fit_pcr_y->getEstimation({meta_model_data.pcr_distances_x[j], meta_model_data.pcr_distances_y[j]});
                    auto predicted_theta = polynomial_fit_theta->getEstimation({meta_model_data.estimated_angles_theta[j], meta_model_data.estimated_angles_phi[j]});
                    auto predicted_phi = polynomial_fit_phi->getEstimation({meta_model_data.estimated_angles_theta[j], meta_model_data.estimated_angles_phi[j]});
                    auto real_x = meta_model_data.real_marker_positions[j].x;
                    auto real_y = meta_model_data.real_marker_positions[j].y;
                    auto real_theta = meta_model_data.real_angles_theta[j];
                    auto real_phi = meta_model_data.real_angles_phi[j];

                    if (std::abs(predicted_x - real_x) < threshold_x_y && std::abs(predicted_y - real_y) < threshold_x_y) {
                        current_x_y++;
                        x_y_samples.push_back(true);
                    } else {
                        x_y_samples.push_back(false);
                    }
                    if (std::abs(predicted_theta - real_theta) < threshold_theta_phi && std::abs(predicted_phi - real_phi) < threshold_theta_phi) {
                        current_theta_phi++;
                        theta_phi_samples.push_back(true);
                    } else {
                        theta_phi_samples.push_back(false);
                    }
                }
                if (current_x_y > best_x_y) {
                    best_x_y = current_x_y;
                    best_x_y_samples = x_y_samples;
                }
                if (current_theta_phi > best_theta_phi) {
                    best_theta_phi = current_theta_phi;
                    best_theta_phi_samples = theta_phi_samples;
                }
            }

            input_x_y_sample.clear();
            input_theta_phi_sample.clear();
            output_x_sample.clear();
            output_y_sample.clear();
            output_theta_sample.clear();
            output_phi_sample.clear();
            for (int i = 0; i < total_samples; i++) {
                if (best_x_y_samples[i]) {
                    input_x_y_sample.push_back({meta_model_data.pcr_distances_x[i], meta_model_data.pcr_distances_y[i]});
                    output_x_sample.push_back(meta_model_data.real_marker_positions[i].x);
                    output_y_sample.push_back(meta_model_data.real_marker_positions[i].y);
                }
                if (best_theta_phi_samples[i]) {
                    input_theta_phi_sample.push_back({meta_model_data.estimated_angles_theta[i], meta_model_data.estimated_angles_phi[i]});
                    output_theta_sample.push_back(meta_model_data.real_angles_theta[i]);
                    output_phi_sample.push_back(meta_model_data.real_angles_phi[i]);
                }
            }

            polynomial_fit_pcr_x->fit(input_x_y_sample, &output_x_sample);
            polynomial_fit_pcr_y->fit(input_x_y_sample, &output_y_sample);
            polynomial_fit_theta->fit(input_theta_phi_sample, &output_theta_sample);
            polynomial_fit_phi->fit(input_theta_phi_sample, &output_phi_sample);

            et::Settings::parameters.user_params[camera_id_]->position_offset = eye_position_offset;
            et::Settings::parameters.user_params[camera_id_]->polynomial_x = polynomial_fit_pcr_x->getCoefficients();
            et::Settings::parameters.user_params[camera_id_]->polynomial_y = polynomial_fit_pcr_y->getCoefficients();
            et::Settings::parameters.user_params[camera_id_]->polynomial_theta = polynomial_fit_theta->getCoefficients();
            et::Settings::parameters.user_params[camera_id_]->polynomial_phi = polynomial_fit_phi->getCoefficients();
            et::Settings::saveSettings();
        } else {
            eye_position_offset = et::Settings::parameters.user_params[camera_id_]->position_offset;
            polynomial_fit_pcr_x->setCoefficients(et::Settings::parameters.user_params[camera_id_]->polynomial_x);
            polynomial_fit_pcr_y->setCoefficients(et::Settings::parameters.user_params[camera_id_]->polynomial_y);
            polynomial_fit_theta->setCoefficients(et::Settings::parameters.user_params[camera_id_]->polynomial_theta);
            polynomial_fit_phi->setCoefficients(et::Settings::parameters.user_params[camera_id_]->polynomial_phi);
        }

        std::vector<double> position_errors{};
        std::vector<double> angle_errors_pcr{};
        std::vector<double> angle_errors_poly_fit{};
        std::vector<std::vector<double>> full_data{};

        for (int j = 0; j < total_samples; j++) {
            std::vector<double> data_point{};

            data_point.push_back(meta_model_data.real_eye_position.x);
            data_point.push_back(meta_model_data.real_eye_position.y);
            data_point.push_back(meta_model_data.real_eye_position.z);
            data_point.push_back(meta_model_data.estimated_eye_positions[j].x + eye_position_offset.x);
            data_point.push_back(meta_model_data.estimated_eye_positions[j].y + eye_position_offset.y);
            data_point.push_back(meta_model_data.estimated_eye_positions[j].z + eye_position_offset.z);
            data_point.push_back(meta_model_data.real_angles_theta[j]);
            data_point.push_back(meta_model_data.real_angles_phi[j]);

            cv::Vec3d real_visual_axis = meta_model_data.real_marker_positions[j] - meta_model_data.real_cornea_positions[j];
            cv::normalize(real_visual_axis, real_visual_axis);

            cv::Point2d real_eye_position{meta_model_data.real_eye_position.x, meta_model_data.real_eye_position.y};
            cv::Point2d estimated_eye_position{meta_model_data.estimated_eye_positions[j].x + eye_position_offset.x, meta_model_data.estimated_eye_positions[j].y + eye_position_offset.y};
            position_errors.push_back(cv::norm(estimated_eye_position - real_eye_position));

            double estimated_pcr_x = polynomial_fit_pcr_x->getEstimation({meta_model_data.pcr_distances_x[j], meta_model_data.pcr_distances_y[j]});
            double estimated_pcr_y = polynomial_fit_pcr_y->getEstimation({meta_model_data.pcr_distances_x[j], meta_model_data.pcr_distances_y[j]});

            cv::Point3d predicted_marker_position = {estimated_pcr_x, estimated_pcr_y, marker_depth};

            cv::Vec3d visual_axis = predicted_marker_position - (meta_model_data.estimated_cornea_positions[j] + eye_position_offset);
            cv::normalize(visual_axis, visual_axis);
            cv::Vec2d predicted_angle{};
            Utils::vectorToAngles(visual_axis, predicted_angle);
            cv::Vec3d vec1 = meta_model_data.real_marker_positions[j] - meta_model_data.real_eye_position;
            cv::Vec3d vec2 = predicted_marker_position - meta_model_data.real_eye_position;
            double angle_error = std::acos(vec1.dot(vec2) / (cv::norm(vec1) * cv::norm(vec2))) * 180.0 / CV_PI;
            angle_errors_pcr.push_back(angle_error);
            data_point.push_back(predicted_angle[0]);
            data_point.push_back(predicted_angle[1]);

            double estimated_theta = polynomial_fit_theta->getEstimation({meta_model_data.estimated_angles_theta[j], meta_model_data.estimated_angles_phi[j]});
            double estimated_phi = polynomial_fit_phi->getEstimation({meta_model_data.estimated_angles_theta[j], meta_model_data.estimated_angles_phi[j]});
            predicted_angle = {estimated_theta, estimated_phi};
            Utils::anglesToVector(predicted_angle, visual_axis);
            double k = (marker_depth - meta_model_data.estimated_cornea_positions[j].z - eye_position_offset.z) / visual_axis[2];
            predicted_marker_position = meta_model_data.estimated_cornea_positions[j] + eye_position_offset + (cv::Point3d) (k * visual_axis);
            vec1 = meta_model_data.real_marker_positions[j] - meta_model_data.real_eye_position;
            vec2 = predicted_marker_position - meta_model_data.real_eye_position;
            angle_error = std::acos(vec1.dot(vec2) / (cv::norm(vec1) * cv::norm(vec2))) * 180.0 / CV_PI;
            angle_errors_poly_fit.push_back(angle_error);
            data_point.push_back(predicted_angle[0]);
            data_point.push_back(predicted_angle[1]);

            data_point.push_back(0);
            data_point.push_back(0);
            full_data.push_back(data_point);
        }

        Utils::writeFloatCsv(full_data, "meta_model_data_full.csv");

        std::cout << std::setprecision(3) << std::fixed;

        auto mean_error = Utils::getMean<double>(position_errors);
        auto std_error = Utils::getStdDev(position_errors);
        std::cout << "Position error: " << mean_error << " ± " << std_error << std::endl;

        mean_error = Utils::getMean<double>(angle_errors_poly_fit);
        std_error = Utils::getStdDev<double>(angle_errors_poly_fit);
        std::cout << "Polynomial fit error: " << mean_error << " ± " << std_error << std::endl;

        mean_error = Utils::getMean<double>(angle_errors_pcr);
        std_error = Utils::getStdDev<double>(angle_errors_pcr);
        std::cout << "Glint-pupil error: " << mean_error << " ± " << std_error << std::endl;
    }
} // et
