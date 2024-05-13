#include "eye_tracker/optimizers/MetaModel.hpp"
#include "eye_tracker/Utils.hpp"
#include "eye_tracker/eye/EyeEstimator.hpp"
#include "eye_tracker/input/InputVideo.hpp"
#include "eye_tracker/image/CameraFeatureAnalyser.hpp"

#include <fstream>
#include <random>
#include <utility>

namespace et {

    bool MetaModel::ransac = false;
    bool MetaModel::calibration_enabled = true;

    MetaModel::MetaModel(int camera_id) : camera_id_(camera_id) {
        cornea_optimizer_ = new CorneaOptimizer();
        cornea_minimizer_function_ = cv::Ptr<cv::DownhillSolver::Function>(cornea_optimizer_);
        cornea_solver_ = cv::DownhillSolver::create();
        cornea_solver_->setFunction(cornea_minimizer_function_);
        cornea_solver_->setTermCriteria(cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 1000, std::numeric_limits<float>::min()));

        eye_estimator_ = std::make_shared<ModelEyeEstimator>(camera_id);
    }

    std::vector<std::vector<double>> MetaModel::findMetaModel(std::vector<CalibrationInput> const& calibration_input, CalibrationOutput const& calibration_output, bool from_scratch, const std::string& output_video_name) {
        auto eye_measurements = Settings::parameters.polynomial_params[camera_id_].eye_measurements;

        double marker_depth = calibration_output.marker_positions[0].z; // Assuming that all markers are at the same depth

        std::vector<std::vector<double>> full_data{};
        if (!output_video_name.empty()) {
            int current_marker = 0;
            for (int i = 0; i < calibration_input.size(); i++) {
                std::vector<double> data_point{};
                data_point.push_back(calibration_output.eye_position.x);
                data_point.push_back(calibration_output.eye_position.y);
                data_point.push_back(calibration_output.eye_position.z);
                data_point.push_back(calibration_output.marker_positions[current_marker].x);
                data_point.push_back(calibration_output.marker_positions[current_marker].y);
                data_point.push_back(calibration_output.marker_positions[current_marker].z);
                data_point.push_back(calibration_input[i].timestamp);
                if (current_marker < calibration_output.marker_positions.size() - 1 && calibration_input[i].timestamp > calibration_output.timestamps[current_marker]) {
                    current_marker++;
                }
                full_data.push_back(data_point);
            }

            std::string header = "real_eye_x,real_eye_y,real_eye_z,real_marker_x,real_marker_y,real_marker_z,timestamp";

            Utils::writeFloatCsv(full_data, output_video_name + ".csv", false, header);
            full_data.clear();
        }

        int total_markers = 0;

        MetaModelData meta_model_data{};
        meta_model_data.real_eye_position = calibration_output.eye_position;

        std::vector<int> cum_samples_per_marker{};
        std::vector<int> marker_numbers{};
        cum_samples_per_marker.push_back(0);
        cum_samples_per_marker.push_back(0);
        double start_timestamp = 0;
        int counter = 0;

        for (const auto& sample: calibration_input) {
            if (!sample.detected) {
                continue;
            }
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

            cv::Vec3d optical_axis;
            Utils::anglesToVector({x.at<double>(0, 0), x.at<double>(0, 1)}, optical_axis);

            real_cornea_position = meta_model_data.real_eye_position + eye_measurements.cornea_curvature_radius * (cv::Point3d) optical_axis;

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
            meta_model_data.angle_offsets.emplace_back(real_angles - sample.angles);
            marker_numbers.push_back(total_markers);

            cum_samples_per_marker[total_markers + 1]++;
        }
        total_markers++;

//        if (remove_outliers) {
//            std::vector<int> position_outliers = Utils::getOutliers(meta_model_data.estimated_eye_positions, 2.0);
//            std::vector<int> cornea_outliers = Utils::getOutliers(meta_model_data.estimated_cornea_positions, 2.0);
//            std::vector<int> theta_outliers = Utils::getOutliers(meta_model_data.estimated_angles_theta, 2.0);
//            std::vector<int> phi_outliers = Utils::getOutliers(meta_model_data.estimated_angles_phi, 2.0);
//            std::vector<int> x_outliers = Utils::getOutliers(meta_model_data.pcr_distances_x, 2.0);
//            std::vector<int> y_outliers = Utils::getOutliers(meta_model_data.pcr_distances_y, 2.0);
//            std::vector<int> all_outliers{};
//            all_outliers.insert(all_outliers.end(), position_outliers.begin(), position_outliers.end());
//            all_outliers.insert(all_outliers.end(), cornea_outliers.begin(), cornea_outliers.end());
//            all_outliers.insert(all_outliers.end(), theta_outliers.begin(), theta_outliers.end());
//            all_outliers.insert(all_outliers.end(), phi_outliers.begin(), phi_outliers.end());
//            all_outliers.insert(all_outliers.end(), x_outliers.begin(), x_outliers.end());
//            all_outliers.insert(all_outliers.end(), y_outliers.begin(), y_outliers.end());
//            std::sort(all_outliers.begin(), all_outliers.end());
//            all_outliers.erase(std::unique(all_outliers.begin(), all_outliers.end()), all_outliers.end());
//
//            for (int i = 0; i < all_outliers.size(); i++) {
//                int index = all_outliers[i] - i; // -i to correct for the fact that the vector is shrinking
//                meta_model_data.estimated_eye_positions.erase(meta_model_data.estimated_eye_positions.begin() + index);
//                meta_model_data.estimated_cornea_positions.erase(meta_model_data.estimated_cornea_positions.begin() + index);
//                meta_model_data.estimated_angles_theta.erase(meta_model_data.estimated_angles_theta.begin() + index);
//                meta_model_data.estimated_angles_phi.erase(meta_model_data.estimated_angles_phi.begin() + index);
//                meta_model_data.real_cornea_positions.erase(meta_model_data.real_cornea_positions.begin() + index);
//                meta_model_data.real_marker_positions.erase(meta_model_data.real_marker_positions.begin() + index);
//                meta_model_data.real_angles_theta.erase(meta_model_data.real_angles_theta.begin() + index);
//                meta_model_data.real_angles_phi.erase(meta_model_data.real_angles_phi.begin() + index);
//                meta_model_data.pcr_distances_x.erase(meta_model_data.pcr_distances_x.begin() + index);
//                meta_model_data.pcr_distances_y.erase(meta_model_data.pcr_distances_y.begin() + index);
//                int erased_marker = marker_numbers[index];
//                for (int j = erased_marker + 1; j < cum_samples_per_marker.size(); j++) {
//                    cum_samples_per_marker[j]--;
//                }
//                marker_numbers.erase(marker_numbers.begin() + index);
//            }
//        }

        int total_samples = static_cast<int>(meta_model_data.estimated_eye_positions.size());

        std::vector<bool> best_x_y_samples{};
        std::vector<bool> best_theta_phi_samples{};

        auto mean_real_cornea_position = Utils::getMean<cv::Point3d>(meta_model_data.real_cornea_positions);
        auto mean_estimated_eye_position = Utils::getTrimmmedMean(meta_model_data.estimated_eye_positions, 0.5);
        auto mean_estimated_cornea_position = Utils::getMean<cv::Point3d>(meta_model_data.estimated_cornea_positions);
        auto eye_position_offset = mean_real_cornea_position - mean_estimated_cornea_position;
//        auto eye_position_offset = meta_model_data.real_eye_position - mean_estimated_eye_position;

        auto angle_offset = Utils::getMean<cv::Point2d>(meta_model_data.angle_offsets);

        std::cout << "Cornea offset: " << eye_position_offset << std::endl;
        std::cout << "Angle offset: " << angle_offset << std::endl;


//        auto mean_estimated_eye_position = Utils::getTrimmmedMean(meta_model_data.estimated_eye_positions, 0.5);
//        auto eye_position_offset = meta_model_data.real_eye_position - mean_estimated_eye_position;

//        eye_position_offset = {28.6498, 8.7856, 26.9740};

        // std::cout << "Centre offset: " << eye_position_offset << std::endl;

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
            int trials_num = 1;
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
                    if (start_index == end_index + 1) {
                        start_index--;
                    }
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

                //std::cout << i << "\n";

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
                if (best_x_y_samples[i] || !ransac) {
                    input_x_y_sample.push_back({meta_model_data.pcr_distances_x[i], meta_model_data.pcr_distances_y[i]});
                    output_x_sample.push_back(meta_model_data.real_marker_positions[i].x);
                    output_y_sample.push_back(meta_model_data.real_marker_positions[i].y);
                }
                if (best_theta_phi_samples[i] || !ransac) {
                    input_theta_phi_sample.push_back({meta_model_data.estimated_angles_theta[i], meta_model_data.estimated_angles_phi[i]});
                    output_theta_sample.push_back(meta_model_data.real_angles_theta[i]);
                    output_phi_sample.push_back(meta_model_data.real_angles_phi[i]);
                }
            }

            polynomial_fit_pcr_x->fit(input_x_y_sample, &output_x_sample);
            polynomial_fit_pcr_y->fit(input_x_y_sample, &output_y_sample);
            polynomial_fit_theta->fit(input_theta_phi_sample, &output_theta_sample);
            polynomial_fit_phi->fit(input_theta_phi_sample, &output_phi_sample);

            if (!calibration_enabled) {
                eye_position_offset = {0, 0, 0};
                angle_offset = {0, 0};
            }

            et::Settings::parameters.user_params[camera_id_]->position_offset = eye_position_offset;
            et::Settings::parameters.user_params[camera_id_]->polynomial_x = polynomial_fit_pcr_x->getCoefficients();
            et::Settings::parameters.user_params[camera_id_]->polynomial_y = polynomial_fit_pcr_y->getCoefficients();
            et::Settings::parameters.user_params[camera_id_]->polynomial_theta = polynomial_fit_theta->getCoefficients();
            et::Settings::parameters.user_params[camera_id_]->polynomial_phi = polynomial_fit_phi->getCoefficients();
            et::Settings::parameters.user_params[camera_id_]->marker_depth = marker_depth;
            et::Settings::parameters.user_params[camera_id_]->angle_offset = angle_offset;
            et::Settings::saveSettings();
        } else {
            eye_position_offset = et::Settings::parameters.user_params[camera_id_]->position_offset;
            polynomial_fit_pcr_x->setCoefficients(et::Settings::parameters.user_params[camera_id_]->polynomial_x);
            polynomial_fit_pcr_y->setCoefficients(et::Settings::parameters.user_params[camera_id_]->polynomial_y);
            polynomial_fit_theta->setCoefficients(et::Settings::parameters.user_params[camera_id_]->polynomial_theta);
            polynomial_fit_phi->setCoefficients(et::Settings::parameters.user_params[camera_id_]->polynomial_phi);
            marker_depth = et::Settings::parameters.user_params[camera_id_]->marker_depth;
            angle_offset = et::Settings::parameters.user_params[camera_id_]->angle_offset;

            if (!calibration_enabled) {
                eye_position_offset = {0, 0, 0};
                angle_offset = {0, 0};
            }
        }

        std::vector<double> position_errors{};
        std::vector<double> angle_errors_offset{};
        std::vector<double> angle_errors_pcr{};
        std::vector<double> angle_errors_poly_fit{};
        full_data.clear();

        std::vector<std::vector<double>> errors{};
        std::vector<double> temp_angle_errors_offset{};
        std::vector<double> temp_angle_errors_pcr{};
        std::vector<double> temp_angle_errors_poly_fit{};

        for (int j = 0; j < total_samples; j++) {
            std::vector<double> data_point{};

            data_point.push_back(meta_model_data.real_eye_position.x);
            data_point.push_back(meta_model_data.real_eye_position.y);
            data_point.push_back(meta_model_data.real_eye_position.z);
            data_point.push_back(meta_model_data.estimated_eye_positions[j].x + eye_position_offset.x);
            data_point.push_back(meta_model_data.estimated_eye_positions[j].y + eye_position_offset.y);
            data_point.push_back(meta_model_data.estimated_eye_positions[j].z + eye_position_offset.z);
            data_point.push_back(meta_model_data.real_cornea_positions[j].x);
            data_point.push_back(meta_model_data.real_cornea_positions[j].y);
            data_point.push_back(meta_model_data.real_cornea_positions[j].z);
            data_point.push_back(meta_model_data.estimated_cornea_positions[j].x + eye_position_offset.x);
            data_point.push_back(meta_model_data.estimated_cornea_positions[j].y + eye_position_offset.y);
            data_point.push_back(meta_model_data.estimated_cornea_positions[j].z + eye_position_offset.z);
            data_point.push_back(meta_model_data.real_angles_theta[j]);
            data_point.push_back(meta_model_data.real_angles_phi[j]);
            data_point.push_back(meta_model_data.real_marker_positions[j].x);
            data_point.push_back(meta_model_data.real_marker_positions[j].y);
            data_point.push_back(meta_model_data.real_marker_positions[j].z);

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
            data_point.push_back(predicted_marker_position.x);
            data_point.push_back(predicted_marker_position.y);
            data_point.push_back(predicted_marker_position.z);

            double estimated_theta = polynomial_fit_theta->getEstimation({meta_model_data.estimated_angles_theta[j], meta_model_data.estimated_angles_phi[j]});
            double estimated_phi = polynomial_fit_phi->getEstimation({meta_model_data.estimated_angles_theta[j], meta_model_data.estimated_angles_phi[j]});

            if (!calibration_enabled) {
                estimated_theta = meta_model_data.estimated_angles_theta[j];
                estimated_phi = meta_model_data.estimated_angles_phi[j];
            }

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
            data_point.push_back(predicted_marker_position.x);
            data_point.push_back(predicted_marker_position.y);
            data_point.push_back(predicted_marker_position.z);

            predicted_angle = {meta_model_data.estimated_angles_theta[j] + angle_offset.x, meta_model_data.estimated_angles_phi[j] + angle_offset.y};
            Utils::anglesToVector(predicted_angle, visual_axis);
            k = (marker_depth - meta_model_data.estimated_cornea_positions[j].z - eye_position_offset.z) / visual_axis[2];
            predicted_marker_position = meta_model_data.estimated_cornea_positions[j] + eye_position_offset + (cv::Point3d) (k * visual_axis);
            vec1 = meta_model_data.real_marker_positions[j] - meta_model_data.real_eye_position;
            vec2 = predicted_marker_position - meta_model_data.real_eye_position;
            angle_error = std::acos(vec1.dot(vec2) / (cv::norm(vec1) * cv::norm(vec2))) * 180.0 / CV_PI;
            angle_errors_offset.push_back(angle_error);
            data_point.push_back(predicted_angle[0]);
            data_point.push_back(predicted_angle[1]);
            data_point.push_back(predicted_marker_position.x);
            data_point.push_back(predicted_marker_position.y);
            data_point.push_back(predicted_marker_position.z);

            full_data.push_back(data_point);

            if (j == total_samples -1 || (j > 0 && (meta_model_data.real_marker_positions[j].x != meta_model_data.real_marker_positions[j - 1].x || meta_model_data.real_marker_positions[j].y != meta_model_data.real_marker_positions[j - 1].y))) {
                if (j == total_samples - 1) {
                    temp_angle_errors_offset.push_back(angle_errors_offset.back());
                    temp_angle_errors_poly_fit.push_back(angle_errors_poly_fit.back());
                    temp_angle_errors_pcr.push_back(angle_errors_pcr.back());
                }

                errors.push_back({Utils::getMean<double>(temp_angle_errors_offset), Utils::getMean<double>(temp_angle_errors_poly_fit), Utils::getMean<double>(temp_angle_errors_pcr)});
                temp_angle_errors_offset.clear();
                temp_angle_errors_poly_fit.clear();
                temp_angle_errors_pcr.clear();
            }

            temp_angle_errors_offset.push_back(angle_errors_offset.back());
            temp_angle_errors_poly_fit.push_back(angle_errors_poly_fit.back());
            temp_angle_errors_pcr.push_back(angle_errors_pcr.back());

//            errors.push_back({angle_errors_offset.back(), angle_errors_poly_fit.back(), angle_errors_pcr.back()});

        }

        std::string header = "real_eye_x,real_eye_y,real_eye_z,est_eye_x,est_eye_y,est_eye_z,real_cornea_x,real_cornea_y,real_cornea_z,"
                             "est_cornea_x,est_cornea_y,est_cornea_z,real_theta,real_phi,real_point_x,real_point_y,real_point_z,pcr_theta,pcr_phi,pcr_point_x,"
                             "pcr_point_y,pcr_point_z,est_theta,est_phi,est_point_x,est_point_y,est_point_z,offset_theta,offset_phi,offset_point_x,"
                             "offset_point_y,offset_point_z";

        Utils::writeFloatCsv(full_data, "full_data.csv", false, header);

        std::cout << std::setprecision(3) << std::fixed;

        auto mean_error = Utils::getMean<double>(position_errors);
        auto std_error = Utils::getStdDev(position_errors);
        std::cout << "Position error: " << mean_error << " ± " << std_error << std::endl;

        mean_error = Utils::getMean<double>(angle_errors_offset);
        std_error = Utils::getStdDev<double>(angle_errors_offset);
        std::cout << "Offset error: " << mean_error << " ± " << std_error << std::endl;

        mean_error = Utils::getMean<double>(angle_errors_poly_fit);
        std_error = Utils::getStdDev<double>(angle_errors_poly_fit);
        std::cout << "Polynomial fit error: " << mean_error << " ± " << std_error << std::endl;

        mean_error = Utils::getMean<double>(angle_errors_pcr);
        std_error = Utils::getStdDev<double>(angle_errors_pcr);
        std::cout << "Glint-pupil error: " << mean_error << " ± " << std_error << std::endl;

        return std::move(errors);
    }

    std::vector<std::vector<double>> MetaModel::findMetaModelFromFile(const std::string& calibration_input_video_path, const std::string& calibration_output_csv_path, bool from_scratch) {
        std::vector<CalibrationInput> calibration_input{};
        CalibrationOutput calibration_output{};
        double time_per_marker = 3;

        auto calibration_output_data = Utils::readFloatRowsCsv(calibration_output_csv_path, true);

        calibration_output.eye_position = {calibration_output_data[0][0], calibration_output_data[0][1], calibration_output_data[0][2]};
        calibration_output.timestamps.push_back(time_per_marker);
        calibration_output.marker_positions.emplace_back(calibration_output_data[0][3], calibration_output_data[0][4], calibration_output_data[0][5]);
        for (int i = 1; i < calibration_output_data.size(); i++) {
            if (calibration_output_data[i][3] != calibration_output_data[i - 1][3] || calibration_output_data[i][4] != calibration_output_data[i - 1][4] || calibration_output_data[i][5] != calibration_output_data[i - 1][5]) {
                calibration_output.timestamps.push_back(time_per_marker * static_cast<int>(1 + calibration_output.timestamps.size()));
                calibration_output.marker_positions.emplace_back(calibration_output_data[i][3], calibration_output_data[i][4], calibration_output_data[i][5]);
            }
        }

        auto image_provider = std::make_shared<InputVideo>(calibration_input_video_path);
        auto feature_detector = std::make_shared<CameraFeatureAnalyser>(camera_id_);
        auto eye_estimator = std::make_shared<ModelEyeEstimator>(camera_id_);
        for (int i = 0; i < calibration_output_data.size(); i++) {
            auto analyzed_frame = image_provider->grabImage();
            const auto now = std::chrono::system_clock::now();
            if (analyzed_frame.pupil.empty() || analyzed_frame.glints.empty()) {
                break;
            }

            feature_detector->preprocessImage(analyzed_frame);
            bool features_found = feature_detector->findPupil();
            cv::Point2d pupil = feature_detector->getPupilUndistorted();
            features_found &= feature_detector->findEllipsePoints();
            auto glints = feature_detector->getGlints();
            auto glints_validity = feature_detector->getGlintsValidity();
            cv::RotatedRect ellipse = feature_detector->getEllipseUndistorted();
            int pupil_radius = feature_detector->getPupilRadiusUndistorted();
            cv::Point3d cornea_centre{};
            if (features_found) {
                EyeInfo eye_info = {
                        .pupil = pupil,
                        .pupil_radius = (double) pupil_radius,
                        .glints = *glints,
                        .glints_validity = *glints_validity,
                        .ellipse = ellipse
                };
                eye_estimator->findEye(eye_info, false);
                eye_estimator->getCorneaCurvaturePosition(cornea_centre);
                auto gaze_point = eye_estimator->getNormalizedGazePoint();
            }

            CalibrationInput sample{};
            sample.detected = features_found;
            sample.timestamp = calibration_output_data[i][6];
            if (features_found) {
                eye_estimator->getEyeCentrePosition(sample.eye_position);
                sample.cornea_position = cornea_centre;
                cv::Vec3d gaze_direction;
                eye_estimator->getGazeDirection(gaze_direction);
                Utils::vectorToAngles(gaze_direction, sample.angles);
                sample.pcr_distance = pupil - (cv::Point2d) ellipse.center;
            }
            calibration_input.push_back(sample);
        }

        auto errors = findMetaModel(calibration_input, calibration_output, from_scratch, "");
        return std::move(errors);
    }

    std::vector<std::vector<double>> MetaModel::getEstimationsAtFrames(const std::string& calibration_input_video_path, const std::string& frames_csv) {
        std::vector<CalibrationInput> calibration_input{};
        CalibrationOutput calibration_output{};
        double time_per_marker = 3;

        auto calibration_output_data = Utils::readFloatRowsCsv(frames_csv, false);
        int current_index = 0;


        auto image_provider = std::make_shared<InputVideo>(calibration_input_video_path);
        auto feature_detector = std::make_shared<CameraFeatureAnalyser>(camera_id_);
        auto eye_estimator = std::make_shared<ModelEyeEstimator>(camera_id_);

        std::vector<std::vector<double>> estimations{};
        cv::Point3d cornea_centre{};

        int frames_back = 0;

        for (int i = 0;; i++) {
            auto analyzed_frame = image_provider->grabImage();

            const auto now = std::chrono::system_clock::now();
            if (analyzed_frame.pupil.empty() || analyzed_frame.glints.empty()) {
                break;
            }

            feature_detector->preprocessImage(analyzed_frame);
            bool features_found = feature_detector->findPupil();
            cv::Point2d pupil = feature_detector->getPupilUndistorted();
            features_found &= feature_detector->findEllipsePoints();
            auto glints = feature_detector->getGlints();
            auto glints_validity = feature_detector->getGlintsValidity();
            cv::RotatedRect ellipse = feature_detector->getEllipseUndistorted();
            int pupil_radius = feature_detector->getPupilRadiusUndistorted();
            if (features_found) {
                EyeInfo eye_info = {
                        .pupil = pupil,
                        .pupil_radius = (double) pupil_radius,
                        .glints = *glints,
                        .glints_validity = *glints_validity,
                        .ellipse = ellipse
                };
                eye_estimator->findEye(eye_info, true);
                eye_estimator->getCorneaCurvaturePosition(cornea_centre);
                auto gaze_point = eye_estimator->getNormalizedGazePoint();
            }
            if (i == (int) (calibration_output_data[current_index][0]) - frames_back) {
                cv::Point2d real_position = {calibration_output_data[current_index][1], calibration_output_data[current_index][2]};
                cv::Point2d estimated_position = {cornea_centre.x, cornea_centre.y};
                double distance = cv::norm(real_position - estimated_position);
                estimations.push_back({cornea_centre.x, cornea_centre.y});
                current_index++;
                std::cout << estimated_position.x << "," << estimated_position.y << ";" << std::endl;
                if (current_index == calibration_output_data.size()) {
                    break;
                }
            }
        }
        return std::move(estimations);
    }

} // et
