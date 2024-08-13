#include <eye_tracker/optimizers/FineTuner.hpp>
#include <eye_tracker/Utils.hpp>
#include <eye_tracker/input/InputVideo.hpp>
#include <eye_tracker/image/FeatureAnalyser.hpp>
#include <eye_tracker/eye/EyeEstimator.hpp>

#include <fstream>
#include <random>
#include <utility>

namespace et {
    FineTuner::FineTuner(int camera_id) : camera_id_(camera_id) {
        optical_axis_optimizer_ = std::make_shared<OpticalAxisOptimizer>();
        cornea_minimizer_function_ = cv::Ptr<cv::DownhillSolver::Function>(optical_axis_optimizer_);
        cornea_solver_ = cv::DownhillSolver::create();
        cornea_solver_->setFunction(cornea_minimizer_function_);
        cornea_solver_->setTermCriteria(cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 1000, std::numeric_limits<float>::min()));

        eye_estimator_ = std::make_shared<EyeEstimator>(camera_id);
    }

    void FineTuner::calculate(std::vector<CalibrationInput> const& calibration_input, CalibrationOutput const& calibration_output) const {
        auto eye_params = Settings::parameters.eye_params[camera_id_];

        double marker_depth = calibration_output.marker_positions[0].z; // Assuming that all markers are at the same depth

        int total_markers = 0;

        FineTuningData meta_model_data{};
        meta_model_data.real_eye_position = calibration_output.eye_position;

        std::vector<int> cum_samples_per_marker{};
        std::vector<int> marker_numbers{};
        cum_samples_per_marker.push_back(0);
        cum_samples_per_marker.push_back(0);
        double start_timestamp = 0;

        for (const auto& [eye_position, cornea_position, angles, timestamp, detected]: calibration_input) {
            if (!detected) {
                continue;
            }
            if (timestamp > calibration_output.timestamps[calibration_output.timestamps.size() - 1]) {
                break;
            }

            if (timestamp >= calibration_output.timestamps[total_markers]) {
                start_timestamp = calibration_output.timestamps[total_markers];
                total_markers++;
                cum_samples_per_marker.push_back(cum_samples_per_marker[total_markers]);
            }

            if (timestamp - start_timestamp < 1.0) {
                continue;
            }

            cv::Mat x = (cv::Mat_<double>(1, 2) << 0.0, 0.0);
            cv::Mat step = cv::Mat::ones(x.rows, x.cols, CV_64F) * 0.1;
            optical_axis_optimizer_->setParameters(eye_params, calibration_output.eye_position, calibration_output.marker_positions[total_markers]);
            cornea_solver_->setInitStep(step);
            cornea_solver_->minimize(x);
            cv::Point3d real_cornea_position{};

            cv::Vec3d optical_axis;
            Utils::anglesToVector({x.at<double>(0, 0), x.at<double>(0, 1)}, optical_axis);

            real_cornea_position = meta_model_data.real_eye_position + eye_params.cornea_curvature_radius * static_cast<cv::Point3d>(optical_axis);

            meta_model_data.real_cornea_positions.push_back(real_cornea_position);

            cv::Vec3d real_visual_axis = calibration_output.marker_positions[total_markers] - real_cornea_position;
            cv::normalize(real_visual_axis, real_visual_axis);
            cv::Vec2d real_angles{};
            Utils::vectorToAngles(real_visual_axis, real_angles);
            meta_model_data.real_marker_positions.push_back(calibration_output.marker_positions[total_markers]);
            meta_model_data.real_angles_theta.push_back(real_angles[0]);
            meta_model_data.real_angles_phi.push_back(real_angles[1]);

            meta_model_data.estimated_eye_positions.push_back(eye_position);
            meta_model_data.estimated_cornea_positions.push_back(cornea_position);
            meta_model_data.estimated_angles_theta.push_back(angles[0]);
            meta_model_data.estimated_angles_phi.push_back(angles[1]);
            marker_numbers.push_back(total_markers);

            cum_samples_per_marker[total_markers + 1]++;
        }
        total_markers++;

        int total_samples = static_cast<int>(meta_model_data.estimated_eye_positions.size());

        std::vector<bool> best_x_y_samples{};
        std::vector<bool> best_theta_phi_samples{};

        auto mean_real_cornea_position = Utils::getMean<cv::Point3d>(meta_model_data.real_cornea_positions);
        auto mean_estimated_cornea_position = Utils::getMean<cv::Point3d>(meta_model_data.estimated_cornea_positions);
        auto eye_position_offset = mean_real_cornea_position - mean_estimated_cornea_position;

        std::shared_ptr<Polynomial> theta_polynomial = std::make_shared<Polynomial>(2, 2);
        std::shared_ptr<Polynomial> phi_polynomial = std::make_shared<Polynomial>(2, 2);

        std::shared_ptr<Polynomial> theta_polynomial_best = std::make_shared<Polynomial>(2, 2);
        std::shared_ptr<Polynomial> phi_polynomial_best = std::make_shared<Polynomial>(2, 2);

        std::random_device random_device;
        std::mt19937 generator(random_device());

        int min_fitting_size = static_cast<int>(theta_polynomial->getCoefficients().size());
        int trials_num = 10'000;
        std::vector<std::vector<int> > trials{};
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
                auto distribution = std::uniform_int_distribution(start_index, end_index);
                int sample_index = distribution(generator);
                trials[i][j] = sample_index;
            }
        }

        constexpr double threshold_theta_phi = 0.5 * CV_PI / 180.0;

        int best_theta_phi = 0;

        std::vector<std::vector<double> > input_theta_phi_sample{};
        std::vector<double> output_theta_sample{};
        std::vector<double> output_phi_sample{};

        for (int i = 0; i < trials_num; i++) {
            input_theta_phi_sample.clear();
            output_theta_sample.clear();
            output_phi_sample.clear();
            for (int j = 0; j < min_fitting_size; j++) {
                int sample_index = trials[i][j];
                input_theta_phi_sample.push_back({meta_model_data.estimated_angles_theta[sample_index], meta_model_data.estimated_angles_phi[sample_index]});
                output_theta_sample.push_back(meta_model_data.real_angles_theta[sample_index]);
                output_phi_sample.push_back(meta_model_data.real_angles_phi[sample_index]);
            }
            theta_polynomial->fit(input_theta_phi_sample, &output_theta_sample);
            phi_polynomial->fit(input_theta_phi_sample, &output_phi_sample);

            int current_theta_phi = 0;
            std::vector<bool> theta_phi_samples{};
            for (int j = 0; j < total_samples; j++) {
                auto predicted_theta = theta_polynomial->getEstimation({meta_model_data.estimated_angles_theta[j], meta_model_data.estimated_angles_phi[j]});
                auto real_theta = meta_model_data.real_angles_theta[j];

                auto predicted_phi = phi_polynomial->getEstimation({meta_model_data.estimated_angles_theta[j], meta_model_data.estimated_angles_phi[j]});
                auto real_phi = meta_model_data.real_angles_phi[j];

                if (std::abs(predicted_theta - real_theta) < threshold_theta_phi && std::abs(predicted_phi - real_phi) < threshold_theta_phi) {
                    current_theta_phi++;
                    theta_phi_samples.push_back(true);
                } else {
                    theta_phi_samples.push_back(false);
                }
            }
            if (current_theta_phi > best_theta_phi) {
                best_theta_phi = current_theta_phi;
                best_theta_phi_samples = theta_phi_samples;
            }
        }

        input_theta_phi_sample.clear();
        output_theta_sample.clear();
        output_phi_sample.clear();
        for (int i = 0; i < total_samples; i++) {
            if (best_theta_phi_samples[i]) {
                input_theta_phi_sample.push_back({meta_model_data.estimated_angles_theta[i], meta_model_data.estimated_angles_phi[i]});
                output_theta_sample.push_back(meta_model_data.real_angles_theta[i]);
                output_phi_sample.push_back(meta_model_data.real_angles_phi[i]);
            }
        }

        theta_polynomial->fit(input_theta_phi_sample, &output_theta_sample);
        phi_polynomial->fit(input_theta_phi_sample, &output_phi_sample);

        et::Settings::parameters.user_params[camera_id_]->position_offset = eye_position_offset;
        et::Settings::parameters.user_params[camera_id_]->polynomial_theta = theta_polynomial->getCoefficients();
        et::Settings::parameters.user_params[camera_id_]->polynomial_phi = phi_polynomial->getCoefficients();
        et::Settings::parameters.user_params[camera_id_]->marker_depth = marker_depth;
        et::Settings::saveSettings();

        std::vector<double> position_errors{};
        std::vector<double> angle_errors_poly_fit{};

        for (int j = 0; j < total_samples; j++) {
            cv::Vec3d real_visual_axis = meta_model_data.real_marker_positions[j] - meta_model_data.real_cornea_positions[j];
            cv::normalize(real_visual_axis, real_visual_axis);

            cv::Point2d real_eye_position{meta_model_data.real_eye_position.x, meta_model_data.real_eye_position.y};
            cv::Point2d estimated_eye_position{meta_model_data.estimated_eye_positions[j].x + eye_position_offset.x, meta_model_data.estimated_eye_positions[j].y + eye_position_offset.y};
            position_errors.push_back(cv::norm(estimated_eye_position - real_eye_position));

            double estimated_theta = theta_polynomial->getEstimation({meta_model_data.estimated_angles_theta[j], meta_model_data.estimated_angles_phi[j]});
            double estimated_phi = phi_polynomial->getEstimation({meta_model_data.estimated_angles_theta[j], meta_model_data.estimated_angles_phi[j]});

            cv::Vec2d predicted_angle = {estimated_theta, estimated_phi};
            cv::Vec3d visual_axis, vec1, vec2;
            Utils::anglesToVector(predicted_angle, visual_axis);
            double k = (marker_depth - meta_model_data.estimated_cornea_positions[j].z - eye_position_offset.z) / visual_axis[2];
            cv::Point3d predicted_marker_position = meta_model_data.estimated_cornea_positions[j] + eye_position_offset + static_cast<cv::Point3d>(k * visual_axis);
            vec1 = meta_model_data.real_marker_positions[j] - meta_model_data.real_eye_position;
            vec2 = predicted_marker_position - meta_model_data.real_eye_position;
            double angle_error = std::acos(vec1.dot(vec2) / (cv::norm(vec1) * cv::norm(vec2))) * 180.0 / CV_PI;
            angle_errors_poly_fit.push_back(angle_error);
        }

        std::clog << std::setprecision(3) << std::fixed;

        auto mean_error = Utils::getMean<double>(position_errors);
        auto std_error = Utils::getStdDev(position_errors);
        std::clog << "Position error: " << mean_error << " ± " << std_error << std::endl;

        mean_error = Utils::getMean<double>(angle_errors_poly_fit);
        std_error = Utils::getStdDev<double>(angle_errors_poly_fit);
        std::clog << "Polynomial fit error: " << mean_error << " ± " << std_error << std::endl;

        std::clog << "Finished calibration" << std::endl;
    }

    void FineTuner::calculate(const std::string& camera_video_path, const std::string& camera_csv_path) const {
        std::vector<CalibrationInput> calibration_input{};

        CalibrationOutput calibration_output{};
        double time_per_marker = 3;

        auto calibration_output_data = Utils::readCsv(camera_csv_path, true);

        calibration_output.eye_position = {calibration_output_data[0][0], calibration_output_data[0][1], calibration_output_data[0][2]};
        calibration_output.timestamps.push_back(time_per_marker);
        calibration_output.marker_positions.emplace_back(calibration_output_data[0][3], calibration_output_data[0][4], calibration_output_data[0][5]);
        for (int i = 1; i < calibration_output_data.size(); i++) {
            if (calibration_output_data[i][3] != calibration_output_data[i - 1][3] || calibration_output_data[i][4] != calibration_output_data[i - 1][4] || calibration_output_data[i][5] != calibration_output_data[i - 1][5]) {
                calibration_output.timestamps.push_back(time_per_marker * static_cast<int>(1 + calibration_output.timestamps.size()));
                calibration_output.marker_positions.emplace_back(calibration_output_data[i][3], calibration_output_data[i][4], calibration_output_data[i][5]);
            }
        }

        auto image_provider = std::make_shared<InputVideo>(camera_video_path);
        auto feature_detector = std::make_shared<FeatureAnalyser>(camera_id_);
        auto eye_estimator = std::make_shared<EyeEstimator>(camera_id_);
        for (auto& i: calibration_output_data) {
            auto analyzed_frame = image_provider->grabImage();
            const auto now = std::chrono::system_clock::now();
            if (analyzed_frame.frame.empty()) {
                break;
            }

            feature_detector->preprocessImage(analyzed_frame);
            bool features_found = feature_detector->findPupil();
            features_found &= feature_detector->findEllipsePoints();

            cv::Point2d pupil;
            feature_detector->getPupilUndistorted(pupil);


            std::vector<cv::Point2d> glints{};
            feature_detector->getGlints(glints);

            std::vector<bool> glints_validity{};
            feature_detector->getGlintsValidity(glints_validity);

            double pupil_radius;
            feature_detector->getPupilRadiusUndistorted(pupil_radius);

            cv::Point3d cornea_centre{};
            if (features_found) {
                EyeInfo eye_info = {.pupil_centre_position_ics = pupil, .pupil_radius_ics = pupil_radius, .glint_positions_ics = std::move(glints), .glints_valid = std::move(glints_validity)};
                eye_estimator->estimateEyePositions(eye_info, false);
                eye_estimator->getCorneaCentrePositionWCS(cornea_centre);
            }

            CalibrationInput sample{};
            sample.detected = features_found;
            sample.timestamp = i[6];
            if (features_found) {
                eye_estimator->getEyeCentrePositionWCS(sample.eye_position);
                sample.cornea_position = cornea_centre;
                cv::Vec3d gaze_direction;
                eye_estimator->getGazeDirection(gaze_direction);
                Utils::vectorToAngles(gaze_direction, sample.angles);
            }
            calibration_input.push_back(sample);
        }

        calculate(calibration_input, calibration_output);
    }
} // et
