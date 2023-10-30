#include "eye_tracker/optimizers/MetaModel.hpp"
#include "eye_tracker/image/CameraFeatureAnalyser.hpp"
#include "eye_tracker/Utils.hpp"
#include "eye_tracker/eye/EyeEstimator.hpp"
#include "eye_tracker/eye/PolynomialEyeEstimator.hpp"
#include "eye_tracker/optimizers/AggregatedPolynomialOptimizer.hpp"
#include "eye_tracker/frameworks/Framework.hpp"
#include "eye_tracker/eye/ModelEyeEstimator.hpp"

#include <fstream>

namespace et
{
    MetaModel::MetaModel(int camera_id) : camera_id_(camera_id)
    {
        eye_centre_optimizer_ = new EyeCentreOptimizer();
        eye_centre_minimizer_function_ = cv::Ptr<cv::DownhillSolver::Function>(eye_centre_optimizer_);
        eye_centre_solver_ = cv::DownhillSolver::create();
        eye_centre_solver_->setFunction(eye_centre_minimizer_function_);

        optical_axis_optimizer_ = new OpticalAxisOptimizer();
        optical_axis_minimizer_function_ = cv::Ptr<cv::DownhillSolver::Function>(optical_axis_optimizer_);
        optical_axis_solver_ = cv::DownhillSolver::create();
        optical_axis_solver_->setFunction(optical_axis_minimizer_function_);

        eye_angles_optimizer_ = new EyeAnglesOptimizer();
        eye_angles_minimizer_function_ = cv::Ptr<cv::DownhillSolver::Function>(eye_angles_optimizer_);
        eye_angles_solver_ = cv::DownhillSolver::create();
        eye_angles_solver_->setFunction(eye_angles_minimizer_function_);
    }

    cv::Point3f MetaModel::findMetaModel(std::shared_ptr<ImageProvider> image_provider,
                                  std::shared_ptr<FeatureAnalyser> feature_analyser, std::string path_to_csv)
    {
        Framework::mutex.lock();

        // Open CSV for reading
        auto csv_file = Utils::readFloatRowsCsv(path_to_csv);
        int current_row = 0;

        std::vector<std::vector<cv::Point3f>> all_front_corners{};
        std::vector<std::vector<cv::Point3f>> all_back_corners{};
        std::vector<cv::Vec3f> all_marker_positions{};
        std::vector<cv::Point2f> camera_pupils{};
        std::vector<cv::RotatedRect> camera_ellipses{};
        std::vector<double> alphas{};
        std::vector<double> betas{};


        std::vector<cv::Point3f> front_corners{};
        std::vector<cv::Point3f> back_corners{};
        for (int i = 0; i < 4; i++)
        {
            front_corners.push_back({csv_file[0][1 + i * 3], csv_file[0][2 + i * 3],
                                     csv_file[0][3 + i * 3]});
        }
        for (int i = 0; i < 4; i++)
        {
            back_corners.push_back({csv_file[0][13 + i * 3], csv_file[0][14 + i * 3],
                                    csv_file[0][15 + i * 3]});
        }

        cv::Point3f cross_point = Utils::findGridIntersection(front_corners, back_corners);
        auto model_eye_estimator = std::make_shared<ModelEyeEstimator>(camera_id_, cross_point);

        while (true)
        {
            auto analyzed_frame_ = image_provider->grabImage();
            if (analyzed_frame_.pupil.empty() || analyzed_frame_.glints.empty())
            {
                break;
            }

            while (current_row < csv_file.size() && csv_file[current_row][0] < analyzed_frame_.frame_num)
            {
                current_row++;
            }

            if (current_row != analyzed_frame_.frame_num)
            {
                continue;
            }

            feature_analyser->preprocessImage(analyzed_frame_);
            bool features_found = feature_analyser->findPupil() && feature_analyser->findEllipsePoints();
            if (!features_found)
            {
                continue;
            }

            cv::Point2f pupil = feature_analyser->getPupilUndistorted();
            cv::RotatedRect ellipse = feature_analyser->getEllipseUndistorted();
            ellipse.angle = 0.0f; // Required due to the problems with angle estimation.

            all_front_corners.push_back(front_corners);
            all_back_corners.push_back(back_corners);
            all_marker_positions.push_back(
                    {csv_file[current_row][25], csv_file[current_row][26], csv_file[current_row][27]});
            camera_pupils.push_back(pupil);
            camera_ellipses.push_back(ellipse);

            auto glints = feature_analyser->getGlints();
            cv::Point2f top_left_side_glint = glints->at(0);
            cv::Point2f bottom_right_side_glint = glints->at(0);
            cv::Point2f mean_glint = {0.0f, 0.0f};
            for (const auto &glint : *glints)
            {
                mean_glint += glint;
            }
            mean_glint.x /= glints->size();
            mean_glint.y /= glints->size();
            for (const auto &glint : *glints)
            {
                if (glint.x < mean_glint.x && glint.y < mean_glint.y)
                {
                    top_left_side_glint = glint;
                }
                if (glint.x > mean_glint.x && glint.y > mean_glint.y)
                {
                    bottom_right_side_glint = glint;
                }
            }

            EyeInfo eye_info = {
                    .pupil = pupil,
                    .glints = {top_left_side_glint, bottom_right_side_glint},
            };

            cv::Point3f nodal_point, eye_centre, visual_axis;
            model_eye_estimator->detectEye(eye_info, nodal_point, eye_centre, visual_axis);

            float alpha{0};
            float beta{0};
            for (int i = 0; i < 1; i++)
            {
                eye_centre_optimizer_->setParameters(front_corners, back_corners, alpha, beta, 5.3);
                cv::Vec3d starting_point = eye_centre_optimizer_->getCrossPoint();

                cv::Mat x = (cv::Mat_<double>(1, 3) << starting_point[0], starting_point[1], starting_point[2]);
                eye_centre_optimizer_->calc(x.ptr<double>());
                cv::Mat step = cv::Mat::ones(x.rows, x.cols, CV_64F) * 0.1;
                eye_centre_solver_->setInitStep(step);
                eye_centre_solver_->minimize(x);

                cv::Point3f grid_eye_centre = cv::Point3f(x.at<double>(0, 0), x.at<double>(0, 1), x.at<double>(0, 2));

//                cv::Vec3f calculated_optical_axis = nodal_point - grid_eye_centre;
                cv::Vec3f calculated_optical_axis = nodal_point - eye_centre;
                calculated_optical_axis = calculated_optical_axis / cv::norm(calculated_optical_axis);

                cv::Vec3f nodal_point_v = nodal_point;

                cv::Vec3f calculated_visual_axis = all_marker_positions.back() - nodal_point_v;
                calculated_visual_axis = calculated_visual_axis / cv::norm(calculated_visual_axis);

                eye_angles_optimizer_->setParameters(calculated_visual_axis, calculated_optical_axis);
                x = (cv::Mat_<double>(1, 2) << 0.0, 0.0);
                step = cv::Mat::ones(x.rows, x.cols, CV_64F) * 0.1;
                eye_angles_solver_->setInitStep(step);
                eye_angles_solver_->minimize(x);

                alpha = x.at<double>(0, 0);
                beta = x.at<double>(0, 1);
            }

            alphas.push_back(alpha);
            betas.push_back(beta);
        }

        double mean_alpha = std::accumulate(alphas.begin(), alphas.end(), 0.0) / alphas.size();
        double mean_beta = std::accumulate(betas.begin(), betas.end(), 0.0) / betas.size();

        double std_alpha = 0.0;
        double std_beta = 0.0;
        for (int i = 0; i < alphas.size(); i++)
        {
            std_alpha += (alphas[i] - mean_alpha) * (alphas[i] - mean_alpha);
            std_beta += (betas[i] - mean_beta) * (betas[i] - mean_beta);
        }
        std_alpha /= alphas.size();
        std_beta /= betas.size();
        std_alpha = sqrt(std_alpha);
        std_beta = sqrt(std_beta);


        std::clog << "Mean alpha: " << mean_alpha << std::endl;
        std::clog << "Mean beta: " << mean_beta << std::endl;

        std::clog << "Std alpha: " << std_alpha << std::endl;
        std::clog << "Std beta: " << std_beta << std::endl;

        int frames = all_front_corners.size();
        int samples_per_ellipse = 5;
        cv::Mat camera_features(frames * (samples_per_ellipse + 1), 4, CV_32F);
        cv::Mat blender_features(frames * (samples_per_ellipse + 1), 4, CV_32F);
        for (int i = 0; i < frames; i++)
        {
            camera_features.at<float>(i * (samples_per_ellipse + 1), 0) = camera_pupils[i].x;
            camera_features.at<float>(i * (samples_per_ellipse + 1), 1) = camera_pupils[i].y;
            camera_features.at<float>(i * (samples_per_ellipse + 1), 2) = 0.0f;
            camera_features.at<float>(i * (samples_per_ellipse + 1), 3) = 1.0f;
            for (int j = 0; j < samples_per_ellipse; j++)
            {
                // Get angle on ellipse between 0 and 2pi
                float angle = (float) j / samples_per_ellipse * 2 * CV_PI;

                // Find intersection between ellipse and line with angle
                cv::Point2f intersection = Utils::findEllipseIntersection(camera_ellipses[i], angle);
                camera_features.at<float>(i * (samples_per_ellipse + 1) + j + 1, 0) = intersection.x;
                camera_features.at<float>(i * (samples_per_ellipse + 1) + j + 1, 1) = intersection.y;
                camera_features.at<float>(i * (samples_per_ellipse + 1) + j + 1, 2) = 0.0f;
                camera_features.at<float>(i * (samples_per_ellipse + 1) + j + 1, 3) = 1.0f;
            }
        }

        float smallest_error = std::numeric_limits<float>::max();
        int best_polynomial_num = 0;

        std::vector<float> errors{};

        for (auto &polynomial_params: Settings::parameters.polynomial_params[camera_id_])
        {
            int num = polynomial_params.first;
            SetupVariables &setup_variables = polynomial_params.second.setup_variables;
            std::shared_ptr<PolynomialEyeEstimator> eye_estimator = std::make_shared<PolynomialEyeEstimator>(camera_id_,
                                                                                                             num);

            std::vector<cv::Point2f> blender_pupils{};
            std::vector<cv::RotatedRect> blender_ellipses{};
            std::vector<cv::Vec3f> visual_axes{};
            std::vector<cv::Vec3f> eye_centres{};

            static float ellipse_points[][2] = {{0.0f,  0.0f},
                                                {0.0f,  1.0f},
                                                {0.0f,  -1.0f},
                                                {1.0f,  0.0f},
                                                {-1.0f, 0.0f},};

            for (int i = 0; i < frames; i++)
            {
                eye_centre_optimizer_->setParameters(all_front_corners[i], all_back_corners[i], setup_variables.alpha,
                                                     setup_variables.beta, setup_variables.cornea_centre_distance);

                cv::Vec3d starting_point = eye_centre_optimizer_->getCrossPoint();

                cv::Mat x = (cv::Mat_<double>(1, 3) << starting_point[0], starting_point[1], starting_point[2]);
                eye_centre_optimizer_->calc(x.ptr<double>());
                cv::Mat step = cv::Mat::ones(x.rows, x.cols, CV_64F) * 0.1;
                eye_centre_solver_->setInitStep(step);
                eye_centre_solver_->minimize(x);

                cv::Vec3f eye_centre = cv::Vec3f(x.at<double>(0, 0), x.at<double>(0, 1), x.at<double>(0, 2));
                cv::Vec3d eye_centre_d = eye_centre;

                optical_axis_optimizer_->setParameters(setup_variables.alpha, setup_variables.beta,
                                                       setup_variables.cornea_centre_distance, eye_centre,
                                                       all_marker_positions[i]);

                cv::Vec3d init_guess = all_marker_positions[i] - eye_centre;
                init_guess = init_guess / cv::norm(init_guess);

                x = (cv::Mat_<double>(1, 3) << init_guess[0], init_guess[1], init_guess[2]);
                step = cv::Mat::ones(x.rows, x.cols, CV_64F) * 0.1;
                optical_axis_solver_->setInitStep(step);
                optical_axis_solver_->minimize(x);
                cv::Vec3f optical_axis = cv::Vec3f(x.at<double>(0, 0), x.at<double>(0, 1), x.at<double>(0, 2));
                optical_axis = optical_axis / cv::norm(optical_axis);

                cv::Vec3f nodal_point = eye_centre + setup_variables.cornea_centre_distance * optical_axis;
                cv::Vec3f visual_axis = Utils::opticalToVisualAxis(optical_axis, setup_variables.alpha,
                                                                   setup_variables.beta);

                EyeInfo eye_info{};

                cv::Point3f nodal_point_p = nodal_point;
                cv::Point3f eye_centre_p = eye_centre;

                eye_estimator->invertEye(nodal_point_p, eye_centre_p, eye_info);

                cv::Point2f expected_pupil = eye_info.pupil;
                cv::RotatedRect expected_ellipse = eye_info.ellipse;
                expected_ellipse.angle = 0.0f; // Required due to the problems with angle estimation.

                blender_pupils.push_back(expected_pupil);
                blender_ellipses.push_back(expected_ellipse);

                eye_centres.push_back(eye_centre);
                visual_axes.push_back(visual_axis);
            }

            for (int i = 0; i < frames; i++)
            {
                blender_features.at<float>(i * (samples_per_ellipse + 1), 0) = blender_pupils[i].x;
                blender_features.at<float>(i * (samples_per_ellipse + 1), 1) = blender_pupils[i].y;
                blender_features.at<float>(i * (samples_per_ellipse + 1), 2) = 0.0f;
                blender_features.at<float>(i * (samples_per_ellipse + 1), 3) = 1.0f;
                for (int j = 0; j < samples_per_ellipse; j++)
                {
                    // Get angle on ellipse between 0 and 2pi
                    float angle = (float) j / samples_per_ellipse * 2 * CV_PI;

                    // Find intersection between ellipse and line with angle
                    cv::Point2f intersection = Utils::findEllipseIntersection(blender_ellipses[i], angle);
                    blender_features.at<float>(i * (samples_per_ellipse + 1) + j + 1, 0) = intersection.x;
                    blender_features.at<float>(i * (samples_per_ellipse + 1) + j + 1, 1) = intersection.y;
                    blender_features.at<float>(i * (samples_per_ellipse + 1) + j + 1, 2) = 0.0f;
                    blender_features.at<float>(i * (samples_per_ellipse + 1) + j + 1, 3) = 1.0f;
                }
            }

            cv::Mat total_matrix = camera_features.inv(cv::DECOMP_SVD) * blender_features;
            // Calculating whole pipeline
            float error = 0.0f;

            // Convert camera features to blender features
            cv::Mat converted_features = camera_features * total_matrix;
            for (int i = 0; i < frames; i++)
            {
                cv::Point2f pupil = cv::Point2f(converted_features.at<float>(i * (samples_per_ellipse + 1), 0),
                                                converted_features.at<float>(i * (samples_per_ellipse + 1), 1));

                std::vector<cv::Point2f> ellipse_points{};
                for (int j = 0; j < samples_per_ellipse; j++)
                {
                    ellipse_points.push_back(
                            cv::Point2f(converted_features.at<float>(i * (samples_per_ellipse + 1) + j + 1, 0),
                                        converted_features.at<float>(i * (samples_per_ellipse + 1) + j + 1, 1)));
                }

                auto ellipse = cv::fitEllipse(ellipse_points);

                cv::Point3f nodal_point{}, eye_centre{}, visual_axis{};

                EyeInfo eye_info{.pupil = pupil, .ellipse = ellipse};

                eye_estimator->detectEye(eye_info, nodal_point, eye_centre, visual_axis);

                float angle = Utils::getAngleBetweenVectors(visual_axis, visual_axes[i]) * 180 / CV_PI;
                eye_centre.z += (eye_centres[i][2] - eye_centre.z) * 0.9f;
                float distance = cv::norm((cv::Vec3f) (eye_centre) - eye_centres[i]);
                error += std::abs(angle) / AggregatedPolynomialOptimizer::ACCEPTABLE_ANGLE_ERROR +
                         distance / AggregatedPolynomialOptimizer::ACCEPTABLE_DISTANCE_ERROR;
            }
            error /= frames;
            errors.push_back(error);

            std::clog << "Found error: " << error << std::endl;

            if (error < smallest_error)
            {
                smallest_error = error;
                best_polynomial_num = num;
            }
        }
        Settings::parameters.user_polynomial_params[camera_id_] = &Settings::parameters.polynomial_params[camera_id_][best_polynomial_num];
        std::clog << "Accepted model with " << smallest_error << " error" << std::endl;

        int last_id = all_front_corners.size() - 1;
        SetupVariables &setup_variables = Settings::parameters.user_polynomial_params[camera_id_]->setup_variables;
        eye_centre_optimizer_->setParameters(all_front_corners[last_id], all_back_corners[last_id], setup_variables.alpha,
                                             setup_variables.beta, setup_variables.cornea_centre_distance);

        cv::Vec3d starting_point = eye_centre_optimizer_->getCrossPoint();

        cv::Mat x = (cv::Mat_<double>(1, 3) << starting_point[0], starting_point[1], starting_point[2]);
        eye_centre_optimizer_->calc(x.ptr<double>());
        cv::Mat step = cv::Mat::ones(x.rows, x.cols, CV_64F) * 0.1;
        eye_centre_solver_->setInitStep(step);
        eye_centre_solver_->minimize(x);

        cv::Point3f eye_centre = cv::Point3f(x.at<double>(0, 0), x.at<double>(0, 1), x.at<double>(0, 2));

        Framework::mutex.unlock();
        return eye_centre;
    }
} // et