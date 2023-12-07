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

        pixel_pos_optimizer_test_ = new PixelPosOptimizerTest();
        pixel_pos_minimizer_function_test_ = cv::Ptr<cv::DownhillSolver::Function>(pixel_pos_optimizer_test_);
        pixel_pos_solver_test_ = cv::DownhillSolver::create();
        pixel_pos_solver_test_->setFunction(pixel_pos_minimizer_function_test_);

        pixel_pos_optimizer_ = new PixelPosOptimizer();
        pixel_pos_minimizer_function_ = cv::Ptr<cv::DownhillSolver::Function>(pixel_pos_optimizer_);
        pixel_pos_solver_ = cv::DownhillSolver::create();
        pixel_pos_solver_->setFunction(pixel_pos_minimizer_function_);
    }

    cv::Point3d MetaModel::findMetaModel(std::shared_ptr<ImageProvider> image_provider,
                                         std::shared_ptr<FeatureAnalyser> feature_analyser, std::string path_to_csv)
    {
        Framework::mutex.lock();

        // Open CSV for reading
        auto csv_file = Utils::readFloatRowsCsv(path_to_csv);
        int current_row = 0;

        std::vector<std::vector<cv::Point3d>> all_front_corners{};
        std::vector<std::vector<cv::Point3d>> all_back_corners{};
        std::vector<cv::Vec3d> all_marker_positions{};
        std::vector<cv::Point2d> camera_pupils{};
        std::vector<cv::RotatedRect> camera_ellipses{};
        std::vector<double> alphas{};
        std::vector<double> betas{};


        std::vector<cv::Point3d> front_corners{};
        std::vector<cv::Point3d> back_corners{};
        for (int i = 0; i < 4; i++) {
            front_corners.push_back({csv_file[0][1 + i * 3], csv_file[0][2 + i * 3], csv_file[0][3 + i * 3]});
        }
        for (int i = 0; i < 4; i++) {
            back_corners.push_back({csv_file[0][13 + i * 3], csv_file[0][14 + i * 3], csv_file[0][15 + i * 3]});
        }

        cv::Point3d cross_point = Utils::findGridIntersection(front_corners, back_corners);
        auto model_eye_estimator = std::make_shared<ModelEyeEstimator>(camera_id_);

        current_row = 0;
        while (true) {
            auto analyzed_frame_ = image_provider->grabImage();
            if (analyzed_frame_.pupil.empty() || analyzed_frame_.glints.empty()) {
                break;
            }

            while (current_row < csv_file.size() && csv_file[current_row][0] < analyzed_frame_.frame_num) {
                current_row++;
            }

            if (current_row != analyzed_frame_.frame_num) {
                continue;
            }

            feature_analyser->preprocessImage(analyzed_frame_);
            bool features_found = feature_analyser->findPupil() && feature_analyser->findEllipsePoints();
            if (!features_found) {
                continue;
            }

            cv::Point2d pupil = feature_analyser->getPupilUndistorted();
            cv::RotatedRect ellipse = feature_analyser->getEllipseUndistorted();
            ellipse.angle = 0.0; // Required due to the problems with angle estimation.

            all_front_corners.push_back(front_corners);
            all_back_corners.push_back(back_corners);
            all_marker_positions.push_back({
                                                   csv_file[current_row][25], csv_file[current_row][26],
                                                   csv_file[current_row][27]
                                           });
            camera_pupils.push_back(pupil);
            camera_ellipses.push_back(ellipse);

            auto glints = feature_analyser->getGlints();
            cv::Point2d top_left_side_glint = glints->at(0);
            cv::Point2d bottom_right_side_glint = glints->at(0);
            cv::Point2d mean_glint = {0.0, 0.0};
            for (const auto &glint: *glints) {
                mean_glint += glint;
            }
            mean_glint.x /= glints->size();
            mean_glint.y /= glints->size();
            for (const auto &glint: *glints) {
                if (glint.x < mean_glint.x && glint.y < mean_glint.y) {
                    top_left_side_glint = glint;
                }
                if (glint.x > mean_glint.x && glint.y > mean_glint.y) {
                    bottom_right_side_glint = glint;
                }
            }

            cv::Point3d real_nodal_point = {
                    ground_truth_test[current_row][1], ground_truth_test[current_row][2],
                    ground_truth_test[current_row][3]
            };
            cv::Point3d real_eye_centre = {
                    ground_truth_test[current_row][4], ground_truth_test[current_row][5],
                    ground_truth_test[current_row][6]
            };

            EyeInfo eye_info_org = {.pupil = pupil, .glints = {top_left_side_glint, bottom_right_side_glint},};


            // model_eye_estimator->invertDetectEye(eye_info_org, real_nodal_point, real_eye_centre);

            std::default_random_engine generator;

            double alpha{0};
            double beta{0};

            for (int i = 0; i < 1; i++) {
                EyeInfo eye_info = eye_info_org;

                std::vector<double> errors = {};
                for (int j = 0; j < 6; j++) {
                    errors.push_back(dis_pixel_error(generator));
                }

                eye_info.pupil.x += errors[0];
                eye_info.pupil.y += errors[1];
                eye_info.glints[0].x += errors[2];
                eye_info.glints[0].y += errors[3];
                eye_info.glints[1].x += errors[4];
                eye_info.glints[1].y += errors[5];

                std::string num_string = std::to_string(analyzed_frame_.frame_num);
                while (num_string.length() < 10) {
                    num_string = "0" + num_string;
                }

                // Load image
                auto image_lights_off = cv::imread(images_folder_path + "image_" + num_string + "_lights_off.jpg");
                auto image_lights_on = cv::imread(images_folder_path + "image_" + num_string + "_lights_on.jpg");

                // Mark pupil
                cv::circle(image_lights_off, pupil, 1, cv::Scalar(0, 0, 255), -1);

                // Mark glints
                cv::circle(image_lights_on, top_left_side_glint, 1, cv::Scalar(0, 0, 255), -1);
                cv::circle(image_lights_on, bottom_right_side_glint, 1, cv::Scalar(0, 0, 255), -1);

                // Mark pupil
                cv::circle(image_lights_off, eye_info.pupil, 1, cv::Scalar(0, 255, 0), -1);

                // Mark glints
                cv::circle(image_lights_on, eye_info.glints[0], 1, cv::Scalar(0, 255, 0), -1);
                cv::circle(image_lights_on, eye_info.glints[1], 1, cv::Scalar(0, 255, 0), -1);

                // Write images
                cv::imwrite(images_folder_path + "image_" + num_string + "_lights_off_mod.jpg", image_lights_off);
                cv::imwrite(images_folder_path + "image_" + num_string + "_lights_on_mod.jpg", image_lights_on);


                cv::Point3d nodal_point, eye_centre, visual_axis;
                model_eye_estimator->detectEye(eye_info, nodal_point, eye_centre, visual_axis);

                ////                cv::Vec3d calculated_optical_axis = nodal_point - grid_eye_centre;
                cv::Vec3d calculated_optical_axis = nodal_point - eye_centre;
                calculated_optical_axis = calculated_optical_axis / cv::norm(calculated_optical_axis);

                cv::Vec3d nodal_point_v = nodal_point;

                cv::Vec3d calculated_visual_axis = all_marker_positions.back() - nodal_point_v;
                calculated_visual_axis = calculated_visual_axis / cv::norm(calculated_visual_axis);


                Utils::getAnglesBetweenVectors(calculated_optical_axis, calculated_visual_axis, alpha, beta);

                std::cout << eye_centre << " " << nodal_point << std::endl;
                //
                //                for (int j = 0; j < 6; j++) {
                //                    std::cout << errors[j] << " ";
                //                }
                //
                //                std::cout << alpha << " " << beta << std::endl;
            }

            alphas.push_back(alpha);
            betas.push_back(beta);
            //            break;
        }

        double mean_alpha = std::accumulate(alphas.begin(), alphas.end(), 0.0) / alphas.size();
        double mean_beta = std::accumulate(betas.begin(), betas.end(), 0.0) / betas.size();

        double std_alpha = 0.0;
        double std_beta = 0.0;
        for (int i = 0; i < alphas.size(); i++) {
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
        cv::Mat camera_features(frames * (samples_per_ellipse + 1), 4, CV_64F);
        cv::Mat blender_features(frames * (samples_per_ellipse + 1), 4, CV_64F);
        for (int i = 0; i < frames; i++) {
            camera_features.at<double>(i * (samples_per_ellipse + 1), 0) = camera_pupils[i].x;
            camera_features.at<double>(i * (samples_per_ellipse + 1), 1) = camera_pupils[i].y;
            camera_features.at<double>(i * (samples_per_ellipse + 1), 2) = 0.0;
            camera_features.at<double>(i * (samples_per_ellipse + 1), 3) = 1.0;
            for (int j = 0; j < samples_per_ellipse; j++) {
                // Get angle on ellipse between 0 and 2pi
                double angle = (double) j / samples_per_ellipse * 2 * CV_PI;

                // Find intersection between ellipse and line with angle
                cv::Point2d intersection = Utils::findEllipseIntersection(camera_ellipses[i], angle);
                camera_features.at<double>(i * (samples_per_ellipse + 1) + j + 1, 0) = intersection.x;
                camera_features.at<double>(i * (samples_per_ellipse + 1) + j + 1, 1) = intersection.y;
                camera_features.at<double>(i * (samples_per_ellipse + 1) + j + 1, 2) = 0.0;
                camera_features.at<double>(i * (samples_per_ellipse + 1) + j + 1, 3) = 1.0;
            }
        }

        double smallest_error = std::numeric_limits<double>::max();
        int best_polynomial_num = 0;

        int last_id = all_front_corners.size() - 1;
        SetupVariables &setup_variables = Settings::parameters.user_polynomial_params[camera_id_]->setup_variables;

        cv::Vec3d starting_point = visual_angles_optimizer_->getCrossPoint();

        cv::Mat x = (cv::Mat_<double>(1, 3) << starting_point[0], starting_point[1], starting_point[2]);
        visual_angles_optimizer_->calc(x.ptr<double>());
        cv::Mat step = cv::Mat::ones(x.rows, x.cols, CV_64F) * 0.1;
        visual_angles_solver_->setInitStep(step);
        visual_angles_solver_->minimize(x);

        cv::Point3d eye_centre = cv::Point3d(x.at<double>(0, 0), x.at<double>(0, 1), x.at<double>(0, 2));

        Framework::mutex.unlock();
        return eye_centre;
    }
} // et
