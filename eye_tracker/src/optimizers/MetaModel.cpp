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

        eye_centre_optimizer_ = new EyeCentreOptimizer();
        eye_centre_minimizer_function_ = cv::Ptr<cv::DownhillSolver::Function>(eye_centre_optimizer_);
        eye_centre_solver_ = cv::DownhillSolver::create();
        eye_centre_solver_->setFunction(eye_centre_minimizer_function_);
    }

    cv::Point3d MetaModel::findMetaModel(std::shared_ptr<ImageProvider> image_provider,
                                         std::shared_ptr<FeatureAnalyser> feature_analyser, std::string path_to_csv, std::string user_id)
    {
        Framework::mutex.lock();

        // Open CSV for reading
        auto csv_file = Utils::readFloatRowsCsv(path_to_csv);
        int current_row = 0;

        std::vector<cv::Point3d> all_marker_positions{};
        std::vector<cv::Point2d> camera_pupils{};
        std::vector<cv::RotatedRect> camera_ellipses{};
        std::vector<cv::Point3d> nodal_points{};

        auto user_profile = &Settings::parameters.polynomial_params[camera_id_][user_id];
        double alpha = user_profile->setup_variables.alpha;
        double beta = user_profile->setup_variables.beta;

        std::vector<cv::Vec3d> optical_axes{};
        std::vector<cv::Vec3d> visual_axes{};

        std::vector<cv::Point3d> front_corners{};
        std::vector<cv::Point3d> back_corners{};
        for (int i = 0; i < 4; i++) {
            front_corners.push_back({csv_file[0][1 + i * 3], csv_file[0][2 + i * 3], csv_file[0][3 + i * 3]});
        }
        for (int i = 0; i < 4; i++) {
            back_corners.push_back({csv_file[0][13 + i * 3], csv_file[0][14 + i * 3], csv_file[0][15 + i * 3]});
        }

        for (int i = 0; i < 4; i++) {
            cv::Vec3d visual_axis = back_corners[i] - front_corners[i];
            visual_axis = visual_axis / cv::norm(visual_axis);
            visual_axes.push_back(visual_axis);
            cv::Vec3d optical_axis = Utils::visualToOpticalAxis(visual_axis, alpha, beta);
            optical_axes.push_back(optical_axis);
        }

        cv::Point3d cross_point = Utils::findGridIntersection(front_corners, back_corners);
        auto model_eye_estimator = std::make_shared<ModelEyeEstimator>(camera_id_);
        auto polynomial_estimator = std::make_shared<PolynomialEyeEstimator>(camera_id_);
        polynomial_estimator->setModel(user_id);

        eye_centre_optimizer_->setParameters(cross_point, visual_axes, optical_axes, user_profile->setup_variables.cornea_centre_distance);
        cv::Mat x = (cv::Mat_<double>(1, 3) << cross_point.x, cross_point.y, cross_point.z);
        eye_centre_solver_->setInitStep(cv::Mat::ones(x.rows, x.cols, CV_64F) * 0.1);
        eye_centre_solver_->minimize(x);

        cv::Point3d eye_centre = cv::Point3d(x.at<double>(0, 0), x.at<double>(0, 1), x.at<double>(0, 2));

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

            all_marker_positions.push_back({
                                                   csv_file[current_row][25], csv_file[current_row][26],
                                                   csv_file[current_row][27]
                                           });
            camera_pupils.push_back(pupil);
            camera_ellipses.push_back(ellipse);

            cv::Vec3d visual_axis = all_marker_positions.back() - eye_centre;
            visual_axis = visual_axis / cv::norm(visual_axis);

            cv::Point3d optical_axis = Utils::visualToOpticalAxis(visual_axis, alpha, beta);
            cv::Point3d nodal_point = eye_centre + optical_axis * user_profile->setup_variables.cornea_centre_distance;
            nodal_points.push_back(nodal_point);
        }

        int frames = nodal_points.size();
        int samples_per_ellipse = 20;
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

        EyeInfo eye_info{};
        for (int i = 0; i < nodal_points.size(); i++) {
            polynomial_estimator->invertEye(nodal_points[i], eye_centre, eye_info);
            blender_features.at<double>(i * (samples_per_ellipse + 1), 0) = eye_info.pupil.x;
            blender_features.at<double>(i * (samples_per_ellipse + 1), 1) = eye_info.pupil.y;
            blender_features.at<double>(i * (samples_per_ellipse + 1), 2) = 0.0;
            blender_features.at<double>(i * (samples_per_ellipse + 1), 3) = 1.0;
            for (int j = 0; j < samples_per_ellipse; j++) {
                // Get angle on ellipse between 0 and 2pi
                double angle = (double) j / samples_per_ellipse * 2 * CV_PI;

                // Find intersection between ellipse and line with angle
                cv::Point2d intersection = Utils::findEllipseIntersection(eye_info.ellipse, angle);
                blender_features.at<double>(i * (samples_per_ellipse + 1) + j + 1, 0) = intersection.x;
                blender_features.at<double>(i * (samples_per_ellipse + 1) + j + 1, 1) = intersection.y;
                blender_features.at<double>(i * (samples_per_ellipse + 1) + j + 1, 2) = 0.0;
                blender_features.at<double>(i * (samples_per_ellipse + 1) + j + 1, 3) = 1.0;
            }
        }

        user_profile->camera_to_blender = camera_features.inv(cv::DECOMP_SVD) * blender_features;
        std::cout << "Camera to blender matrix: " << std::endl << user_profile->camera_to_blender << std::endl;
        Framework::mutex.unlock();
        return eye_centre;
    }
} // et
