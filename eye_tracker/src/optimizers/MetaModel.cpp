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
                                         std::shared_ptr<FeatureAnalyser> feature_analyser, std::string path_to_csv,
                                         std::string user_id)
    {
        Framework::mutex.lock();

        // Open CSV for reading
        auto csv_file = Utils::readFloatRowsCsv(path_to_csv);
        int current_row = 0;

        std::vector<cv::Point3d> all_marker_positions{};
        std::vector<cv::Point2d> camera_pupils{};
        std::vector<cv::RotatedRect> camera_ellipses{};
        std::vector<cv::Point3d> nodal_points{};
        std::vector<cv::Point3d> eye_centres{};

        auto user_profile = &Settings::parameters.polynomial_params[camera_id_][user_id];
        double alpha = user_profile->setup_variables.alpha;
        double beta = user_profile->setup_variables.beta;

        std::vector<cv::Vec3d> optical_axes{};
        std::vector<cv::Vec3d> visual_axes{};

        std::vector<cv::Vec3d> estimated_visual_axes{};
        std::vector<cv::Point3d> estimated_nodal_points{};
        std::vector<cv::Point3d> estimated_eye_centres{};

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

        eye_centre_optimizer_->setParameters(cross_point, visual_axes, optical_axes,
                                             user_profile->setup_variables.cornea_centre_distance);
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

            cv::Point3d visual_axis = all_marker_positions.back() - eye_centre;
            visual_axis = visual_axis / cv::norm(visual_axis);

            cv::Point3d optical_axis = Utils::visualToOpticalAxis(visual_axis, alpha, beta);
            cv::Point3d nodal_point = eye_centre + optical_axis * user_profile->setup_variables.cornea_centre_distance;
            nodal_points.push_back(nodal_point);
            eye_centres.push_back(eye_centre);

            EyeInfo eye_info = {
                    .pupil = pupil,
                    .ellipse = ellipse,
            };

            cv::Point3d estimated_nodal_point, estimated_eye_centre, estimated_visual_axis;
            polynomial_estimator->detectEye(eye_info, estimated_nodal_point, estimated_eye_centre,
                                            estimated_visual_axis, alpha, beta);
            estimated_nodal_points.push_back(estimated_nodal_point);
            estimated_visual_axes.push_back(estimated_visual_axis);
            estimated_eye_centres.push_back(estimated_eye_centre);
        }

        double train_fraction = 1.0;

        // Show current error
        double error = 0.0;

        for (int i = estimated_nodal_points.size() * train_fraction; i < estimated_nodal_points.size(); i++) {
            cv::Point3d estimated_nodal_point = estimated_nodal_points[i];
            cv::Point3d nodal_point = nodal_points[i];
            error += cv::norm(estimated_nodal_point - nodal_point);
        }
        error /= estimated_nodal_points.size() * (1 - train_fraction);
        std::cout << "Nodal point" << std::endl;
        std::cout << "Old error: " << error << std::endl;

        cv::Point3d offset = {0.0, 0.0, 0.0};
        for (int i = 0; i < estimated_nodal_points.size() * train_fraction; i++) {
            offset += nodal_points[i] - estimated_nodal_points[i];
        }
        offset /= estimated_nodal_points.size() * train_fraction;

        user_profile->camera_to_blender = Utils::getTransformationBetweenMatrices(
                std::vector<cv::Point3d>(estimated_nodal_points.begin(),
                                         estimated_nodal_points.begin() +
                                         estimated_nodal_points.size() * train_fraction),
                std::vector<cv::Point3d>(nodal_points.begin(),
                                         nodal_points.begin() + nodal_points.size() * train_fraction));
        std::cout << user_profile->camera_to_blender << std::endl;

        cv::Mat new_nodal_point_estimations = Utils::convertToHomogeneous(cv::Mat(estimated_nodal_points).reshape(1)) *
                                              user_profile->camera_to_blender;
        for (int i = 0; i < estimated_nodal_points.size(); i++) {
//            estimated_nodal_points[i] = cv::Point3d(new_nodal_point_estimations.at<double>(i, 0),
//                                                    new_nodal_point_estimations.at<double>(i, 1),
//                                                    new_nodal_point_estimations.at<double>(i, 2));
            estimated_nodal_points[i] = estimated_nodal_points[i] + offset;
        }


        error = 0.0;
        for (int i = estimated_nodal_points.size() * train_fraction; i < estimated_nodal_points.size(); i++) {
            error += cv::norm(estimated_nodal_points[i] - nodal_points[i]);
        }
        error /= estimated_nodal_points.size() * (1 - train_fraction);
        std::cout << "New error: " << error << std::endl;

        offset = {0.0, 0.0, 0.0};
        for (int i = 0; i < estimated_eye_centres.size() * train_fraction; i++) {
            offset += eye_centres[i] - estimated_eye_centres[i];
        }
        offset /= estimated_eye_centres.size() * train_fraction;

        cv::Mat new_eye_centre_estimations = Utils::convertToHomogeneous(cv::Mat(estimated_eye_centres).reshape(1)) *
                                             user_profile->camera_to_blender;
        for (int i = 0; i < estimated_eye_centres.size(); i++) {
//            estimated_eye_centres[i] = cv::Point3d(new_eye_centre_estimations.at<double>(i, 0),
//                                                   new_eye_centre_estimations.at<double>(i, 1),
//                                                   new_eye_centre_estimations.at<double>(i, 2));
            estimated_eye_centres[i] = estimated_eye_centres[i] + offset;
        }
        std::vector<cv::Vec3d> real_visual_axes{};
        for (int i = 0; i < estimated_visual_axes.size(); i++) {
            cv::Vec3d estimated_visual_axis = estimated_nodal_points[i] - estimated_eye_centres[i];
            estimated_visual_axes[i] = estimated_visual_axis / cv::norm(estimated_visual_axis);
            estimated_visual_axes[i] = Utils::opticalToVisualAxis(estimated_visual_axes[i], alpha, beta);
            cv::Vec3d real_visual_axis = all_marker_positions[i] - nodal_points[i];
            real_visual_axis = real_visual_axis / cv::norm(real_visual_axis);
            real_visual_axes.push_back(real_visual_axis);
        }

        error = 0.0;
        for (int i = estimated_visual_axes.size() * train_fraction; i < estimated_visual_axes.size(); i++) {
            cv::Vec3d estimated_visual_axis = estimated_visual_axes[i];
            cv::Vec3d real_visual_axis = real_visual_axes[i];
            error += cv::norm(estimated_visual_axis - real_visual_axis);
        }

        error /= estimated_visual_axes.size() * (1 - train_fraction);
        std::cout << "Visual axis" << std::endl;
        std::cout << "Old error: " << error << std::endl;

        cv::Mat transformation = Utils::getTransformationBetweenMatrices(
                std::vector<cv::Point3d>(estimated_visual_axes.begin(),
                                         estimated_visual_axes.begin() + estimated_visual_axes.size() * train_fraction),
                std::vector<cv::Point3d>(real_visual_axes.begin(),
                                         real_visual_axes.begin() + real_visual_axes.size() * train_fraction));
        std::cout << transformation << std::endl;

        cv::Mat new_visual_axis_estimations = Utils::convertToHomogeneous(cv::Mat(estimated_visual_axes).reshape(1)) *
                                      transformation;
        for (int i = 0; i < estimated_eye_centres.size(); i++) {
//            estimated_visual_axes[i] = cv::Point3d(new_visual_axis_estimations.at<double>(i, 0),
//                                                   new_visual_axis_estimations.at<double>(i, 1),
//                                                   new_visual_axis_estimations.at<double>(i, 2));
            estimated_visual_axes[i] = estimated_nodal_points[i] - estimated_eye_centres[i];
            estimated_visual_axes[i] = estimated_visual_axes[i] / cv::norm(estimated_visual_axes[i]);
            estimated_visual_axes[i] = Utils::opticalToVisualAxis(estimated_visual_axes[i], alpha, beta);
        }
        error = 0.0;
        for (int i = estimated_visual_axes.size() * train_fraction; i < estimated_visual_axes.size(); i++) {
            cv::Vec3d estimated_visual_axis = estimated_visual_axes[i];
            estimated_visual_axis = estimated_visual_axis / cv::norm(estimated_visual_axis);
            cv::Vec3d real_visual_axis = real_visual_axes[i];
            error += cv::norm(estimated_visual_axis - real_visual_axis);
        }
        error /= estimated_visual_axes.size() * (1 - train_fraction);
        std::cout << "New error: " << error << std::endl;

        // Calculate horizontal and vertical angle error between visual axes
        std::vector<double> horizontal_angles{};
        std::vector<double> vertical_angles{};
        for (int i = estimated_visual_axes.size() * train_fraction; i < estimated_visual_axes.size(); i++) {
            cv::Vec3d estimated_visual_axis = estimated_visual_axes[i];
            estimated_visual_axis = estimated_visual_axis / cv::norm(estimated_visual_axis);
            double k = (all_marker_positions[i].z - estimated_nodal_points[i].z) / estimated_visual_axis[2];
            cv::Point3d estimated_marker_point =
                    estimated_nodal_points[i] + k * (cv::Point3d) (estimated_visual_axes[i]);
            cv::Vec3d estimated_marker_eye_dir = estimated_marker_point - nodal_points[i];
            estimated_marker_eye_dir = estimated_marker_eye_dir / cv::norm(estimated_marker_eye_dir);
            cv::Vec3d real_visual_axis = real_visual_axes[i];
            double horizontal_angle{}, vertical_angle{};
            Utils::getAnglesBetweenVectors(estimated_marker_eye_dir, real_visual_axis, horizontal_angle,
                                           vertical_angle);
            horizontal_angles.push_back(std::abs(horizontal_angle));
            vertical_angles.push_back(std::abs(vertical_angle));
        }

        std::cout << "Mean horizontal angle error: "
                  << std::accumulate(horizontal_angles.begin(), horizontal_angles.end(), 0.0) / horizontal_angles.size()
                  << std::endl;
        std::cout << "Mean vertical angle error: "
                  << std::accumulate(vertical_angles.begin(), vertical_angles.end(), 0.0) / vertical_angles.size()
                  << std::endl;

        Framework::mutex.unlock();
        return eye_centre;
    }
} // et
