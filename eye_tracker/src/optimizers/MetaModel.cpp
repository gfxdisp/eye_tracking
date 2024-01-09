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

        auto user_profile = &Settings::parameters.polynomial_params[camera_id_][user_id];
        double alpha = user_profile->setup_variables.alpha;
        double beta = user_profile->setup_variables.beta;

        std::vector<cv::Point3d> eye_centres{};
        std::vector<cv::Vec2d> angles{};

        std::vector<cv::Point3d> estimated_eye_centres{};
        std::vector<cv::Vec2d> estimated_angles{};

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
        polynomial_estimator->setModel("default");

        et::Settings::parameters.user_params[camera_id_]->eye_centre_offset = {0.0, 0.0, 0.0};
        et::Settings::parameters.user_params[camera_id_]->angles_offset = {0.0, 0.0};

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

            if (csv_file[current_row][28] < 1.0) {
                continue;
            }

            all_marker_positions.push_back({
                                                   csv_file[current_row][25], csv_file[current_row][26],
                                                   csv_file[current_row][27]
                                           });
            camera_pupils.push_back(pupil);
            camera_ellipses.push_back(ellipse);

            cv::Point3d gaze_direction = all_marker_positions.back() - eye_centre;
            gaze_direction = gaze_direction / cv::norm(gaze_direction);

            cv::Vec2d angle{};
            Utils::vectorToAngles(gaze_direction, angle);
            eye_centres.push_back(eye_centre);
            angles.push_back(angle);

            EyeInfo eye_info = {
                    .pupil = pupil,
                    .ellipse = ellipse,
            };

            cv::Point3d estimated_eye_centre;
            cv::Vec2d estimated_angle{};
            polynomial_estimator->detectEye(eye_info, estimated_eye_centre, estimated_angle);
            estimated_eye_centres.push_back(estimated_eye_centre);
            estimated_angles.push_back(estimated_angle);
        }

        cv::Point3d eye_centre_offset{};
        cv::Vec2d angle_offset{};

        for (int i = 0; i < estimated_eye_centres.size(); i++) {
            eye_centre_offset += estimated_eye_centres[i] - eye_centres[i];
            angle_offset += estimated_angles[i] - angles[i];
        }
        eye_centre_offset.x /= (int)(estimated_eye_centres.size());
        eye_centre_offset.y /= (int)(estimated_eye_centres.size());
        eye_centre_offset.z /= (int)(estimated_eye_centres.size());
        angle_offset[0] /= (int)(estimated_eye_centres.size());
        angle_offset[1] /= (int)(estimated_eye_centres.size());
        std::cout << "Eye centre offset: " << eye_centre_offset << "\n";

        std::vector<double> errors_x{};
        std::vector<double> errors_y{};
        std::vector<double> errors_z{};
        std::vector<double> errors_theta{};
        std::vector<double> errors_phi{};
        for (int i = 0; i < estimated_eye_centres.size(); i++) {
            errors_x.push_back(cv::norm(estimated_eye_centres[i].x - eye_centres[i].x - eye_centre_offset.x));
            errors_y.push_back(cv::norm(estimated_eye_centres[i].y - eye_centres[i].y - eye_centre_offset.y));
            errors_z.push_back(cv::norm(estimated_eye_centres[i].z - eye_centres[i].z - eye_centre_offset.z));
        }

        std::cout << "Mean error X: " << std::accumulate(errors_x.begin(), errors_x.end(), 0.0) / errors_x.size()
        << " ± " << Utils::getStdDev(errors_x) << "\n";
        std::cout << "Mean error Y: " << std::accumulate(errors_y.begin(), errors_y.end(), 0.0) / errors_y.size()
                  << " ± " << Utils::getStdDev(errors_y) << "\n";
        std::cout << "Mean error Z: " << std::accumulate(errors_z.begin(), errors_z.end(), 0.0) / errors_z.size()
                  << " ± " << Utils::getStdDev(errors_z) << "\n";

        std::cout << "Angles offset: " << angle_offset * 180 / CV_PI << "\n";
        for (int i = 0; i < estimated_angles.size(); i++) {
            errors_theta.push_back(std::abs(estimated_angles[i][0] - angles[i][0] - angle_offset[0]));
            errors_phi.push_back(std::abs(estimated_angles[i][1] - angles[i][1] - angle_offset[1]));
        }

        std::cout << "Error theta: " << std::accumulate(errors_theta.begin(), errors_theta.end(), 0.0) / errors_theta.size() * 180 / CV_PI
        << " ± " << Utils::getStdDev(errors_theta) * 180 / CV_PI << "\n";

        std::cout << "Error phi: " << std::accumulate(errors_phi.begin(), errors_phi.end(), 0.0) / errors_phi.size() * 180 / CV_PI
                  << " ± " << Utils::getStdDev(errors_phi) * 180 / CV_PI << std::endl;

        et::Settings::parameters.user_params[camera_id_]->eye_centre_offset = eye_centre_offset;
        et::Settings::parameters.user_params[camera_id_]->angles_offset = angle_offset;
        et::Settings::saveSettings();

        Framework::mutex.unlock();
        return eye_centre;
    }
} // et
