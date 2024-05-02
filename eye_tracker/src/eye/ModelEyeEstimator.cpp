#include "eye_tracker/eye/ModelEyeEstimator.hpp"
#include "eye_tracker/Utils.hpp"
#include "eye_tracker/optimizers/GlintPositionOptimizer.hpp"
#include "eye_tracker/optimizers/PupilPositionOptimizer.hpp"

#include <opencv2/core/core.hpp>

namespace et {
    ModelEyeEstimator::ModelEyeEstimator(int camera_id) : EyeEstimator(camera_id) {
        // Create a minimizer for used for finding cornea centre.
        nodal_point_optimizer_ = new NodalPointOptimizer(camera_id_);
        nodal_point_optimizer_->initialize();
        minimizer_function_ = cv::Ptr<cv::DownhillSolver::Function>{nodal_point_optimizer_};
        solver_ = cv::DownhillSolver::create();
        solver_->setTermCriteria(
                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 1000, std::numeric_limits<float>::min()));
        solver_->setFunction(minimizer_function_);
        cv::Mat step = (cv::Mat_<double>(1, 2) << 0.1, 0.1);
        solver_->setInitStep(step);
    }

    bool ModelEyeEstimator::detectEye(EyeInfo& eye_info, cv::Point3d& eye_centre, cv::Point3d& nodal_point, cv::Vec2d& angles) {
        cv::Vec3d pupil_position{ICStoCCS(eye_info.pupil)};

        std::vector<cv::Vec3d> glint_positions{};

        int leds_num = Settings::parameters.leds_positions[camera_id_].size();
        std::vector<cv::Vec3d> leds{};

        cv::Mat camera_pos_mat = Settings::parameters.camera_params[camera_id_].extrinsic_matrix.t();

//        leds = {
//                leds[0],
//                leds[13]
//        };
//        leds_num = 2;
//        eye_info.glints = {
//                eye_info.glints[0],
//                eye_info.glints[13]
//        };


        // calculate planes with glints, LEDs, eye's nodal point, and camera's
        // nodal point.
        std::vector<cv::Vec3d> v1v2s{};
        for (int i = 0; i < leds_num; i++) {
            if (eye_info.glints_validity[i]) {
                cv::Vec3d led = Settings::parameters.leds_positions[camera_id_][i];
                cv::Vec4d led_homo;
                led_homo[0] = led[0];
                led_homo[1] = led[1];
                led_homo[2] = led[2];
                led_homo[3] = 1.0;
                cv::Mat converted_leds = cv::Mat(led_homo).t() * camera_pos_mat;
                leds.push_back({converted_leds.at<double>(0), converted_leds.at<double>(1), converted_leds.at<double>(2)});

                cv::Vec3d v1{leds.back()};
                cv::normalize(v1, v1);
                cv::Vec3d v2{ICStoCCS(eye_info.glints[i])};
                glint_positions.push_back(v2);
                cv::normalize(v2, v2);
                cv::Vec3d v1v2{v1.cross(v2)};
                cv::normalize(v1v2, v1v2);
                v1v2s.push_back(v1v2);
            }
        }

        // Find the intersection of all planes which is a vector between camera's
        // nodal point and eye's nodal point.
        std::vector<cv::Vec3d> np2c_dirs{};
        int counter{0};
        for (int i = 0; i < v1v2s.size(); i++) {
            for (int j = i + 1; j < v1v2s.size(); j++) {
                double angle = Utils::getAngleBetweenVectors(v1v2s[i], v1v2s[j]) * 180 / M_PI;
                if (angle < 0) {
                    angle += 180;
                }
                if (angle > 90) {
                    angle = 180 - angle;
                }
                if (angle < 45) {
                    continue;
                }

                cv::Vec3d np2c_dir{v1v2s[i].cross(v1v2s[j])};
                cv::normalize(np2c_dir, np2c_dir);
                if (np2c_dir(2) < 0) {
                    np2c_dir = -np2c_dir;
                }
                // If NaN
                if (np2c_dir != np2c_dir) {
                    continue;
                }
                np2c_dirs.push_back(np2c_dir);
                counter++;
            }
        }

        if (counter == 0) {
            return false;
        }

        cv::Vec3d avg_np2c_dir = Utils::getMedian(np2c_dirs);
        avg_np2c_dir = cv::normalize(avg_np2c_dir);

        if (nodal_point_optimizer_) {
            nodal_point_optimizer_->setParameters(avg_np2c_dir, glint_positions.data(), leds, camera_nodal_point_,
                                                  eye_measurements.cornea_curvature_radius);
        }
        cv::Mat x = (cv::Mat_<double>(1, 2) << 300, 300);
        // Finds the best candidate for cornea centre.
        solver_->minimize(x);
        double k = x.at<double>(0, 0);
        cv::Vec3d nodal_point_vec = camera_nodal_point_ + avg_np2c_dir * k;
        nodal_point = nodal_point_vec;

        cv::Vec3d pupil = calculatePositionOnPupil(pupil_position, nodal_point_vec);

        cv::Vec3d eye_centre_vec{};
        if (pupil != cv::Vec3d()) {
            cv::Vec3d pupil_direction{nodal_point_vec - pupil};
            cv::normalize(pupil_direction, pupil_direction);
            // Eye centre lies in the same vector as cornea centre and pupil centre.

            eye_centre_vec = nodal_point_vec + eye_measurements.cornea_centre_distance * pupil_direction;
        } else {
            eye_centre_vec = nodal_point_vec + eye_measurements.cornea_centre_distance * cv::Vec3d{0, 0, -1};
        }
        eye_centre = eye_centre_vec;

        eye_centre = CCStoWCS(eye_centre);

        nodal_point = CCStoWCS(nodal_point);

        cv::Point3d optical_axis_p = nodal_point - eye_centre;
        cv::Vec3d optical_axis = optical_axis_p;
        cv::normalize(optical_axis, optical_axis);
        cv::Point3d visual_axis = Utils::opticalToVisualAxis(optical_axis, eye_measurements.alpha,
                                                             eye_measurements.beta);
        Utils::vectorToAngles(visual_axis, angles);

        if (eye_centre_vec == cv::Vec3d()) {
            return false;
        }
        return true;
    }

    ModelEyeEstimator::~ModelEyeEstimator() {
        if (!solver_->empty()) {
            solver_.release();
        }
        if (!minimizer_function_.empty()) {
            minimizer_function_.release();
        }
    }

    bool ModelEyeEstimator::invertDetectEye(EyeInfo& eye_info, const cv::Point3d& nodal_point,
                                            const cv::Point3d& eye_centre,
                                            const EyeMeasurements& measurements) {
        cv::Vec3d optical_axis = nodal_point - eye_centre;
        cv::normalize(optical_axis, optical_axis);

        cv::Point3d pupil_position = static_cast<cv::Vec3d>(nodal_point) + measurements.pupil_cornea_distance *
                                                                           optical_axis;
        eye_info.pupil = WCStoICS(pupil_position);

        eye_info.glints.clear();

        static PupilPositionOptimizer* pupil_position_optimizer;
        static cv::Ptr<cv::DownhillSolver::Function> pupil_minimizer_function{};
        static cv::Ptr<cv::DownhillSolver> pupil_solver{};

        static bool initialized = false;

        if (!initialized) {
            pupil_position_optimizer = new PupilPositionOptimizer();
            pupil_minimizer_function = cv::Ptr<cv::DownhillSolver::Function>{pupil_position_optimizer};
            pupil_solver = cv::DownhillSolver::create();
            pupil_solver->setFunction(pupil_minimizer_function);
            const cv::Mat step = (cv::Mat_<double>(1, 2) << 0.5, 0.5);
            pupil_solver->setInitStep(step);
            pupil_solver->setTermCriteria(cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 1000, std::numeric_limits<float>::min()));

            initialized = true;
        }

        cv::Point3d wcs_camera_nodal_point = CCStoWCS(camera_nodal_point_);

        auto leds = Settings::parameters.leds_positions[camera_id_];
        cv::Mat x = (cv::Mat_<double>(1, 2) << 0.5, 0.5);
        for (int i = 0; i < leds.size(); i++) {
            cv::Vec3d glint_position = Utils::getReflectionPoint(leds[i], nodal_point, wcs_camera_nodal_point,
                                                                 measurements.cornea_curvature_radius);

            eye_info.glints.push_back(WCStoICS(glint_position));
        }

        pupil_position_optimizer->setParameters(nodal_point, pupil_position, CCStoWCS(camera_nodal_point_),
                                                measurements);
        // x = (cv::Mat_<double>(1, 3) << 1.0, 1.0, 1.0);
        // pupil_solver->minimize(x);
        // cv::Vec3d ray_direction;
        // cv::multiply(-static_cast<cv::Vec3d>(pupil_position - wcs_camera_nodal_point),
        //              cv::Vec3d(x.at<double>(0), x.at<double>(1), x.at<double>(2)), ray_direction);
        // cv::normalize(ray_direction, ray_direction);
        //
        // double t{};
        // Utils::getRaySphereIntersection(wcs_camera_nodal_point, ray_direction, cornea_position,
        //                                 measurements.cornea_curvature_radius, t);
        //
        // pupil_position = static_cast<cv::Vec3d>(wcs_camera_nodal_point) + t * ray_direction;
        // std::cout << "Pupil position: " << pupil_position << std::endl;
        //
        // eye_info.pupil = WCStoICS(pupil_position);

        x = (cv::Mat_<double>(1, 2) << 0.0, 0.0);
        pupil_solver->minimize(x);
        pupil_position.x = nodal_point.x - measurements.cornea_curvature_radius * sin(x.at<double>(0)) * cos(x.at<double>(1));
        pupil_position.y = nodal_point.y + measurements.cornea_curvature_radius * sin(x.at<double>(1));
        pupil_position.z = nodal_point.z - measurements.cornea_curvature_radius * cos(x.at<double>(0)) * cos(x.at<double>(1));
        eye_info.pupil = WCStoICS(pupil_position);
        // Expected: [168.717, 96.1413, 784.517]

        return true;
    }
} // et
