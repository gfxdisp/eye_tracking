#include <eye_tracker/eye/EyeEstimator.hpp>
#include <eye_tracker/Utils.hpp>
#include <eye_tracker/optimizers/FineTuner.hpp>

namespace et {
    EyeEstimator::EyeEstimator(const int camera_id) : camera_id_{camera_id} {
        features_params_ = Settings::parameters.user_params[camera_id];
        eye_params_ = Settings::parameters.eye_params[camera_id];

        intrinsic_matrix_ = &Settings::parameters.camera_params[camera_id].intrinsic_matrix;
        capture_offset_ = &Settings::parameters.camera_params[camera_id].capture_offset;
        dimensions_ = &Settings::parameters.camera_params[camera_id].dimensions;
        inv_extrinsic_matrix_ = Settings::parameters.camera_params[camera_id].extrinsic_matrix.inv();
        extrinsic_matrix_ = Settings::parameters.camera_params[camera_id].extrinsic_matrix;

        theta_polynomial_ = std::make_shared<Polynomial>(2, 2);
        phi_polynomial_ = std::make_shared<Polynomial>(2, 2);

        camera_nodal_position_ccs_ = {0, 0, 0};
        updateFineTuning();

        cornea_centre_position_ccs_optimizer_ = std::make_shared<NodalPointOptimizer>(camera_id_);
        cornea_centre_position_ccs_optimizer_->initialize();
    }

    bool EyeEstimator::detectEye(EyeInfo& eye_info, cv::Point3d& eye_centre_position_wcs, cv::Point3d& cornea_centre_position_wcs, cv::Vec2d& gaze_direction_angles_rad) const {
        cv::Vec3d pupil_centre_position_at_sensor_ccs = ICStoCCS(eye_info.pupil_centre_position_ics);

        std::vector<cv::Vec3d> glint_positions_at_sensor_ccs{};

        int leds_num = static_cast<int>(Settings::parameters.leds_positions[camera_id_].size());
        std::vector<cv::Vec3d> led_positions_ccs{};

        // calculate planes, each containing: a glints, an LED, eye's nodal point, and camera's nodal point.
        std::vector<cv::Vec3d> plane_normals_ccs{};
        for (int i = 0; i < leds_num; i++) {
            if (eye_info.glints_valid[i]) {
                auto const& led_position_wcs = Settings::parameters.leds_positions[camera_id_][i];
                cv::Vec4d led_position_wcs_homo;
                led_position_wcs_homo[0] = led_position_wcs[0];
                led_position_wcs_homo[1] = led_position_wcs[1];
                led_position_wcs_homo[2] = led_position_wcs[2];
                led_position_wcs_homo[3] = 1.0;
                cv::Mat led_position_ccs_homo = extrinsic_matrix_ * cv::Mat(led_position_wcs_homo);
                led_positions_ccs.push_back({led_position_ccs_homo.at<double>(0), led_position_ccs_homo.at<double>(1), led_position_ccs_homo.at<double>(2)});

                cv::Vec3d camera_led_vector_ccs = led_positions_ccs.back();
                cv::normalize(camera_led_vector_ccs, camera_led_vector_ccs);
                cv::Vec3d camera_glint_at_sensor_vector_ccs = ICStoCCS(eye_info.glint_positions_ics[i]);
                glint_positions_at_sensor_ccs.push_back(camera_glint_at_sensor_vector_ccs);
                cv::normalize(camera_glint_at_sensor_vector_ccs, camera_glint_at_sensor_vector_ccs);
                cv::Vec3d plane_normal = camera_led_vector_ccs.cross(camera_glint_at_sensor_vector_ccs);
                cv::normalize(plane_normal, plane_normal);
                plane_normals_ccs.push_back(plane_normal);
            }
        }

        // Find the intersection of all pairs of planes which is a vector between camera's nodal point and cornea centre.
        std::vector<cv::Vec3d> camera_cornea_directions_ccs{};
        for (int i = 0; i < plane_normals_ccs.size(); i++) {
            for (int j = i + 1; j < plane_normals_ccs.size(); j++) {
                double angle = Utils::getAngleBetweenVectors(plane_normals_ccs[i], plane_normals_ccs[j]) * 180 / M_PI;
                if (angle < 0) {
                    angle += 180;
                }
                if (angle > 90) {
                    angle = 180 - angle;
                }
                if (angle < 45) {
                    continue;
                }

                cv::Vec3d camera_cornea_candidate_direction = plane_normals_ccs[i].cross(plane_normals_ccs[j]);
                cv::normalize(camera_cornea_candidate_direction, camera_cornea_candidate_direction);
                if (camera_cornea_candidate_direction(2) < 0) {
                    camera_cornea_candidate_direction = -camera_cornea_candidate_direction;
                }

                // Check if NaN
                if (camera_cornea_candidate_direction != camera_cornea_candidate_direction) {
                    continue;
                }
                camera_cornea_directions_ccs.push_back(camera_cornea_candidate_direction);
            }
        }

        if (camera_cornea_directions_ccs.empty()) {
            return false;
        }

        cv::Vec3d camera_cornea_direction_ccs = Utils::getMedian(camera_cornea_directions_ccs);
        camera_cornea_direction_ccs = cv::normalize(camera_cornea_direction_ccs);

        cornea_centre_position_ccs_optimizer_->setParameters(camera_cornea_direction_ccs, glint_positions_at_sensor_ccs.data(), led_positions_ccs, camera_nodal_position_ccs_, eye_params_.cornea_curvature_radius);
        const double k = cornea_centre_position_ccs_optimizer_->goldenSectionSearch(eye_camera_distance_min_mm, eye_camera_distance_max_mm, 1e-7);

        cv::Vec3d cornea_centre_position_ccs = camera_nodal_position_ccs_ + camera_cornea_direction_ccs * k;

        cv::Vec3d pupil_centre_position_ccs = calculatePupilPositionCCS(pupil_centre_position_at_sensor_ccs, cornea_centre_position_ccs);

        cv::Vec3d eye_centre_position_ccs{};
        if (pupil_centre_position_ccs != cv::Vec3d()) {
            cv::Vec3d pupil_centre_direction_ccs = cornea_centre_position_ccs - pupil_centre_position_ccs;
            cv::normalize(pupil_centre_direction_ccs, pupil_centre_direction_ccs);
            // Eye centre lies in the same vector as cornea centre and pupil centre.

            eye_centre_position_ccs = cornea_centre_position_ccs + eye_params_.cornea_centre_distance * pupil_centre_direction_ccs;
        } else {
            eye_centre_position_ccs = cornea_centre_position_ccs + eye_params_.cornea_centre_distance * cv::Vec3d{0, 0, -1};
        }

        eye_centre_position_wcs = CCStoWCS(eye_centre_position_ccs);

        cornea_centre_position_wcs = CCStoWCS(cornea_centre_position_ccs);

        cv::Vec3d optical_axis_wcs = cornea_centre_position_wcs - eye_centre_position_wcs;
        cv::normalize(optical_axis_wcs, optical_axis_wcs);
        cv::Point3d visual_axis_wcs = Utils::opticalToVisualAxis(optical_axis_wcs, eye_params_.alpha, eye_params_.beta);
        Utils::vectorToAngles(visual_axis_wcs, gaze_direction_angles_rad);
        return true;
    }

    cv::Vec3d EyeEstimator::ICStoCCS(const cv::Point2d point) const {
        double z = -intrinsic_matrix_->at<double>(cv::Point(0, 0)) * 6.144 / dimensions_->width;
        const double shift_x = intrinsic_matrix_->at<double>(cv::Point(2, 0)) - dimensions_->width * 0.5;
        const double shift_y = intrinsic_matrix_->at<double>(cv::Point(2, 1)) - dimensions_->height * 0.5;
        double x = -(point.x - shift_x + capture_offset_->width - dimensions_->width * 0.5) / (dimensions_->width * 0.5) * 6.144 / 2;
        double y = -(point.y - shift_y + capture_offset_->height - dimensions_->height * 0.5) / (dimensions_->height * 0.5) * 4.915 / 2;

        return {x, y, z};
    }

    cv::Vec3d EyeEstimator::CCStoWCS(const cv::Vec3d& point) const {
        const cv::Vec4d homo_point{point[0], point[1], point[2], 1.0};
        cv::Mat world_pos = inv_extrinsic_matrix_ * cv::Mat(homo_point);
        double x = world_pos.at<double>(0) / world_pos.at<double>(3);
        double y = world_pos.at<double>(1) / world_pos.at<double>(3);
        double z = world_pos.at<double>(2) / world_pos.at<double>(3);

        return {x, y, z};
    }

    cv::Vec3d EyeEstimator::ICStoWCS(const cv::Point2d point) const {
        return CCStoWCS(ICStoCCS(point));
    }

    cv::Point2d EyeEstimator::CCStoICS(const cv::Point3d& point) const {
        const cv::Vec3d homo_point{point.x, point.y, point.z};
        cv::Mat ccs_pos = *intrinsic_matrix_ * cv::Mat(homo_point);
        const double x = ccs_pos.at<double>(0) / ccs_pos.at<double>(2);
        const double y = ccs_pos.at<double>(1) / ccs_pos.at<double>(2);
        return {x - capture_offset_->width, y - capture_offset_->height};
    }

    cv::Point2d EyeEstimator::WCStoICS(const cv::Point3d& point) const {
        return CCStoICS(WCStoCCS(point));
    }

    cv::Point3d EyeEstimator::WCStoCCS(const cv::Point3d& point) const {
        const cv::Vec4d homo_point{point.x, point.y, point.z, 1.0};
        cv::Mat ccs_pos = extrinsic_matrix_ * cv::Mat(homo_point);
        double x = ccs_pos.at<double>(0) / ccs_pos.at<double>(3);
        double y = ccs_pos.at<double>(1) / ccs_pos.at<double>(3);
        double z = ccs_pos.at<double>(2) / ccs_pos.at<double>(3);

        return {x, y, z};
    }

    cv::Vec3d EyeEstimator::calculatePupilPositionCCS(const cv::Vec3d& pupil_centre_position_at_sensor_ccs, const cv::Vec3d& cornea_centre_position_ccs) const {
        cv::Vec3d pupil_centre_position_ccs{};
        double t{};
        cv::Vec3d pupil_direction_ccs = -pupil_centre_position_at_sensor_ccs;
        cv::normalize(pupil_direction_ccs, pupil_direction_ccs);
        bool intersected = Utils::getRaySphereIntersection(cv::Vec3d(0.0), pupil_direction_ccs, cornea_centre_position_ccs, eye_params_.cornea_curvature_radius, t);

        if (intersected) {
            const cv::Vec3d pupil_on_cornea_ccs = t * pupil_direction_ccs;
            cv::Vec3d cornea_normal_ccs = pupil_on_cornea_ccs - cornea_centre_position_ccs;
            cv::normalize(cornea_normal_ccs, cornea_normal_ccs);
            const cv::Vec3d direction{Utils::getRefractedRay(pupil_direction_ccs, cornea_normal_ccs, eye_params_.cornea_refraction_index)};
            intersected = Utils::getRaySphereIntersection(pupil_on_cornea_ccs, direction, cornea_centre_position_ccs, eye_params_.pupil_cornea_distance, t);
            if (intersected) {
                pupil_centre_position_ccs = pupil_on_cornea_ccs + t * direction;
            }
        }
        return pupil_centre_position_ccs;
    }

    bool EyeEstimator::findPupilDiameter(const cv::Point2d pupil_centre_position_ics, const double pupil_radius_ics, const cv::Vec3d& cornea_centre_position_wcs, double& pupil_diameter_mm) const {
        const cv::Vec3d cornea_centre_position_ccs = WCStoCCS(cornea_centre_position_wcs);
        const cv::Vec3d pupil_centre_position_at_sensor_ccs = ICStoCCS(pupil_centre_position_ics);

        const cv::Vec3d pupil_right_position_at_sensor_ccs = ICStoCCS(pupil_centre_position_ics + cv::Point2d(pupil_radius_ics, 0.0));

        cv::Vec3d pupil_centre_position_ccs = calculatePupilPositionCCS(pupil_centre_position_at_sensor_ccs, cornea_centre_position_ccs);
        cv::Vec3d pupil_right_position_ccs = calculatePupilPositionCCS(pupil_right_position_at_sensor_ccs, cornea_centre_position_ccs);

        if (pupil_centre_position_ccs != cv::Vec3d() && pupil_right_position_ccs != cv::Vec3d()) {
            // We don't have good depth estimation, so we assume the pupil is perpendicular to the camera
            pupil_centre_position_ccs[2] = 0.0;
            pupil_right_position_ccs[2] = 0.0;
            pupil_diameter_mm = 2 * cv::norm(pupil_centre_position_ccs - pupil_right_position_ccs);
            return true;
        }

        return false;
    }

    void EyeEstimator::getEyeCentrePositionWCS(cv::Point3d& eye_centre_position_wcs) {
        mtx_eye_position_.lock();
        eye_centre_position_wcs = eye_centre_position_wcs_;
        mtx_eye_position_.unlock();
    }

    void EyeEstimator::getCorneaCentrePositionWCS(cv::Point3d& cornea_centre) {
        mtx_eye_position_.lock();
        cornea_centre = cornea_centre_position_wcs_;
        mtx_eye_position_.unlock();
    }

    void EyeEstimator::getPupilDiameter(double& pupil_diameter_mm) {
        mtx_eye_position_.lock();
        pupil_diameter_mm = pupil_diameter_mm_;
        mtx_eye_position_.unlock();
    }

    void EyeEstimator::getCorneaCentrePositionICS(cv::Point2d& cornea_centre_position_ics, const bool use_offset) {
        mtx_eye_position_.lock();
        if (use_offset) {
            cornea_centre_position_ics = cornea_centre_position_ics_;
        } else {
            cornea_centre_position_ics = WCStoICS(cornea_centre_position_wcs_ - eye_position_offset_);
        }
        mtx_eye_position_.unlock();
    }

    bool EyeEstimator::estimateEyePositions(EyeInfo& eye_info, const bool add_correction) {
        cv::Point3d eye_centre_position_wcs{};
        cv::Point2d eye_centre_position_ics{}, cornea_centre_position_ics{};
        cv::Vec2d gaze_direction_angles_precorrection_rad{}, gaze_direction_angles_rad{};
        double pupil_diameter_mm{};
        cv::Point3d cornea_centre_position_wcs{};

        const bool result = detectEye(eye_info, eye_centre_position_wcs, cornea_centre_position_wcs, gaze_direction_angles_precorrection_rad);
        if (!result) {
            return false;
        }
        findPupilDiameter(eye_info.pupil_centre_position_ics, eye_info.pupil_radius_ics, cornea_centre_position_wcs, pupil_diameter_mm);

        if (add_correction) {
            eye_centre_position_wcs += eye_position_offset_;
            cornea_centre_position_wcs += eye_position_offset_;

            double theta = theta_polynomial_->getEstimation({gaze_direction_angles_precorrection_rad[0], gaze_direction_angles_precorrection_rad[1]});
            double phi = phi_polynomial_->getEstimation({gaze_direction_angles_precorrection_rad[0], gaze_direction_angles_precorrection_rad[1]});
            gaze_direction_angles_rad = {theta, phi};
        } else {
            gaze_direction_angles_rad = gaze_direction_angles_precorrection_rad;
        }

        cv::Vec3d model_gaze_direction{};
        Utils::anglesToVector(gaze_direction_angles_rad, model_gaze_direction);

        eye_centre_position_ics = WCStoICS(eye_centre_position_wcs);
        cornea_centre_position_ics = WCStoICS(cornea_centre_position_wcs);

        const double k = (marker_depth_ - cornea_centre_position_wcs.z) / model_gaze_direction[2];
        auto gaze_point = cornea_centre_position_wcs + static_cast<cv::Point3d>(k * model_gaze_direction);
        gaze_point_sum_ += gaze_point;

        if (gaze_point_history_full_) {
            gaze_point_sum_ -= gaze_point_buffer_[gaze_point_index_];
            gaze_point_buffer_[gaze_point_index_] = gaze_point;
            gaze_point_index_ = (gaze_point_index_ + 1) % GAZE_BUFFER;
            gaze_point = gaze_point_sum_ / GAZE_BUFFER;
        } else {
            gaze_point_buffer_[gaze_point_index_] = gaze_point;
            gaze_point_index_ = (gaze_point_index_ + 1) % GAZE_BUFFER;
            gaze_point = gaze_point_sum_ / gaze_point_index_;
        }
        if (gaze_point_index_ == 0) {
            gaze_point_history_full_ = true;
        }

        cv::Vec3d averaged_model_gaze_direction = gaze_point - cornea_centre_position_wcs;
        cv::normalize(averaged_model_gaze_direction, averaged_model_gaze_direction);

        mtx_eye_position_.lock();
        cornea_centre_position_wcs_ = cornea_centre_position_wcs;
        eye_centre_position_wcs_ = eye_centre_position_wcs;
        eye_centre_position_ics_ = eye_centre_position_ics;
        cornea_centre_position_ics_ = cornea_centre_position_ics;
        pupil_diameter_mm_ = pupil_diameter_mm;
        model_gaze_direction_ = averaged_model_gaze_direction;
        gaze_point_ = {gaze_point.x, gaze_point.y};
        normalized_gaze_point_.x = (gaze_point.x - bottom_left_gaze_window_x_mm_) / (upper_right_gaze_window_x_mm_ - bottom_left_gaze_window_x_mm_);
        normalized_gaze_point_.y = (gaze_point.y - bottom_left_gaze_window_y_mm_) / (upper_right_gaze_window_y_mm_ - bottom_left_gaze_window_y_mm_);
        mtx_eye_position_.unlock();

        return result;
    }

    void EyeEstimator::getGazeDirection(cv::Vec3d& gaze_direction) {
        mtx_eye_position_.lock();
        gaze_direction = model_gaze_direction_;
        mtx_eye_position_.unlock();
    }

    void EyeEstimator::updateFineTuning() {
        eye_position_offset_ = features_params_->position_offset;
        theta_polynomial_->setCoefficients(features_params_->polynomial_theta);
        phi_polynomial_->setCoefficients(features_params_->polynomial_phi);
        marker_depth_ = features_params_->marker_depth;
    }

    cv::Point2d EyeEstimator::getNormalizedGazePoint() const {
        return normalized_gaze_point_;
    }
} // et
