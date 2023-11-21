#include "eye_tracker/eye/ModelEyeEstimator.hpp"
#include "eye_tracker/Utils.hpp"

#include <opencv2/core/core.hpp>

namespace et
{
    ModelEyeEstimator::ModelEyeEstimator(int camera_id, cv::Point3d eye_position) : EyeEstimator(camera_id, eye_position)
    {
        // Create a minimizer for used for finding cornea centre.
        nodal_point_optimizer_ = new NodalPointOptimizer(camera_id_);
        nodal_point_optimizer_->initialize();
        minimizer_function_ = cv::Ptr<cv::DownhillSolver::Function>{nodal_point_optimizer_};
        solver_ = cv::DownhillSolver::create();
        solver_->setFunction(minimizer_function_);
        cv::Mat step = (cv::Mat_<double>(1, 2) << 50, 50);
        solver_->setInitStep(step);

        pupil_cornea_distance_ = 4.2;
        pupil_eye_centre_distance_ = 9.5;
    }

    bool ModelEyeEstimator::detectEye(EyeInfo &eye_info, cv::Point3d &nodal_point, cv::Point3d &eye_centre, cv::Point3d &visual_axis)
    {
        cv::Vec3d pupil_position{ICStoCCS(eye_info.pupil)};

        std::vector<cv::Vec3d> glint_positions{};

        auto leds = Settings::parameters.leds_positions[camera_id_];
        cv::Mat camera_pos_mat = Settings::parameters.camera_params[camera_id_].extrinsic_matrix.t();
        for (int i = 0; i < leds.size(); i++)
        {
            cv::Vec4d led_homo;
            led_homo[0] = leds[i][0];
            led_homo[1] = leds[i][1];
            led_homo[2] = leds[i][2];
            led_homo[3] = 1.0;
            cv::Mat converted_leds = cv::Mat(led_homo).t() * camera_pos_mat;
            leds[i] = {converted_leds.at<double>(0), converted_leds.at<double>(1), converted_leds.at<double>(2)};
        }

        // calculate planes with glints, LEDs, eye's nodal point, and camera's
        // nodal point.
        std::vector<cv::Vec3d> v1v2s{};
        for (int i = 0; i < eye_info.glints.size(); i++)
        {
            cv::Vec3d v1{leds[i]};
            cv::normalize(v1, v1);
            cv::Vec3d v2{ICStoCCS(eye_info.glints[i])};
            glint_positions.push_back(v2);
            cv::normalize(v2, v2);
            cv::Vec3d v1v2{v1.cross(v2)};
            cv::normalize(v1v2, v1v2);
            v1v2s.push_back(v1v2);
        }

        // Find the intersection of all planes which is a vector between camera's
        // nodal point and eye's nodal point.
        cv::Vec3d avg_np2c_dir{};
        int counter{0};
        for (int i = 0; i < v1v2s.size(); i++)
        {
            if (i == 1 || i == 4)
            {
                continue;
            }
            for (int j = i + 1; j < v1v2s.size(); j++)
            {
                cv::Vec3d np2c_dir{v1v2s[i].cross(v1v2s[j])};
                cv::normalize(np2c_dir, np2c_dir);
                if (np2c_dir(2) < 0)
                {
                    np2c_dir = -np2c_dir;
                }
                avg_np2c_dir += np2c_dir;
                counter++;
            }
        }

        if (counter == 0)
        {
            return false;
        }

        for (int i = 0; i < 3; i++)
        {
            avg_np2c_dir(i) = avg_np2c_dir(i) / counter;
        }

        if (nodal_point_optimizer_)
        {
            nodal_point_optimizer_->setParameters(avg_np2c_dir, glint_positions.data(), leds, camera_nodal_point_);
        }
        cv::Mat x = (cv::Mat_<double>(1, 2) << 300, 300);
        // Finds the best candidate for cornea centre.
        solver_->minimize(x);
        double k = x.at<double>(0, 0);
        cv::Vec3d nodal_point_vec = camera_nodal_point_ + avg_np2c_dir * k;
        nodal_point = nodal_point_vec;

        cv::Vec3d pupil = calculatePositionOnPupil(pupil_position, nodal_point_vec);

        cv::Vec3d eye_centre_vec{};
        if (pupil != cv::Vec3d())
        {
            cv::Vec3d pupil_direction{nodal_point_vec - pupil};
            cv::normalize(pupil_direction, pupil_direction);
            // Eye centre lies in the same vector as cornea centre and pupil centre.

            eye_centre_vec = pupil + pupil_eye_centre_distance_ * pupil_direction;
        }
        else
        {
            return false;
        }
        eye_centre = eye_centre_vec;

        eye_centre = CCStoWCS(eye_centre);
        nodal_point = CCStoWCS(nodal_point);

        cv::Point3d optical_axis_p = nodal_point - eye_centre;
        cv::Vec3d optical_axis = optical_axis_p;
        cv::normalize(optical_axis, optical_axis);
        visual_axis = Utils::opticalToVisualAxis(optical_axis, setup_variables_->alpha, setup_variables_->beta);

        if (eye_centre_vec == cv::Vec3d())
        {
            return false;
        }
        return true;
    }

    ModelEyeEstimator::~ModelEyeEstimator()
    {
        if (!solver_->empty())
        {
            solver_.release();
        }
        if (!minimizer_function_.empty())
        {
            minimizer_function_.release();
        }
    }
} // et