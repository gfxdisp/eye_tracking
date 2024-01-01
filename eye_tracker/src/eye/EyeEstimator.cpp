#include "eye_tracker/eye/EyeEstimator.hpp"
#include "eye_tracker/Utils.hpp"

namespace et
{
    EyeEstimator::EyeEstimator(int camera_id) : camera_id_{camera_id}
    {
        setup_variables_ = &Settings::parameters.user_polynomial_params[camera_id]->setup_variables;
        intrinsic_matrix_ = &Settings::parameters.camera_params[camera_id].intrinsic_matrix;
        capture_offset_ = &Settings::parameters.camera_params[camera_id].capture_offset;
        dimensions_ = &Settings::parameters.camera_params[camera_id].dimensions;
        createInvertedProjectionMatrix();
        inv_extrinsic_matrix_ = Settings::parameters.camera_params[camera_id].extrinsic_matrix.inv().t();
        extrinsic_matrix_ = Settings::parameters.camera_params[camera_id].extrinsic_matrix.t();

        camera_nodal_point_ = {0, 0, 0};

        pupil_cornea_distance_ = 4.2;
        eye_cornea_distance_ = 5.3;
        cornea_radius_ = 7.8;
        refraction_index_ = 1.3375;
    }

    cv::Vec3d EyeEstimator::ICStoCCS(const cv::Point2d point)
    {
        cv::Vec4d homo_point{point.x + capture_offset_->width, point.y + capture_offset_->height, 0, 1};
        // Multiplies by inverted projection matrix to get camera space coordinates.
        cv::Mat p{cv::Mat(homo_point).t() * inv_projection_matrix_};
        double x = p.at<double>(0) / p.at<double>(3);
        double y = p.at<double>(1) / p.at<double>(3);
        double z = p.at<double>(2) / p.at<double>(3);

        // z = -intrinsic_matrix_->at<double>(cv::Point(0, 0)) * 6.144 / dimensions_->width;
        // double shift_x = intrinsic_matrix_->at<double>(cv::Point(0, 2)) - dimensions_->width * 0.5;
        // double shift_y = intrinsic_matrix_->at<double>(cv::Point(1, 2)) - dimensions_->height * 0.5;
        // x = -(point.x - shift_x + capture_offset_->width - dimensions_->width * 0.5) / (dimensions_->width * 0.5) * 6.144 / 2;
        // y = -(point.y - shift_y + capture_offset_->height - dimensions_->height * 0.5) / (dimensions_->height * 0.5) * 4.915 / 2;

        return {x, y, z};
    }

    cv::Vec3d EyeEstimator::CCStoWCS(const cv::Vec3d point)
    {
        cv::Vec4d homo_point{point[0], point[1], point[2], 1.0};
        cv::Mat world_pos = cv::Mat(homo_point).t() * inv_extrinsic_matrix_;
        double x = world_pos.at<double>(0) / world_pos.at<double>(3);
        double y = world_pos.at<double>(1) / world_pos.at<double>(3);
        double z = world_pos.at<double>(2) / world_pos.at<double>(3);

        return {x, y, z};
    }

    cv::Vec3d EyeEstimator::ICStoWCS(const cv::Point2d point)
    {
        return CCStoWCS(ICStoCCS(point));
    }

    cv::Point2d EyeEstimator::CCStoICS(cv::Point3d point)
    {
        cv::Vec3d homo_point{point.x, point.y, point.z};
        cv::Mat ccs_pos = cv::Mat(homo_point).t() * (*intrinsic_matrix_);
        double x = ccs_pos.at<double>(0) / ccs_pos.at<double>(2);
        double y = ccs_pos.at<double>(1) / ccs_pos.at<double>(2);
        return {x - capture_offset_->width, y - capture_offset_->height};
    }

    cv::Point2d EyeEstimator::WCStoICS(cv::Point3d point)
    {
        return CCStoICS(WCStoCCS(point));
    }

    cv::Point3d EyeEstimator::WCStoCCS(cv::Point3d point)
    {
        cv::Vec4d homo_point{point.x, point.y, point.z, 1.0};
        cv::Mat ccs_pos = cv::Mat(homo_point).t() * extrinsic_matrix_;
        double x = ccs_pos.at<double>(0) / ccs_pos.at<double>(3);
        double y = ccs_pos.at<double>(1) / ccs_pos.at<double>(3);
        double z = ccs_pos.at<double>(2) / ccs_pos.at<double>(3);

        return {x, y, z};
    }

    void EyeEstimator::createInvertedProjectionMatrix()
    {
        // Projection matrix created according to the lecture notes:
        // https://www.cl.cam.ac.uk/teaching/2122/AGIP/lf_rendering.pdf
        cv::Point3d normal{0, 0, 1};
//        cv::Point3d wf = pupil_plane_position;
        // cv::Point3d wf{0, 0, -28.3074};
        cv::Point3d wf{0, 0, -1.0};
        double view_data[4][4]{{1,        0,        0,        0},
                              {0,        1,        0,        0},
                              {normal.x, normal.y, normal.z, -normal.dot(wf)},
                              {0,        0,        1,        0}};
        cv::Mat projection_matrix{4, 4, CV_64FC1, view_data};
        projection_matrix = projection_matrix.t();


        double intrinsic_matrix_arr[9];
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                intrinsic_matrix_arr[i * 3 + j] = intrinsic_matrix_->at<double>(cv::Point(i, j));
            }
        }

//        // Convert Matlab's 1-based indexing to C++'s 0-based indexing.
//        intrinsic_matrix_arr[2] -= 1;
//        intrinsic_matrix_arr[5] -= 1;

        double intrinsic_data[4][4]{{intrinsic_matrix_arr[0], intrinsic_matrix_arr[3], intrinsic_matrix_arr[6], 0},
                                   {intrinsic_matrix_arr[1], intrinsic_matrix_arr[4], intrinsic_matrix_arr[7], 0},
                                   {0,                       0,                       intrinsic_matrix_arr[8], 0},
                                   {intrinsic_matrix_arr[2], intrinsic_matrix_arr[5], 0,                       1}};

        cv::Mat intrinsic_matrix{4, 4, CV_64FC1, intrinsic_data};

        inv_projection_matrix_ = projection_matrix * intrinsic_matrix;
        inv_projection_matrix_ = inv_projection_matrix_.inv();
    }

    cv::Vec3d EyeEstimator::calculatePositionOnPupil(const cv::Vec3d &pupil_px_position, const cv::Vec3d &cornea_centre)
    {
        cv::Vec3d pupil_position{};
        double t{};
        cv::Vec3d pupil_dir{-pupil_px_position};
        cv::normalize(pupil_dir, pupil_dir);
        bool intersected{Utils::getRaySphereIntersection(cv::Vec3d(0.0), pupil_dir, cornea_centre,
                                                         cornea_radius_, t)};

        if (intersected)
        {
            cv::Vec3d pupil_on_cornea{t * pupil_dir};
            cv::Vec3d nv{pupil_on_cornea - cornea_centre};
            cv::normalize(nv, nv);
            cv::Vec3d m_dir{-pupil_px_position};
            cv::normalize(m_dir, m_dir);
            cv::Vec3d direction{Utils::getRefractedRay(m_dir, nv, refraction_index_)};
            intersected = Utils::getRaySphereIntersection(pupil_on_cornea, direction, cornea_centre,
                                                          pupil_cornea_distance_, t);
            if (intersected)
            {
                pupil_position = pupil_on_cornea + t * direction;
            }
        }
        return pupil_position;
    }

    bool EyeEstimator::findPupilDiameter(cv::Point2d pupil_pix_position, int pupil_px_radius,
                                         const cv::Vec3d &cornea_centre_position, double &diameter)
    {
        cv::Vec3d pupil_position{ICStoCCS(pupil_pix_position)};

        cv::Vec3d pupil_right_position = ICStoCCS(pupil_pix_position + cv::Point2d(pupil_px_radius, 0.0));

        // Estimates pupil position based on its position in the image and the cornea centre.
        cv::Vec3d pupil = calculatePositionOnPupil(pupil_position, cornea_centre_position);
        // Estimates position of the right part of the pupil.
        cv::Vec3d pupil_right = calculatePositionOnPupil(pupil_right_position, cornea_centre_position);

        if (pupil != cv::Vec3d() && pupil_right != cv::Vec3d())
        {
            cv::Vec3d pupil_proj = pupil;
            cv::Vec3d pupil_right_proj = pupil_right;
            pupil_proj(2) = 0.0;
            pupil_right_proj(2) = 0.0;
            // Diameter is estimated as double the distance between pupil's centre
            // and pupil's right point.
            diameter = (2 * cv::norm(pupil_proj - pupil_right_proj));
            return true;
        }

        return false;
    }

    void EyeEstimator::getGazeDirection(cv::Point3d nodal_point, cv::Point3d eye_centre, cv::Vec3d &gaze_direction)
    {
        cv::Vec3d optical_axis = nodal_point - eye_centre;
        cv::normalize(optical_axis, optical_axis);
        gaze_direction = Utils::opticalToVisualAxis(optical_axis, setup_variables_->alpha, setup_variables_->beta);
    }

    void EyeEstimator::getEyeCentrePosition(cv::Point3d &eye_centre)
    {
        mtx_eye_position_.lock();
        eye_centre = eye_centre_;
        mtx_eye_position_.unlock();
    }

    void EyeEstimator::getCorneaCurvaturePosition(cv::Point3d &cornea_centre)
    {
        mtx_eye_position_.lock();
        cornea_centre = cornea_centre_;
        mtx_eye_position_.unlock();
    }

    void EyeEstimator::getPupilDiameter(double &pupil_diameter)
    {
        mtx_eye_position_.lock();
        pupil_diameter = pupil_diameter_;
        mtx_eye_position_.unlock();
    }

    cv::Point2d EyeEstimator::getCorneaCurvaturePixelPosition()
    {
        return eye_centre_pixel_;
    }

    cv::Point2d EyeEstimator::getEyeCentrePixelPosition()
    {
        return cornea_centre_pixel_;
    }

    bool EyeEstimator::findEye(EyeInfo &eye_info)
    {
        cv::Point3d nodal_point{}, eye_centre{}, visual_axis{};
        cv::Point2d eye_centre_pixel{}, cornea_centre_pixel{};
        double pupil_diameter{};
        cv::Vec3d gaze_direction{};

        bool result = detectEye(eye_info, nodal_point, eye_centre, visual_axis);
        eye_centre_pixel = WCStoICS(eye_centre);
        cornea_centre_pixel = WCStoICS(nodal_point);
        findPupilDiameter(eye_info.pupil, eye_info.pupil_radius, nodal_point, pupil_diameter);
        getGazeDirection(nodal_point, eye_centre, gaze_direction);

        mtx_eye_position_.lock();
        cornea_centre_ = nodal_point;
        eye_centre_ = eye_centre;
        eye_centre_pixel_ = eye_centre_pixel;
        cornea_centre_pixel_ = cornea_centre_pixel;
        pupil_diameter_ = pupil_diameter;
        gaze_direction_ = gaze_direction;
        mtx_eye_position_.unlock();

        return result;
    }

    void EyeEstimator::getGazeDirection(cv::Vec3d &gaze_direction)
    {
        mtx_eye_position_.lock();
        gaze_direction = gaze_direction_;
        mtx_eye_position_.unlock();
    }
} // et