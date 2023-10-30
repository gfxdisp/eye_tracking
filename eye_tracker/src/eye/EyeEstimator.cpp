#include "eye_tracker/eye/EyeEstimator.hpp"
#include "eye_tracker/Utils.hpp"

namespace et
{
    EyeEstimator::EyeEstimator(int camera_id, cv::Point3f eye_position) : camera_id_{camera_id}
    {
        setup_variables_ = &Settings::parameters.user_polynomial_params[camera_id]->setup_variables;
        intrinsic_matrix_ = &Settings::parameters.camera_params[camera_id].intrinsic_matrix;
        capture_offset_ = &Settings::parameters.camera_params[camera_id].capture_offset;
        cv::Vec4f homo_eye_position{eye_position.x, eye_position.y, eye_position.z, 1.0f};
        cv::Mat camera_pos_mat = cv::Mat(homo_eye_position).t() * Settings::parameters.camera_params[camera_id].extrinsic_matrix.t();
        cv::Point3f camera_pos{camera_pos_mat.at<float>(0), camera_pos_mat.at<float>(1), camera_pos_mat.at<float>(2)};
        createInvertedProjectionMatrix(camera_pos);
        inv_extrinsic_matrix_ = Settings::parameters.camera_params[camera_id].extrinsic_matrix.inv().t();

        camera_nodal_point_ = {0, 0, 0};

        pupil_cornea_distance_ = 4.2;
    }

    cv::Vec3f EyeEstimator::ICStoCCS(const cv::Point2f point)
    {
        cv::Vec4f homo_point{point.x + capture_offset_->width, point.y + capture_offset_->height, 0, 1};
        // Multiplies by inverted projection matrix to get camera space coordinates.
        cv::Mat p{cv::Mat(homo_point).t() * inv_projection_matrix_};
        float x = p.at<float>(0) / p.at<float>(3);
        float y = p.at<float>(1) / p.at<float>(3);
        float z = p.at<float>(2) / p.at<float>(3);

        return {x, y, z};
    }

    cv::Vec3f EyeEstimator::CCStoWCS(const cv::Vec3f point)
    {
        cv::Vec4f homo_point{point[0], point[1], point[2], 1.0f};
        cv::Mat world_pos = cv::Mat(homo_point).t() * inv_extrinsic_matrix_;
        float x = world_pos.at<float>(0) / world_pos.at<float>(3);
        float y = world_pos.at<float>(1) / world_pos.at<float>(3);
        float z = world_pos.at<float>(2) / world_pos.at<float>(3);

        return {x, y, z};
    }

    cv::Vec3f EyeEstimator::ICStoWCS(const cv::Point2f point)
    {
        return CCStoWCS(ICStoCCS(point));
    }

    void EyeEstimator::createInvertedProjectionMatrix(cv::Point3f eye_position)
    {
        cv::Point3f pupil_plane_position = eye_position;
        // Shifting from the cornea apex to pupil
        pupil_plane_position.z -= 13.1 - 4.2 / 2;
        // Projection matrix created according to the lecture notes:
        // https://www.cl.cam.ac.uk/teaching/2122/AGIP/lf_rendering.pdf
        cv::Point3f normal{0, 0, 1};
//        cv::Point3f wf = pupil_plane_position;
        cv::Point3f wf{0, 0, -1};
        float view_data[4][4]{{1,        0,        0,        0},
                              {0,        1,        0,        0},
                              {normal.x, normal.y, normal.z, -normal.dot(wf)},
                              {0,        0,        1,        0}};
        cv::Mat projection_matrix{4, 4, CV_32FC1, view_data};
        projection_matrix = projection_matrix.t();


        float intrinsic_matrix_arr[9];
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                intrinsic_matrix_arr[i * 3 + j] = intrinsic_matrix_->at<float>(cv::Point(i, j));
            }
        }

        // Convert Matlab's 1-based indexing to C++'s 0-based indexing.
        intrinsic_matrix_arr[2] -= 1;
        intrinsic_matrix_arr[5] -= 1;

        float intrinsic_data[4][4]{{intrinsic_matrix_arr[0], intrinsic_matrix_arr[3], intrinsic_matrix_arr[6], 0},
                                   {intrinsic_matrix_arr[1], intrinsic_matrix_arr[4], intrinsic_matrix_arr[7], 0},
                                   {0,                       0,                       intrinsic_matrix_arr[8], 0},
                                   {intrinsic_matrix_arr[2], intrinsic_matrix_arr[5], 0,                       1}};

        cv::Mat intrinsic_matrix{4, 4, CV_32FC1, intrinsic_data};

        inv_projection_matrix_ = projection_matrix * intrinsic_matrix;
        inv_projection_matrix_ = inv_projection_matrix_.inv();
    }

    cv::Vec3f EyeEstimator::calculatePositionOnPupil(const cv::Vec3f &pupil_px_position, const cv::Vec3f &cornea_centre)
    {
        cv::Vec3f pupil_position{};
        double t{};
        cv::Vec3f pupil_dir{-pupil_px_position};
        cv::normalize(pupil_dir, pupil_dir);
        bool intersected{Utils::getRaySphereIntersection(cv::Vec3f(0.0f), pupil_dir, cornea_centre,
                                                         setup_variables_->cornea_curvature_radius, t)};

        if (intersected)
        {
            cv::Vec3f pupil_on_cornea{t * pupil_dir};
            cv::Vec3f nv{pupil_on_cornea - cornea_centre};
            cv::normalize(nv, nv);
            cv::Vec3d m_dir{-pupil_px_position};
            cv::normalize(m_dir, m_dir);
            cv::Vec3f direction{getRefractedRay(m_dir, nv, setup_variables_->cornea_refraction_index)};
            intersected = Utils::getRaySphereIntersection(pupil_on_cornea, direction, cornea_centre,
                                                          pupil_cornea_distance_, t);
            if (intersected)
            {
                pupil_position = pupil_on_cornea + t * direction;
            }
        }
        return pupil_position;
    }

    cv::Vec3d
    EyeEstimator::getRefractedRay(const cv::Vec3d &direction, const cv::Vec3d &normal, double refraction_index)
    {
        double nr{1 / refraction_index};
        double m_cos{(-direction).dot(normal)};
        double m_sin{nr * nr * (1 - m_cos * m_cos)};
        cv::Vec3d t{nr * direction + (nr * m_cos - std::sqrt(1 - m_sin)) * normal};
        cv::normalize(t, t);
        return t;
    }

    bool EyeEstimator::findPupilDiameter(cv::Point2f pupil_pix_position, int pupil_px_radius,
                                         const cv::Vec3f &cornea_centre_position, float &diameter)
    {
        cv::Vec3f pupil_position{ICStoCCS(pupil_pix_position)};

        cv::Vec3f pupil_right_position = ICStoCCS(pupil_pix_position + cv::Point2f((float) pupil_px_radius, 0.0f));

        // Estimates pupil position based on its position in the image and the cornea centre.
        cv::Vec3f pupil = calculatePositionOnPupil(pupil_position, cornea_centre_position);
        // Estimates position of the right part of the pupil.
        cv::Vec3f pupil_right = calculatePositionOnPupil(pupil_right_position, cornea_centre_position);

        if (pupil != cv::Vec3f() && pupil_right != cv::Vec3f())
        {
            cv::Vec3f pupil_proj = pupil;
            cv::Vec3f pupil_right_proj = pupil_right;
            pupil_proj(2) = 0.0f;
            pupil_right_proj(2) = 0.0f;
            // Diameter is estimated as double the distance between pupil's centre
            // and pupil's right point.
            diameter = (float) (2 * cv::norm(pupil_proj - pupil_right_proj));
            return true;
        }

        return false;
    }

    void EyeEstimator::unproject(cv::Point3f point, cv::Point2f &pixel)
    {
        cv::Vec3f point_vec{point};
        // Multiplies by intrinsic matrix to get image space coordinates.
        cv::Mat unprojected = intrinsic_matrix_->t() * point_vec;
        float x = unprojected.at<float>(0);
        float y = unprojected.at<float>(1);
        float w = unprojected.at<float>(2);
        // Shifts the position to account for region-of-interest.
        pixel = {x / w - (float) capture_offset_->width, y / w - (float) capture_offset_->height};
    }

    void EyeEstimator::getGazeDirection(cv::Point3f nodal_point, cv::Point3f eye_centre, cv::Vec3f &gaze_direction)
    {
        cv::Vec3f optical_axis = nodal_point - eye_centre;
        cv::normalize(optical_axis, optical_axis);
        gaze_direction = Utils::opticalToVisualAxis(optical_axis, setup_variables_->alpha, setup_variables_->beta);
    }

    void EyeEstimator::getEyeCentrePosition(cv::Point3f &eye_centre)
    {
        mtx_eye_position_.lock();
        eye_centre = eye_centre_;
        mtx_eye_position_.unlock();
    }

    void EyeEstimator::getCorneaCurvaturePosition(cv::Point3f &cornea_centre)
    {
        mtx_eye_position_.lock();
        cornea_centre = cornea_centre_;
        mtx_eye_position_.unlock();
    }

    void EyeEstimator::getPupilDiameter(float &pupil_diameter)
    {
        mtx_eye_position_.lock();
        pupil_diameter = pupil_diameter_;
        mtx_eye_position_.unlock();
    }

    cv::Point2f EyeEstimator::getCorneaCurvaturePixelPosition()
    {
        return eye_centre_pixel_;
    }

    cv::Point2f EyeEstimator::getEyeCentrePixelPosition()
    {
        return cornea_centre_pixel_;
    }

    bool EyeEstimator::findEye(EyeInfo &eye_info)
    {
        cv::Point3f nodal_point{}, eye_centre{}, visual_axis{};
        cv::Point2f eye_centre_pixel{}, cornea_centre_pixel{};
        float pupil_diameter{};
        cv::Vec3f gaze_direction{};

        bool result = detectEye(eye_info, nodal_point, eye_centre, visual_axis);
        unproject(eye_centre_, eye_centre_pixel_);
        unproject(cornea_centre_, cornea_centre_pixel_);
        findPupilDiameter(eye_info.pupil, eye_info.pupil_radius, cornea_centre_, pupil_diameter_);
        getGazeDirection(nodal_point, eye_centre, gaze_direction_);

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

    void EyeEstimator::getGazeDirection(cv::Vec3f &gaze_direction)
    {
        mtx_eye_position_.lock();
        gaze_direction = gaze_direction_;
        mtx_eye_position_.unlock();
    }
} // et