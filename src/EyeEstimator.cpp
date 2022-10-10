#include "EyeEstimator.hpp"
#include "RayPointMinimizer.hpp"
#include "Settings.hpp"
#include "Utils.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>

using KFMatD = cv::Mat_<double>;

namespace et {

cv::Mat EyeEstimator::visual_axis_rotation_matrix_{};

EyeEstimator::EyeEstimator() {
    ray_point_minimizer_ = new RayPointMinimizer();
    ray_point_minimizer_->initialize();
    minimizer_function_ =
        cv::Ptr<cv::DownhillSolver::Function>{ray_point_minimizer_};
    solver_ = cv::DownhillSolver::create();
    solver_->setFunction(minimizer_function_);
    cv::Mat step = (cv::Mat_<double>(1, 2) << 50, 50);
    solver_->setInitStep(step);

    createVisualAxis();

    static std::string ellipse_fitting_coeffs_name = "ellipse_fitting_coeffs.txt";

    if (std::filesystem::exists(ellipse_fitting_coeffs_name)) {
        auto coefficients =
            Utils::readFloatRowsCsv(ellipse_fitting_coeffs_name);
        eye_np_pos_x_fit_.setCoefficients(coefficients[0]);
        eye_centre_pos_x_fit_.setCoefficients(coefficients[1]);
        eye_np_pos_y_fit_.setCoefficients(coefficients[2]);
        eye_centre_pos_y_fit_.setCoefficients(coefficients[3]);
        eye_np_pos_z_fit_.setCoefficients(coefficients[4]);
        eye_centre_pos_z_fit_.setCoefficients(coefficients[5]);
    } else {
        std::vector<std::vector<float>> poly_fit_input_data =
            Utils::readFloatColumnsCsv("detected_data.csv");

        auto pupil_x = &poly_fit_input_data[2];
        auto pupil_y = &poly_fit_input_data[3];
        auto el_centre_x = &poly_fit_input_data[4];
        auto el_centre_y = &poly_fit_input_data[5];
        auto el_width = &poly_fit_input_data[6];
        auto el_height = &poly_fit_input_data[7];
        auto el_angle = &poly_fit_input_data[8];

        std::vector<std::vector<float>> poly_fit_output_data =
            Utils::readFloatColumnsCsv("eye_data.csv");

        auto nodal_x = &poly_fit_output_data[1];
        auto nodal_y = &poly_fit_output_data[2];
        auto nodal_z = &poly_fit_output_data[3];
        auto centre_x = &poly_fit_output_data[4];
        auto centre_y = &poly_fit_output_data[5];
        auto centre_z = &poly_fit_output_data[6];

        std::vector<std::vector<float> *> input_vars{5};
        input_vars[0] = pupil_x;
        input_vars[1] = el_centre_x;
        input_vars[2] = el_width;
        input_vars[3] = el_height;
        input_vars[4] = el_angle;
        eye_np_pos_x_fit_.fit(input_vars, nodal_x);
        eye_centre_pos_x_fit_.fit(input_vars, centre_x);

        input_vars[0] = pupil_y;
        input_vars[1] = el_centre_y;
        eye_np_pos_y_fit_.fit(input_vars, nodal_y);
        eye_centre_pos_y_fit_.fit(input_vars, centre_y);

        input_vars[0] = el_centre_x;
        eye_np_pos_z_fit_.fit(input_vars, nodal_z);
        eye_centre_pos_z_fit_.fit(input_vars, centre_z);

        std::vector<std::vector<float>> all_coefficients{};
        all_coefficients.push_back(eye_np_pos_x_fit_.getCoefficients());
        all_coefficients.push_back(eye_centre_pos_x_fit_.getCoefficients());
        all_coefficients.push_back(eye_np_pos_y_fit_.getCoefficients());
        all_coefficients.push_back(eye_centre_pos_y_fit_.getCoefficients());
        all_coefficients.push_back(eye_np_pos_z_fit_.getCoefficients());
        all_coefficients.push_back(eye_centre_pos_z_fit_.getCoefficients());
        Utils::writeFloatCsv(all_coefficients, ellipse_fitting_coeffs_name);
    }
}

EyeEstimator::~EyeEstimator() {
    solver_.release();
    minimizer_function_.release();
}

void EyeEstimator::initialize(int camera_id) {
    leds_positions_ = &Settings::parameters.leds_positions[camera_id];
    gaze_shift_ = &Settings::parameters.camera_params[camera_id].gaze_shift;
    kalman_eye_ = makeKalmanFilter(et::Settings::parameters.camera_params[camera_id].framerate);
    kalman_gaze_ = makeKalmanFilter(et::Settings::parameters.camera_params[camera_id].framerate);
    intrinsic_matrix_ = &Settings::parameters.camera_params[camera_id].intrinsic_matrix;
    capture_offset_ = &Settings::parameters.camera_params[camera_id].capture_offset;
    dimensions_ = &et::Settings::parameters.camera_params[camera_id].dimensions;
    distortion_coefficients_ = &Settings::parameters.camera_params[camera_id].distortion_coefficients;

    cv::Point3f normal{0, 0, 1};
    cv::Point3f wf{0, 0, -1};
    float view_data[4][4]{{1, 0, 0, 0},
                          {0, 1, 0, 0},
                          {normal.x, normal.y, normal.z, -normal.dot(wf)},
                          {0, 0, 1, 0}};
    cv::Mat projection_matrix{4, 4, CV_32FC1, view_data};
    projection_matrix = projection_matrix.t();

    float intr_mtx[9];
    CameraParams *camera_params =
        &et::Settings::parameters.camera_params[camera_id];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            intr_mtx[i * 3 + j] =
                camera_params->intrinsic_matrix.at<float>(
                    cv::Point(i, j));
        }
    }

    float intrinsic_data[4][4]{{intr_mtx[0], intr_mtx[3], intr_mtx[6], 0},
                               {intr_mtx[1], intr_mtx[4], intr_mtx[7], 0},
                               {0, 0, intr_mtx[8], 0},
                               {intr_mtx[2], intr_mtx[5], 0, 1}};

    cv::Mat intrinsic_matrix{4, 4, CV_32FC1, intrinsic_data};

    full_projection_matrices_ =
        projection_matrix * intrinsic_matrix;
    full_projection_matrices_ =
        full_projection_matrices_.inv();
}

void EyeEstimator::getEyeFromModel(cv::Point2f pupil_pix_position,
                               std::vector<cv::Point2f> *glint_pix_positions,
                               int pupil_radius) {
    std::optional<cv::Vec3f> cornea_curvature{}, pupil{}, eye_centre{};

    pupil_pix_position_ = pupil_pix_position;
    glint_pix_positions_ = glint_pix_positions;

    cv::Vec3f pupil_position{ICStoCCS(undistort(pupil_pix_position_))};

    cv::Vec3f pupil_top_position = ICStoCCS(
        undistort(pupil_pix_position_ + cv::Point2f(pupil_radius, 0.0f)));

    std::vector<cv::Vec3f> glint_positions{};

    std::vector<cv::Vec3d> v1v2s{};
    for (int i = 0; i < glint_pix_positions_->size(); i++) {
        cv::Vec3f v1{(*leds_positions_)[i]};
        cv::normalize(v1, v1);
        cv::Vec3f v2{ICStoCCS(undistort((*glint_pix_positions_)[i]))};
        glint_positions.push_back(v2);
        cv::normalize(v2, v2);
        cv::Vec3d v1v2{v1.cross(v2)};
        cv::normalize(v1v2, v1v2);
        v1v2s.push_back(v1v2);
    }

    cv::Vec3d avg_bnorm{};
    int counter{0};
    for (int i = 0; i < v1v2s.size(); i++) {
        if (i == 1 || i == 4)
            continue;
        for (int j = i + 1; j < v1v2s.size(); j++) {
            cv::Vec3d bnorm{v1v2s[i].cross(v1v2s[j])};
            cv::normalize(bnorm, bnorm);
            if (bnorm(2) < 0) {
                bnorm = -bnorm;
            }
            avg_bnorm += bnorm;
            counter++;
        }
    }

    if (counter == 0) {
        return;
    }

    for (int i = 0; i < 3; i++) {
        avg_bnorm(i) = avg_bnorm(i) / counter;
    }

    ray_point_minimizer_->setParameters(avg_bnorm, glint_positions.data(),
                                        *leds_positions_);
    cv::Mat x = (cv::Mat_<double>(1, 2) << 400, 400);
    solver_->minimize(x);
    double k = x.at<double>(0, 0);
    cornea_curvature = avg_bnorm * k;

    kalman_eye_.correct((KFMatD(3, 1) << (*cornea_curvature)(0),
                         (*cornea_curvature)(1), (*cornea_curvature)(2)));

    cornea_curvature = toPoint(kalman_eye_.predict());

    pupil = ICStoEyePosition(pupil_position, *cornea_curvature);
    cv::Vec3f pupil_top =
        ICStoEyePosition(pupil_top_position, *cornea_curvature);
    if (*pupil != cv::Vec3f()) {
        cv::Vec3f pupil_direction{*cornea_curvature - *pupil};
        cv::normalize(pupil_direction, pupil_direction);
        eye_centre = *pupil
            + Settings::parameters.eye_params.pupil_eye_centre_distance
                * pupil_direction;
    }
    mtx_eye_position_.lock();
    eye_position_ = {cornea_curvature, pupil, eye_centre};
    if (*pupil != cv::Vec3f() && pupil_top != cv::Vec3f()) {
        cv::Vec3f pupil_proj = *pupil;
        cv::Vec3f pupil_top_proj = pupil_top;
        pupil_proj(2) = 0.0f;
        pupil_top_proj(2) = 0.0f;
        pupil_diameter_ = 2 * cv::norm(pupil_proj - pupil_top_proj);
    }
    mtx_eye_position_.unlock();
}

void EyeEstimator::getEyeCentrePosition(cv::Vec3d &eye_centre) {
    mtx_eye_position_.lock();
    eye_centre = *eye_position_.eye_centre;
    mtx_eye_position_.unlock();
}

void EyeEstimator::getCorneaCurvaturePosition(cv::Vec3d &cornea_centre) {
    mtx_eye_position_.lock();
    cornea_centre = *eye_position_.cornea_curvature;
    mtx_eye_position_.unlock();
}

void EyeEstimator::getGazeDirection(cv::Vec3f &gaze_direction) {

    cv::Vec4f homo_inv_optical_axis{};
    mtx_eye_position_.lock();
    inv_optical_axis_ = *eye_position_.cornea_curvature - *eye_position_.pupil;
    inv_optical_axis_ -= *gaze_shift_;
    cv::normalize(inv_optical_axis_, inv_optical_axis_);
    for (int i = 0; i < 3; i++)
        homo_inv_optical_axis(i) = inv_optical_axis_(i);
    mtx_eye_position_.unlock();
    homo_inv_optical_axis(3) = 1.0f;
    cv::Mat visual_axis{visual_axis_rotation_matrix_ * homo_inv_optical_axis};
    cv::Vec4f homo_gaze_direction{};
    homo_gaze_direction = visual_axis.reshape(4).at<cv::Vec4f>();
    for (int i = 0; i < 3; i++)
        gaze_direction(i) = -homo_gaze_direction(i) / homo_gaze_direction(3);
}

void EyeEstimator::getPupilDiameter(float &pupil_diameter) {
    mtx_eye_position_.lock();
    pupil_diameter = pupil_diameter_;
    mtx_eye_position_.unlock();
}

void EyeEstimator::getEyeData(EyeData &eye_data) {
    if (eye_position_.cornea_curvature) {
        eye_data.cornea_curvature = *eye_position_.cornea_curvature;
    } else {
        eye_data.cornea_curvature = cv::Vec3f(0, 0, 0);
    }
    if (eye_position_.pupil) {
        eye_data.pupil = *eye_position_.pupil;
    } else {
        eye_data.pupil = cv::Vec3f(0, 0, 0);
    }
    if (eye_position_.eye_centre) {
        eye_data.eye_centre = *eye_position_.eye_centre;
    } else {
        eye_data.eye_centre = cv::Vec3f(0, 0, 0);
    }
}

cv::Point2f EyeEstimator::getCorneaCurvaturePixelPosition() {
    if (eye_position_) {
        return unproject(*eye_position_.cornea_curvature);
    }
    return {0.0, 0.0};
}

cv::Point2f EyeEstimator::getEyeCentrePixelPosition() {
    if (eye_position_) {
        return unproject(*eye_position_.eye_centre);
    }
    return {0.0, 0.0};
}

cv::Vec3f EyeEstimator::project(const cv::Vec3f &point) const {
    return point;
}

cv::Vec2f EyeEstimator::unproject(const cv::Vec3f &point) const {
    cv::Mat unprojected =
        (*intrinsic_matrix_).t()
        * point;
    float x = unprojected.at<float>(0);
    float y = unprojected.at<float>(1);
    float w = unprojected.at<float>(2);
    return {x / w - capture_offset_->width, y / w - capture_offset_->height};
}

cv::Vec3f EyeEstimator::ICStoCCS(const cv::Point2f &point) const {

    cv::Vec4f expanded_point{point.x, point.y, 0, 1};
    cv::Mat p{cv::Mat(expanded_point).t() * full_projection_matrices_};
    float x = p.at<float>(0) / p.at<float>(3);
    float y = p.at<float>(1) / p.at<float>(3);
    float z = p.at<float>(2) / p.at<float>(3);

    return {x, y, z};
}

cv::Vec3f EyeEstimator::CCStoWCS(const cv::Vec3f &point) const {
    return point;
}

cv::Vec3f EyeEstimator::WCStoCCS(const cv::Vec3f &point) const {
    return point;
}

cv::Vec2f EyeEstimator::CCStoICS(cv::Vec3f point) const {
    const double pixel_pitch = 0;
    return static_cast<cv::Point2f>(*dimensions_) / 2
        + cv::Point2f(point(0), point(1)) / pixel_pitch;
}

std::vector<cv::Vec3d>
EyeEstimator::lineSphereIntersections(const cv::Vec3d &sphere_centre, float radius,
                                  const cv::Vec3d &line_point,
                                  const cv::Vec3d &line_direction) {
    /* We are looking for points of the form line_point + k*line_direction, which are also radius away
         * from sphere_centre. This can be expressed as a quadratic in k: ak² + bk + c = radius². */
    const double a{cv::norm(line_direction, cv::NORM_L2SQR)};
    const double b{2 * line_direction.dot(line_point - sphere_centre)};
    const double c{cv::norm(line_point, cv::NORM_L2SQR)
                   + cv::norm(sphere_centre, cv::NORM_L2SQR)
                   - 2 * line_point.dot(sphere_centre)};
    const double DISCRIMINANT{std::pow(b, 2)
                              - 4 * a * (c - std::pow(radius, 2))};
    if (std::abs(DISCRIMINANT) < 1e-6) {
        return {line_point - line_direction * b / (2 * a)}; // One solution
    } else if (DISCRIMINANT < 0) {
        return {}; // No solutions
    } else {       // Two solutions
        const double sqrtDISCRIMINANT{std::sqrt(DISCRIMINANT)};
        return {line_point + line_direction * (-b + sqrtDISCRIMINANT) / (2 * a),
                line_point
                    + line_direction * (-b - sqrtDISCRIMINANT) / (2 * a)};
    }
}

cv::KalmanFilter EyeEstimator::makeKalmanFilter(float framerate) {
    constexpr static double VELOCITY_DECAY = 1.0;
    const static cv::Mat TRANSITION_MATRIX =
        (KFMatD(6, 6) << 1, 0, 0, 1.0f / framerate, 0, 0, 0, 1, 0, 0,
         1.0f / framerate, 0, 0, 0, 1, 0, 0, 1.0f / framerate, 0, 0, 0,
         VELOCITY_DECAY, 0, 0, 0, 0, 0, 0, VELOCITY_DECAY, 0, 0, 0, 0, 0, 0,
         VELOCITY_DECAY);
    const static cv::Mat MEASUREMENT_MATRIX = cv::Mat::eye(3, 6, CV_64F);
    const static cv::Mat PROCESS_NOISE_COV = cv::Mat::eye(6, 6, CV_64F) * 1000;
    const static cv::Mat MEASUREMENT_NOISE_COV =
        cv::Mat::eye(3, 3, CV_64F) * 0.1;
    const static cv::Mat ERROR_COV_POST = cv::Mat::eye(6, 6, CV_64F) * 1000;
    const static cv::Mat STATE_POST = cv::Mat::zeros(6, 1, CV_64F);

    cv::KalmanFilter KF(6, 3);
    // clone() is needed as, otherwise, the matrices will be used by reference, and all the filters will be the same
    KF.transitionMatrix = TRANSITION_MATRIX.clone();
    KF.measurementMatrix = MEASUREMENT_MATRIX.clone();
    KF.processNoiseCov = PROCESS_NOISE_COV.clone();
    KF.measurementNoiseCov = MEASUREMENT_NOISE_COV.clone();
    KF.errorCovPost = ERROR_COV_POST.clone();
    KF.statePost = STATE_POST.clone();
    KF.predict(); // Without this line, OpenCV complains about incorrect matrix dimensions
    return KF;
}

bool EyeEstimator::getRaySphereIntersection(const cv::Vec3f &ray_pos,
                                        const cv::Vec3d &ray_dir,
                                        const cv::Vec3f &sphere_pos,
                                        double sphere_radius, double &t) {
    double A{ray_dir.dot(ray_dir)};
    cv::Vec3f v{ray_pos - sphere_pos};
    double B{2 * v.dot(ray_dir)};
    double C{v.dot(v) - sphere_radius * sphere_radius};
    double delta{B * B - 4 * A * C};
    if (delta > 0) {
        double t1{(-B - std::sqrt(delta)) / (2 * A)};
        double t2{(-B + std::sqrt(delta)) / (2 * A)};
        if (t1 < 1e-5) {
            t = t2;
        } else if (t1 < 1e-5) {
            t = t1;
        } else {
            t = std::min(t1, t2);
        }
    }
    return (delta > 0);
}

cv::Vec3d EyeEstimator::getRefractedRay(const cv::Vec3d &direction,
                                    const cv::Vec3d &normal,
                                    double refraction_index) {
    double nr{1 / refraction_index};
    double mcos{(-direction).dot(normal)};
    double msin{nr * nr * (1 - mcos * mcos)};
    cv::Vec3d t{nr * direction + (nr * mcos - std::sqrt(1 - msin)) * normal};
    cv::normalize(t, t);
    return t;
}

bool EyeEstimator::isSetupUpdated() {
    return setup_updated_;
}

cv::Mat EyeEstimator::euler2rot(float *euler_angles) {
    cv::Mat rotationMatrix(4, 4, CV_32F);

    float x = euler_angles[0];
    float y = euler_angles[1];
    float z = euler_angles[2];

    // Assuming the angles are in radians.
    float ch = cosf(z);
    float sh = sinf(z);
    float ca = cosf(y);
    float sa = sinf(y);
    float cb = cosf(x);
    float sb = sinf(x);

    float m00, m01, m02, m10, m11, m12, m20, m21, m22;

    m00 = ch * ca;
    m01 = sh * sb - ch * sa * cb;
    m02 = ch * sa * sb + sh * cb;
    m10 = sa;
    m11 = ca * cb;
    m12 = -ca * sb;
    m20 = -sh * ca;
    m21 = sh * sa * cb + ch * sb;
    m22 = -sh * sa * sb + ch * cb;

    rotationMatrix.at<float>(0, 0) = m00;
    rotationMatrix.at<float>(0, 1) = m01;
    rotationMatrix.at<float>(0, 2) = m02;
    rotationMatrix.at<float>(1, 0) = m10;
    rotationMatrix.at<float>(1, 1) = m11;
    rotationMatrix.at<float>(1, 2) = m12;
    rotationMatrix.at<float>(2, 0) = m20;
    rotationMatrix.at<float>(2, 1) = m21;
    rotationMatrix.at<float>(2, 2) = m22;
    for (int i = 0; i < 3; i++) {
        rotationMatrix.at<float>(3, i) = 0.0f;
        rotationMatrix.at<float>(i, 3) = 0.0f;
    }
    rotationMatrix.at<float>(3, 3) = 1.0f;

    return rotationMatrix;
}

cv::Point2f EyeEstimator::undistort(cv::Point2f point) {
    float fx{intrinsic_matrix_->at<float>(cv::Point(0, 0))};
    float fy{intrinsic_matrix_->at<float>(cv::Point(1, 1))};
    float cx{intrinsic_matrix_->at<float>(cv::Point(0, 2))};
    float cy{intrinsic_matrix_->at<float>(cv::Point(1, 2))};
    float k1{(*distortion_coefficients_)[0]};
    float k2{(*distortion_coefficients_)[1]};
    cv::Point2f new_point{};
    float x{(float) (point.x + capture_offset_->width - cx) / fx};
    float y{(float) (point.y + capture_offset_->height - cy) / fy};
    float x0{x};
    float y0{y};
    for (int i = 0; i < 3; i++) {
        float r2{x * x + y * y};
        float k_inv{1 / (1 + k1 * r2 + k2 * r2 * r2)};
        x = x0 * k_inv;
        y = y0 * k_inv;
    }
    new_point.x = x * fx + cx;
    new_point.y = y * fy + cy;
    return new_point;
}

void EyeEstimator::createProjectionMatrix() {

}

cv::Vec3f EyeEstimator::ICStoEyePosition(const cv::Vec3f &point,
                                     const cv::Vec3f &cornea_centre) const {
    cv::Vec3f eye_position{};
    double t{};
    cv::Vec3f pupil_dir{-point};
    cv::normalize(pupil_dir, pupil_dir);
    bool intersected{getRaySphereIntersection(
        cv::Vec3f(0.0f), pupil_dir, cornea_centre,
        Settings::parameters.eye_params.cornea_curvature_radius, t)};

    if (intersected) {
        cv::Vec3f pupil_on_cornea{t * pupil_dir};
        cv::Vec3f nv{pupil_on_cornea - cornea_centre};
        cv::normalize(nv, nv);
        cv::Vec3d mdir{-point};
        cv::normalize(mdir, mdir);
        cv::Vec3f direction{getRefractedRay(
            mdir, nv, Settings::parameters.eye_params.cornea_refraction_index)};
        intersected = getRaySphereIntersection(
            pupil_on_cornea, direction, cornea_centre,
            Settings::parameters.eye_params.pupil_cornea_distance, t);
        if (intersected) {
            eye_position = pupil_on_cornea + t * direction;
        }
    }
    return eye_position;
}

void EyeEstimator::createVisualAxis() {
    float angles[]{Settings::parameters.user_params->alpha,
                   Settings::parameters.user_params->beta, 0};
    visual_axis_rotation_matrix_ = euler2rot(angles);
}

void EyeEstimator::getEyeFromPolynomial(cv::Point2f pupil_pix_position,
                                     cv::RotatedRect ellipse) {

    cv::Point3f nodal_point{}, eye_centre{};
    static std::vector<float> input_data{5};

    input_data[0] = pupil_pix_position.x;
    input_data[1] = ellipse.center.x;
    input_data[2] = ellipse.size.width;
    input_data[3] = ellipse.size.height;
    input_data[4] = ellipse.angle;
    eye_centre.x = eye_centre_pos_x_fit_.getEstimation(input_data);
    nodal_point.x = eye_np_pos_x_fit_.getEstimation(input_data);

    input_data[0] = pupil_pix_position.y;
    input_data[1] = ellipse.center.y;
    eye_centre.y = eye_centre_pos_y_fit_.getEstimation(input_data);
    nodal_point.y = eye_np_pos_y_fit_.getEstimation(input_data);

    input_data[0] = ellipse.center.x;
    eye_centre.z = eye_centre_pos_z_fit_.getEstimation(input_data);
    nodal_point.z = eye_np_pos_z_fit_.getEstimation(input_data);

    eye_position_ = {nodal_point, nodal_point, eye_centre};
}

} // namespace et