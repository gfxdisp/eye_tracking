#include "EyeTracker.hpp"
#include "RayPointMinimizer.hpp"
#include "Settings.hpp"

#include <chrono>
#include <iostream>
#include <thread>
#include <fstream>

using KFMat = cv::Mat_<double>;

namespace et {


cv::Mat EyeTracker::visual_axis_rotation_matrix_{};

EyeTracker::EyeTracker(ImageProvider *image_provider)
    : image_provider_(image_provider) {
    ray_point_minimizer_ = new RayPointMinimizer();
    minimizer_function_ =
        cv::Ptr<cv::DownhillSolver::Function>{ray_point_minimizer_};
    solver_ = cv::DownhillSolver::create();
    solver_->setFunction(minimizer_function_);
    cv::Mat step = (cv::Mat_<double>(1, 2) << 50, 50);
    solver_->setInitStep(step);

    createProjectionMatrix();
    createVisualAxis();
}

EyeTracker::~EyeTracker() {
    solver_.release();
    minimizer_function_.release();
}

void EyeTracker::calculateJoined(cv::Point2f pupil_pix_position,
                                 std::vector<cv::Point2f> &glint_pix_positions,
                                 float pupil_radius) {
    std::optional<cv::Vec3f> cornea_curvature{}, pupil{}, eye_centre{};

//    std::ofstream output{"output.txt", std::ios_base::app};
//    output << pupil_pix_position.x << ", " << pupil_pix_position.y;
//    for (auto & glint_pix_position : glint_pix_positions) {
//        output << ", " << glint_pix_position.x << ", " << glint_pix_position.y;
//    }
//    output << ", 245.935319 171.827663 847.168401\n";
//    output.close();

    pupil_pix_position_ = pupil_pix_position;
    glint_pix_positions_ = glint_pix_positions;

    cv::Vec3f pupil_position{ICStoCCS(undistort(pupil_pix_position_))};

    cv::Vec3f pupil_top_position = ICStoCCS(
        undistort(pupil_pix_position_ + cv::Point2f(pupil_radius, 0.0f)));

    std::vector<cv::Vec3f> glint_positions{};

    std::vector<cv::Vec3d> v1v2s{};
    for (int i = 0; i < glint_pix_positions_.size(); i++) {
        cv::Vec3f v1{Settings::parameters.leds_positions[i]};
        cv::normalize(v1, v1);
        cv::Vec3f v2{ICStoCCS(undistort(glint_pix_positions_[i]))};
        glint_positions.push_back(v2);
        cv::normalize(v2, v2);
        cv::Vec3d v1v2{v1.cross(v2)};
        cv::normalize(v1v2, v1v2);
        v1v2s.push_back(v1v2);
    }

    cv::Vec3d avg_bnorm{};
    int counter{0};
    for (int i = 0; i < v1v2s.size(); i++) {
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
                                        Settings::parameters.leds_positions);
    cv::Mat x = (cv::Mat_<double>(1, 2) << 400, 400);
    solver_->minimize(x);
    double k = x.at<double>(0, 0);
    cornea_curvature = avg_bnorm * k;

    kalman_.correct((KFMat(3, 1) << (*cornea_curvature)(0),
                     (*cornea_curvature)(1), (*cornea_curvature)(2)));

    cornea_curvature = toPoint(kalman_.predict());

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

void EyeTracker::getCorneaCurvaturePosition(cv::Vec3d &eye_centre) {
    mtx_eye_position_.lock();
    eye_centre = *eye_position_.cornea_curvature;
    mtx_eye_position_.unlock();
}

void EyeTracker::getGazeDirection(cv::Vec3f &gaze_direction) {
    cv::Vec3f inv_optical_axis{};
    mtx_eye_position_.lock();
    if (eye_position_) {
        inv_optical_axis =
            *eye_position_.cornea_curvature - *eye_position_.pupil;
    } else {
        inv_optical_axis = cv::Vec3f(1.0, 0.0, 0.0);
    }
    mtx_eye_position_.unlock();
    cv::normalize(inv_optical_axis, inv_optical_axis);
    cv::Vec4f homo_inv_optical_axis{};
    for (int i = 0; i < 3; i++)
        homo_inv_optical_axis(i) = inv_optical_axis(i);
    homo_inv_optical_axis(3) = 1.0f;
    cv::Mat visual_axis{visual_axis_rotation_matrix_ * homo_inv_optical_axis};
    cv::Vec4f homo_gaze_direction{};
    homo_gaze_direction = visual_axis.reshape(4).at<cv::Vec4f>();
    for (int i = 0; i < 3; i++)
        gaze_direction(i) = -homo_gaze_direction(i) / homo_gaze_direction(3);
}

void EyeTracker::getPupilDiameter(float &pupil_diameter) {
    mtx_eye_position_.lock();
    pupil_diameter = pupil_diameter_;
    mtx_eye_position_.unlock();
}

void EyeTracker::getEyeData(EyeData &eye_data) {
    eye_data.pupil_pix_position = pupil_pix_position_;
    eye_data.glint_pix_positions = glint_pix_positions_;
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

cv::Point2f EyeTracker::getCorneaCurvaturePixelPosition() {
    if (eye_position_) {
        return unproject(*eye_position_.cornea_curvature);
    }
    return {0.0, 0.0};
}

cv::Point2f EyeTracker::getEyeCentrePixelPosition() {
    if (eye_position_) {
        return unproject(*eye_position_.eye_centre);
    }
    return {0.0, 0.0};
}

void EyeTracker::initializeKalmanFilter(float framerate) {
    kalman_ = makeKalmanFilter(framerate);
}

cv::Vec3f EyeTracker::project(const cv::Vec3f &point) const {
    return point;
}

cv::Vec2f EyeTracker::unproject(const cv::Vec3f &point) const {
    cv::Mat unprojected =
        Settings::parameters.camera_params.intrinsic_matrix.t() * point;
    cv::Size2i offset{Settings::parameters.camera_params.capture_offset};
    float x = unprojected.at<float>(0);
    float y = unprojected.at<float>(1);
    float w = unprojected.at<float>(2);
    return {x / w - offset.width, y / w - offset.height};
}

cv::Vec3f EyeTracker::ICStoCCS(const cv::Point2f &point) const {

    cv::Vec4f expanded_point{point.x, point.y, 0, 1};
    cv::Mat p{cv::Mat(expanded_point).t() * full_projection_matrix_};
    float x = p.at<float>(0) / p.at<float>(3);
    float y = p.at<float>(1) / p.at<float>(3);
    float z = p.at<float>(2) / p.at<float>(3);

    return {x, y, z};
}

cv::Vec3f EyeTracker::CCStoWCS(const cv::Vec3f &point) const {
    return point;
}

cv::Vec3f EyeTracker::WCStoCCS(const cv::Vec3f &point) const {
    return point;
}

cv::Vec2f EyeTracker::CCStoICS(cv::Vec3f point) const {
    const double pixel_pitch = 0;
    cv::Size2i resolution = et::Settings::parameters.camera_params.dimensions;
    return static_cast<cv::Point2f>(resolution) / 2
        + cv::Point2f(point(0), point(1)) / pixel_pitch;
}

std::vector<cv::Vec3d>
EyeTracker::lineSphereIntersections(const cv::Vec3d &sphere_centre,
                                    float radius, const cv::Vec3d &line_point,
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

cv::KalmanFilter EyeTracker::makeKalmanFilter(float framerate) {
    constexpr static double VELOCITY_DECAY = 1.0;
    const static cv::Mat TRANSITION_MATRIX =
        (KFMat(6, 6) << 1, 0, 0, 1.0f / framerate, 0, 0, 0, 1, 0, 0,
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

bool EyeTracker::getRaySphereIntersection(const cv::Vec3f &ray_pos,
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

cv::Vec3d EyeTracker::getRefractedRay(const cv::Vec3d &direction,
                                      const cv::Vec3d &normal,
                                      double refraction_index) {
    double nr{1 / refraction_index};
    double mcos{(-direction).dot(normal)};
    double msin{nr * nr * (1 - mcos * mcos)};
    cv::Vec3d t{nr * direction + (nr * mcos - std::sqrt(1 - msin)) * normal};
    cv::normalize(t, t);
    return t;
}

bool EyeTracker::isSetupUpdated() {
    return setup_updated_;
}

cv::Mat EyeTracker::euler2rot(float *euler_angles) {
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

cv::Point2f EyeTracker::undistort(cv::Point2f point) {
    float fx{Settings::parameters.camera_params.intrinsic_matrix.at<float>(
        cv::Point(0, 0))};
    float fy{Settings::parameters.camera_params.intrinsic_matrix.at<float>(
        cv::Point(1, 1))};
    float cx{Settings::parameters.camera_params.intrinsic_matrix.at<float>(
        cv::Point(0, 2))};
    float cy{Settings::parameters.camera_params.intrinsic_matrix.at<float>(
        cv::Point(1, 2))};
    float k1{Settings::parameters.camera_params.distortion_coefficients[0]};
    float k2{Settings::parameters.camera_params.distortion_coefficients[1]};
    cv::Size2i offset{Settings::parameters.camera_params.capture_offset};
    cv::Point2f new_point{};
    float x{(float) (point.x + offset.width - cx) / fx};
    float y{(float) (point.y + offset.height - cy) / fy};
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

void EyeTracker::createProjectionMatrix() {
    cv::Point3f normal{0, 0, 1};
    cv::Point3f wf{0, 0, -1};
    float view_data[4][4]{{1, 0, 0, 0},
                          {0, 1, 0, 0},
                          {normal.x, normal.y, normal.z, -normal.dot(wf)},
                          {0, 0, 1, 0}};
    cv::Mat projection_matrix{4, 4, CV_32FC1, view_data};
    projection_matrix = projection_matrix.t();

    float intr_mtx[9];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            intr_mtx[i * 3 + j] =
                Settings::parameters.camera_params.intrinsic_matrix.at<float>(
                    cv::Point(i, j));
        }
    }

    float intrinsic_data[4][4]{{intr_mtx[0], intr_mtx[3], intr_mtx[6], 0},
                               {intr_mtx[1], intr_mtx[4], intr_mtx[7], 0},
                               {0, 0, intr_mtx[8], 0},
                               {intr_mtx[2], intr_mtx[5], 0, 1}};

    cv::Mat intrinsic_matrix{4, 4, CV_32FC1, intrinsic_data};

    full_projection_matrix_ = projection_matrix * intrinsic_matrix;
    full_projection_matrix_ = full_projection_matrix_.inv();
}

cv::Vec3f EyeTracker::ICStoEyePosition(const cv::Vec3f &point,
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

void EyeTracker::createVisualAxis() {
    float angles[]{Settings::parameters.user_params->alpha,
                   Settings::parameters.user_params->beta, 0};
    visual_axis_rotation_matrix_ = euler2rot(angles);
}

} // namespace et