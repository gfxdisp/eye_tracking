#include "EyeEstimator.hpp"
#include "RayPointMinimizer.hpp"
#include "Settings.hpp"
#include "Utils.hpp"

#include <filesystem>
#include <iostream>

using KFMatD = cv::Mat_<double>;

namespace fs = std::filesystem;

namespace et {

cv::Mat EyeEstimator::visual_axis_rotation_matrix_{};

EyeEstimator::~EyeEstimator() {
    if (!solver_->empty()) {
        solver_.release();
    }
    if (!minimizer_function_.empty()) {
        minimizer_function_.release();
    }
}

void EyeEstimator::initialize(const std::string &settings_path,
                              bool kalman_filtering_enabled, int camera_id) {

    // Create a minimizer for used for finding cornea centre.
    ray_point_minimizer_ = new RayPointMinimizer();
    ray_point_minimizer_->initialize();
    minimizer_function_ =
        cv::Ptr<cv::DownhillSolver::Function>{ray_point_minimizer_};
    solver_ = cv::DownhillSolver::create();
    solver_->setFunction(minimizer_function_);
    cv::Mat step = (cv::Mat_<double>(1, 2) << 50, 50);
    solver_->setInitStep(step);

    kalman_filtering_enabled_ = kalman_filtering_enabled;

    createVisualAxis();

    auto coeffs_path = fs::path(settings_path)
        / ("ellipse_fitting_coeffs_" + std::to_string(camera_id) + ".csv");
    auto eye_features_path = fs::path(settings_path)
        / ("eye_features_" + std::to_string(camera_id) + ".csv");

    auto image_features_path = fs::path(settings_path)
        / ("image_features_" + std::to_string(camera_id) + ".csv");

    loadPolynomialCoefficients(coeffs_path, eye_features_path,
                               image_features_path);

    leds_positions_ = &Settings::parameters.leds_positions[camera_id];
    gaze_shift_ = &Settings::parameters.camera_params[camera_id].gaze_shift;
    kalman_eye_ = makeKalmanFilter(
        et::Settings::parameters.camera_params[camera_id].framerate);
    intrinsic_matrix_ =
        &Settings::parameters.camera_params[camera_id].intrinsic_matrix;
    // Convert Matlab's 1-based indexing to C++'s 0-based indexing.
    intrinsic_matrix_->at<double>(0, 2) -= 1;
    intrinsic_matrix_->at<double>(1, 2) -= 1;
    capture_offset_ =
        &Settings::parameters.camera_params[camera_id].capture_offset;

    camera_params_ = &et::Settings::parameters.camera_params[camera_id];

    createInvProjectionMatrix();
}

void EyeEstimator::getEyeFromModel(
    cv::Point2f pupil_pix_position,
    std::vector<cv::Point2f> *glint_pix_positions) {
    cv::Vec3f pupil_position{ICStoCCS(pupil_pix_position)};

    std::vector<cv::Vec3f> glint_positions{};

    // calculate planes with glints, LEDs, eye's nodal point, and camera's
    // nodal point.
    std::vector<cv::Vec3d> v1v2s{};
    for (int i = 0; i < glint_pix_positions->size(); i++) {
        cv::Vec3f v1{(*leds_positions_)[i]};
        cv::normalize(v1, v1);
        cv::Vec3f v2{ICStoCCS((*glint_pix_positions)[i])};
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
    for (int i = 0; i < v1v2s.size(); i++) {
        if (i == 1 || i == 4)
            continue;
        for (int j = i + 1; j < v1v2s.size(); j++) {
            cv::Vec3d np2c_dir{v1v2s[i].cross(v1v2s[j])};
            cv::normalize(np2c_dir, np2c_dir);
            if (np2c_dir(2) < 0) {
                np2c_dir = -np2c_dir;
            }
            avg_np2c_dir += np2c_dir;
            counter++;
        }
    }

    if (counter == 0) {
        return;
    }

    for (int i = 0; i < 3; i++) {
        avg_np2c_dir(i) = avg_np2c_dir(i) / counter;
    }

    if (ray_point_minimizer_) {
        ray_point_minimizer_->setParameters(
            avg_np2c_dir, glint_positions.data(), *leds_positions_);
    }
    cv::Mat x = (cv::Mat_<double>(1, 2) << 300, 300);
    // Finds the best candidate for cornea centre.
    solver_->minimize(x);
    double k = x.at<double>(0, 0);
    cv::Vec3f cornea_centre = avg_np2c_dir * k;

    if (kalman_filtering_enabled_) {
        kalman_eye_.correct((KFMatD(3, 1) << cornea_centre(0), cornea_centre(1),
                             cornea_centre(2)));

        cornea_centre = toPoint(kalman_eye_.predict());
    }

    cv::Vec3f pupil = calculatePositionOnPupil(pupil_position, cornea_centre);

    cv::Vec3f eye_centre{};
    if (pupil != cv::Vec3f()) {
        cv::Vec3f pupil_direction{cornea_centre - pupil};
        cv::normalize(pupil_direction, pupil_direction);
        // Eye centre lies in the same vector as cornea centre and pupil centre.
        eye_centre = pupil
            + Settings::parameters.eye_params.pupil_eye_centre_distance
                * pupil_direction;
    }
    mtx_eye_position_.lock();
    cornea_centre_ = cornea_centre;
    eye_centre_ = eye_centre;
    mtx_eye_position_.unlock();
}

void EyeEstimator::getEyeCentrePosition(cv::Vec3d &eye_centre) {
    mtx_eye_position_.lock();
    eye_centre = eye_centre_;
    mtx_eye_position_.unlock();
}

void EyeEstimator::getCorneaCurvaturePosition(cv::Vec3d &cornea_centre) {
    mtx_eye_position_.lock();
    cornea_centre = cornea_centre_;
    mtx_eye_position_.unlock();
}

void EyeEstimator::getGazeDirection(cv::Vec3f &gaze_direction) {

    cv::Vec4f homo_inv_optical_axis{};
    mtx_eye_position_.lock();
    // Inverts optical axis to make it point towards the back of the eye.
    inv_optical_axis_ = eye_centre_ - cornea_centre_;
    inv_optical_axis_ -= *gaze_shift_;
    cv::normalize(inv_optical_axis_, inv_optical_axis_);
    for (int i = 0; i < 3; i++)
        homo_inv_optical_axis(i) = inv_optical_axis_(i);
    mtx_eye_position_.unlock();
    homo_inv_optical_axis(3) = 1.0f;
    // Multiplies optical axis with a matrix to create visual axis pointing at
    // the retina.
    cv::Mat visual_axis{visual_axis_rotation_matrix_ * homo_inv_optical_axis};
    cv::Vec4f homo_gaze_direction{};
    homo_gaze_direction = visual_axis.reshape(4).at<cv::Vec4f>();
    // Inverts the visual axis to point at the viewing direction.
    for (int i = 0; i < 3; i++)
        gaze_direction(i) = -homo_gaze_direction(i) / homo_gaze_direction(3);
}

void EyeEstimator::getPupilDiameter(float &pupil_diameter) {
    mtx_eye_position_.lock();
    pupil_diameter = pupil_diameter_;
    mtx_eye_position_.unlock();
}

cv::Point2f EyeEstimator::getCorneaCurvaturePixelPosition() {
    return unproject(cornea_centre_);
}

cv::Point2f EyeEstimator::getEyeCentrePixelPosition() {
    return unproject(eye_centre_);
}

cv::Vec2f EyeEstimator::unproject(const cv::Vec3f &point) const {
    // Multiplies by intrinsic matrix to get image space coordinates.
    cv::Mat unprojected = (*intrinsic_matrix_).t() * point;
    float x = unprojected.at<float>(0);
    float y = unprojected.at<float>(1);
    float w = unprojected.at<float>(2);
    // Shifts the position to account for region-of-interest.
    return {x / w - (float) capture_offset_->width,
            y / w - (float) capture_offset_->height};
}

cv::Vec3f EyeEstimator::ICStoCCS(const cv::Point2f &point) const {

    cv::Vec4f homo_point{point.x, point.y, 0, 1};
    // Multiplies by inverted projection matrix to get camera space coordinates.
    cv::Mat p{cv::Mat(homo_point).t() * inv_projection_matrix_};
    float x = p.at<float>(0) / p.at<float>(3);
    float y = p.at<float>(1) / p.at<float>(3);
    float z = p.at<float>(2) / p.at<float>(3);

    return {x, y, z};
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
    KF.transitionMatrix = TRANSITION_MATRIX.clone();
    KF.measurementMatrix = MEASUREMENT_MATRIX.clone();
    KF.processNoiseCov = PROCESS_NOISE_COV.clone();
    KF.measurementNoiseCov = MEASUREMENT_NOISE_COV.clone();
    KF.errorCovPost = ERROR_COV_POST.clone();
    KF.statePost = STATE_POST.clone();
    // Without this line, OpenCV complains about incorrect matrix dimensions.
    KF.predict();
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
    double m_cos{(-direction).dot(normal)};
    double m_sin{nr * nr * (1 - m_cos * m_cos)};
    cv::Vec3d t{nr * direction + (nr * m_cos - std::sqrt(1 - m_sin)) * normal};
    cv::normalize(t, t);
    return t;
}

cv::Mat EyeEstimator::euler2rot(const float *euler_angles) {
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

void EyeEstimator::createInvProjectionMatrix() {

    // Projection matrix created according to the lecture notes:
    // https://www.cl.cam.ac.uk/teaching/2122/AGIP/lf_rendering.pdf
    cv::Point3f normal{0, 0, 1};
    cv::Point3f wf{0, 0, -1};
    float view_data[4][4]{{1, 0, 0, 0},
                          {0, 1, 0, 0},
                          {normal.x, normal.y, normal.z, -normal.dot(wf)},
                          {0, 0, 1, 0}};
    cv::Mat projection_matrix{4, 4, CV_32FC1, view_data};
    projection_matrix = projection_matrix.t();

    float intrinsic_matrix_arr[9];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            intrinsic_matrix_arr[i * 3 + j] =
                intrinsic_matrix_->at<float>(cv::Point(i, j));
        }
    }

    float intrinsic_data[4][4]{
        {intrinsic_matrix_arr[0], intrinsic_matrix_arr[3],
         intrinsic_matrix_arr[6], 0},
        {intrinsic_matrix_arr[1], intrinsic_matrix_arr[4],
         intrinsic_matrix_arr[7], 0},
        {0, 0, intrinsic_matrix_arr[8], 0},
        {intrinsic_matrix_arr[2], intrinsic_matrix_arr[5], 0, 1}};

    cv::Mat intrinsic_matrix{4, 4, CV_32FC1, intrinsic_data};

    inv_projection_matrix_ = projection_matrix * intrinsic_matrix;
    inv_projection_matrix_ = inv_projection_matrix_.inv();
}

cv::Vec3f
EyeEstimator::calculatePositionOnPupil(const cv::Vec3f &pupil_px_position,
                                       const cv::Vec3f &cornea_centre) {
    cv::Vec3f pupil_position{};
    double t{};
    cv::Vec3f pupil_dir{-pupil_px_position};
    cv::normalize(pupil_dir, pupil_dir);
    bool intersected{getRaySphereIntersection(
        cv::Vec3f(0.0f), pupil_dir, cornea_centre,
        Settings::parameters.eye_params.cornea_curvature_radius, t)};

    if (intersected) {
        cv::Vec3f pupil_on_cornea{t * pupil_dir};
        cv::Vec3f nv{pupil_on_cornea - cornea_centre};
        cv::normalize(nv, nv);
        cv::Vec3d m_dir{-pupil_px_position};
        cv::normalize(m_dir, m_dir);
        cv::Vec3f direction{getRefractedRay(
            m_dir, nv,
            Settings::parameters.eye_params.cornea_refraction_index)};
        intersected = getRaySphereIntersection(
            pupil_on_cornea, direction, cornea_centre,
            Settings::parameters.eye_params.pupil_cornea_distance, t);
        if (intersected) {
            pupil_position = pupil_on_cornea + t * direction;
        }
    }
    return pupil_position;
}

void EyeEstimator::createVisualAxis() {
    static float alpha = -0.0872664600610733;
    static float beta = 0.02617993950843811;
    float angles[]{alpha, beta, 0};
    visual_axis_rotation_matrix_ = euler2rot(angles);
}

void EyeEstimator::getEyeFromPolynomial(cv::Point2f pupil_pix_position,
                                        cv::RotatedRect ellipse) {

    cv::Point3f cornea_centre{}, eye_centre{};
    static std::vector<float> input_data{5};

    // Uses different sets of data for different estimated parameters.
    input_data[0] = pupil_pix_position.x;
    input_data[1] = ellipse.center.x;
    input_data[2] = ellipse.size.width;
    input_data[3] = ellipse.size.height;
    input_data[4] = ellipse.angle;
    eye_centre.x = eye_centre_pos_x_fit_.getEstimation(input_data);
    cornea_centre.x = cornea_centre_pos_x_fit_.getEstimation(input_data);

    input_data[0] = pupil_pix_position.y;
    input_data[1] = ellipse.center.y;
    eye_centre.y = eye_centre_pos_y_fit_.getEstimation(input_data);
    cornea_centre.y = cornea_centre_pos_y_fit_.getEstimation(input_data);

    input_data[0] = ellipse.center.x;
    eye_centre.z = eye_centre_pos_z_fit_.getEstimation(input_data);
    cornea_centre.z = cornea_centre_pos_z_fit_.getEstimation(input_data);

    mtx_eye_position_.lock();
    eye_centre_ = eye_centre;
    cornea_centre_ = cornea_centre;
    mtx_eye_position_.unlock();
}

void EyeEstimator::loadPolynomialCoefficients(
    const std::string &coefficients_filename,
    const std::string &eye_data_filename,
    const std::string &features_data_filename) {

    if (std::filesystem::exists(coefficients_filename)) {
        // Coefficients loaded from a file.
        auto coefficients = Utils::readFloatRowsCsv(coefficients_filename);
        cornea_centre_pos_x_fit_.setCoefficients(coefficients[0]);
        eye_centre_pos_x_fit_.setCoefficients(coefficients[1]);
        cornea_centre_pos_y_fit_.setCoefficients(coefficients[2]);
        eye_centre_pos_y_fit_.setCoefficients(coefficients[3]);
        cornea_centre_pos_z_fit_.setCoefficients(coefficients[4]);
        eye_centre_pos_z_fit_.setCoefficients(coefficients[5]);
    } else {
        // Coefficients calculated based on the Blender data.
        std::vector<std::vector<float>> poly_fit_input_data =
            Utils::readFloatColumnsCsv(features_data_filename);

        if (poly_fit_input_data.empty()) {
            return;
        }
        auto pupil_x = &poly_fit_input_data[1];
        auto pupil_y = &poly_fit_input_data[2];
        auto el_centre_x = &poly_fit_input_data[3];
        auto el_centre_y = &poly_fit_input_data[4];
        auto el_width = &poly_fit_input_data[5];
        auto el_height = &poly_fit_input_data[6];
        auto el_angle = &poly_fit_input_data[7];

        std::vector<std::vector<float>> poly_fit_output_data =
            Utils::readFloatColumnsCsv(eye_data_filename);

        if (poly_fit_output_data.empty()) {
            return;
        }

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
        cornea_centre_pos_x_fit_.fit(input_vars, nodal_x);
        eye_centre_pos_x_fit_.fit(input_vars, centre_x);

        input_vars[0] = pupil_y;
        input_vars[1] = el_centre_y;
        cornea_centre_pos_y_fit_.fit(input_vars, nodal_y);
        eye_centre_pos_y_fit_.fit(input_vars, centre_y);

        input_vars[0] = el_centre_x;
        cornea_centre_pos_z_fit_.fit(input_vars, nodal_z);
        eye_centre_pos_z_fit_.fit(input_vars, centre_z);

        std::vector<std::vector<float>> all_coefficients{};
        all_coefficients.push_back(cornea_centre_pos_x_fit_.getCoefficients());
        all_coefficients.push_back(eye_centre_pos_x_fit_.getCoefficients());
        all_coefficients.push_back(cornea_centre_pos_y_fit_.getCoefficients());
        all_coefficients.push_back(eye_centre_pos_y_fit_.getCoefficients());
        all_coefficients.push_back(cornea_centre_pos_z_fit_.getCoefficients());
        all_coefficients.push_back(eye_centre_pos_z_fit_.getCoefficients());
        Utils::writeFloatCsv(all_coefficients, coefficients_filename);
    }
}

void EyeEstimator::calculatePupilDiameter(
    cv::Point2f pupil_pix_position, int pupil_px_radius,
    const cv::Vec3f &cornea_centre_position) {
    cv::Vec3f pupil_position{ICStoCCS(pupil_pix_position)};

    cv::Vec3f pupil_right_position = ICStoCCS(
        pupil_pix_position + cv::Point2f((float) pupil_px_radius, 0.0f));

    // Estimates pupil position based on its position in the image and the cornea centre.
    cv::Vec3f pupil =
        calculatePositionOnPupil(pupil_position, cornea_centre_position);
    // Estimates position of the right part of the pupil.
    cv::Vec3f pupil_right =
        calculatePositionOnPupil(pupil_right_position, cornea_centre_position);

    if (pupil != cv::Vec3f() && pupil_right != cv::Vec3f()) {
        cv::Vec3f pupil_proj = pupil;
        cv::Vec3f pupil_right_proj = pupil_right;
        pupil_proj(2) = 0.0f;
        pupil_right_proj(2) = 0.0f;
        // Diameter is estimated as double the distance between pupil's centre
        // and pupil's right point.
        pupil_diameter_ = (float) (2 * cv::norm(pupil_proj - pupil_right_proj));
    }
}

} // namespace et