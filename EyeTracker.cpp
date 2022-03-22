#include "EyeTracker.hpp"
#include "EyePositionFunction.hpp"
#include "RayPointMinimizer.hpp"

#include <iostream>

using KFMat = cv::Mat_<double>;

namespace et {
EyeTracker::EyeTracker(SetupLayout &setup_layout, ImageProvider *image_provider)
    : setup_layout_(std::move(setup_layout)), image_provider_(image_provider) {
    ray_point_minimizer_ = new RayPointMinimizer(setup_layout_.camera_nodal_point_position);
    minimizer_function_ = cv::Ptr<cv::DownhillSolver::Function>{ray_point_minimizer_};
    solver_ = cv::Ptr<cv::DownhillSolver>{cv::DownhillSolver::create()};
    solver_->setFunction(minimizer_function_);

    double step[] = {50, 50};
    cv::Mat step_data{1, 2, CV_64F, step};

    solver_->setInitStep(step_data);
}

EyeTracker::~EyeTracker() {
    delete ray_point_minimizer_;
}
/// This code is based on Guestrin & Eizenman, pp1125-1126.
/// Algorithm:
/// - convert reflectionPixel to the WCS
/// - project it onto the presumed position of the eye (using positions.cameraEyeDistance)
/// - use (3), the fact that the light (l), the Purkyně reflection (q), the camera nodal point (o), and the
///   centre of curvature of the cornea (c) are coplanar; and (4), the law of reflection, to find a line
///   containing c.
/// - use (2), the fact that the cornea is spherical and its radius is known, to find c (find the intersection
///   of the line and the sphere, and use the Z coordinate to discriminate between multiple intersections.
/// - use (6), the fact that the point of refraction of the pupil centre also lies on the cornea, to project
///   pupilPixel onto the cornea and find this point.
/// - use (7), the coplanarity of the pupil centre (p), pupil centre's point of refraction (r), o and c, and
///   Snell's law to find a line containing p.
/// - use (9) to find p (again, finding the intersections of a sphere and the line).
/// - trace the vector p-c to find d, the centre of rotation of the eye.
void EyeTracker::calculateEyePosition(const cv::Point2f &pupil_pixel_position, cv::Point2f *glints_pixel_positions) {
    if (new_setup_layout_needed_) {
        mtx_setup_to_change_.lock();
        setup_layout_ = new_setup_layout_;
        new_setup_layout_needed_ = false;
        mtx_setup_to_change_.unlock();
    }

    std::optional<cv::Vec3d> cornea_curvature{};
    for (int i = 0; i < FeatureDetector::LED_COUNT; i++) {
        cv::Vec3d glint_position{project(glints_pixel_positions[i])};
        cv::Vec3d loqo{(setup_layout_.led_positions[i] - setup_layout_.camera_nodal_point_position)
                           .cross(glint_position - setup_layout_.camera_nodal_point_position)};
        cv::Vec3d lqoq{(setup_layout_.led_positions[i] - glint_position)
                       * cv::norm(setup_layout_.camera_nodal_point_position - glint_position)};
        cv::Vec3d oqlq{(setup_layout_.camera_nodal_point_position - glint_position)
                       * cv::norm(setup_layout_.led_positions[i] - glint_position)};
        cv::Vec3d oqlqlqoq{oqlq - lqoq};

        cv::Matx22d square_plane_matrix{loqo(0), loqo(1), oqlqlqoq(0), oqlqlqoq(1)};
        // Calculate rank
        cv::Mat1d singular_values{};
        cv::Mat left_singular_vectors{}, right_singular_vectors_t{};// Unused outputs
        cv::SVDecomp(square_plane_matrix, singular_values, left_singular_vectors, right_singular_vectors_t,
                     cv::SVD::NO_UV);
        if (cv::countNonZero(singular_values > 1e-4) < 2) {
            // The line lies in the plane z = 0.
            // Very unexpected, as the eye and the camera are facing each other on the z axis.
            // Should not occur in normal operation.
            // Can still be solved, just requires writing a lot of extra code for a situation that should never occur.
            // It will also break the logic used to distinguish between duplicate solutions of quadratics
            // (which assumes that the eye is facing roughly in the negative z direction). */
            continue;
        } else {// Far more likely
            // We now consider z = 0 and z = 1, and find two points (x, y, 0) and (x', y', 1), which define the line.
            cv::Vec2d b{loqo.dot(setup_layout_.camera_nodal_point_position), oqlqlqoq.dot(glint_position)};
            cv::Vec2d last_row{loqo(2), oqlqlqoq(2)};
            cv::Vec2d point_a_xy{}, point_b_xy{};
            if (!cv::solve(square_plane_matrix, b, point_a_xy))
                return;// z = 0
            if (!cv::solve(square_plane_matrix, b - last_row, point_b_xy))
                return;// z = 1
            cv::Vec2d direction_xy{point_b_xy - point_a_xy};
            cv::Vec3d point_a{point_a_xy(0), point_a_xy(1), 0};
            cv::Vec3d direction{direction_xy(0), direction_xy(1), 1};

            // Now we have q, the centre of a sphere of radius R on which c lies (2), and two points, pointA and pointB,
            // defining a line on which c also lies. */
            std::vector<cv::Vec3d> intersections{
                lineSphereIntersections(glint_position, EyeProperties::cornea_curvature_radius, point_a, direction)};
            std::optional<cv::Vec3d> estimated_cornea_curvature{};// c
            switch (intersections.size()) {
                case 1: estimated_cornea_curvature = intersections[0]; break;
                case 2:
                    // Take the one with the highest Z. The eye can't be pointed backwards...
                    estimated_cornea_curvature = intersections[intersections[0](2) > intersections[1](2) ? 0 : 1];
                    break;
                default: break;
            }
            if (!estimated_cornea_curvature) {
                continue;
            }
            cornea_curvature = *cornea_curvature + *estimated_cornea_curvature;
        }
    }

    if (!cornea_curvature) {
        return;
    }

    cornea_curvature = *cornea_curvature / FeatureDetector::LED_COUNT;

    // (6): We now project the pupil from the image sensor (flat) onto the cornea (spherical).
    cv::Vec3d pupil_image{ICStoWCS(pupil_pixel_position)};
    std::vector<cv::Vec3d> intersections{
        lineSphereIntersections(*cornea_curvature, EyeProperties::cornea_curvature_radius, pupil_image,
                                setup_layout_.camera_nodal_point_position - pupil_image)};
    std::optional<cv::Vec3d> pupil{};
    switch (intersections.size()) {
        case 1: pupil = intersections[0]; break;
        case 2:
            // Take the one with the lowest Z.
            pupil = intersections[intersections[0](2) < intersections[1](2) ? 0 : 1];
            break;
    }
    if (!pupil) {// No solution, but at least we have c
        eye_position_.cornea_curvature = *cornea_curvature;
        return;
    }

    // Now we find p in a somewhat similar way.
    // (7):
    cv::Vec3d roco{(*pupil - setup_layout_.camera_nodal_point_position)
                       .cross(*cornea_curvature - setup_layout_.camera_nodal_point_position)};
    // Now dot(roco, p) = dot(roco, o) - a plane containing p.
    // (8): n_1 · ‖o - r‖ / ‖(r - c) × (o - r)‖ = ‖p - r‖ / ‖(r - c) × (p - r)‖
    double n1orrcor{EyeProperties::refraction_index * cv::norm(setup_layout_.camera_nodal_point_position - *pupil)
                    / cv::norm((*pupil - *cornea_curvature).cross(setup_layout_.camera_nodal_point_position - *pupil))};
    /* This is easier to solve if we extract the angle from the remaining × product:
         * ‖p - r‖ / ‖(r - c) × (p - r)‖ = ‖p - r‖ / (‖r - c‖ · ‖p - r‖ · sin(π+θ))
         * where θ = ∠PRC, the angle between the optic axis of the eye and the
         * normal at the point of refraction of the pupil centre.
         * The ‖p - r‖ term cancels, and we are left with
         * n1orrcor * ‖r - c‖ = 1 / sin(π+θ). */
    double angle{std::asin(-1 / (n1orrcor * cv::norm(*pupil - *cornea_curvature)))};// θ
    /* We now have three constraints on p: a plane, the angle ∠PRC, and the sphere of radius K centred on c.
         * It is easy to combine the first two contraints: (7) states that p, r, o and c are coplanar.
         * Furthermore, ∠PRC is known. This allows us to construct a ray from r in the direction of p, which lies
         * in the plane of p, r, o and c.
         * Then, p lies at the intersection of this ray and the sphere.
         * p = r + μw, where w ∝ (p-r)
         * We construct w first.
         * roco is the normal of our plane, it is at 90° to w.
         * c - r is at θ to w. */
    cv::Vec3d perpendicular{(*cornea_curvature - *pupil).cross(roco)};
    perpendicular /= cv::norm(perpendicular);
    /* w = (c-r)*cos(θ) ± perpendicular*sin(θ)
         * https://math.stackexchange.com/a/2320448
         * This in itself is ambiguous: the w given by this formula can be on either side of r - c, the normal at
         * the point of refraction. However, because of how the cross products are oriented, the positive direction
         * seems to be the right one. */
    cv::Vec3d direction{(*cornea_curvature - *pupil) * std::cos(angle) / cv::norm(*cornea_curvature - *pupil)
                        + perpendicular * std::abs(std::sin(angle))};// w
    intersections = lineSphereIntersections(*cornea_curvature, EyeProperties::pupil_cornea_distance, *pupil, direction);
    switch (intersections.size()) {
        case 1: pupil = intersections[0]; break;
        case 2:
            // Take the one with the lowest Z; the pupil is in front of the lens.
            pupil = intersections[intersections[0](2) < intersections[1](2) ? 0 : 1];
            break;
        default: pupil = {}; break;
    }
    if (!pupil) {
        // No solution for p, but at least we have c.
        eye_position_.cornea_curvature = *cornea_curvature;
        return;
    }

    /* We have p and c. Together, they give the position and orientation of the eye. We now need to trace the
         * line p - c to the point d, the centre of rotation of the eye, using D, a further eye parameter not used
         * by G&E. d will be our head position.
         * NB: The eye is not actually spherical, so this may move around in unexpected ways. */

    cv::Vec3d eye_centre{
        *pupil + EyeProperties::eye_ball_radius * (*cornea_curvature - *pupil) / cv::norm(*cornea_curvature - *pupil)};

    kalman_.correct((KFMat(3, 1) << (*cornea_curvature)(0), (*cornea_curvature)(1), (*cornea_curvature)(2)));
    mtx_eye_position_.lock();
    eye_position_ = {cornea_curvature, pupil, eye_centre};
    mtx_eye_position_.unlock();
}

EyePosition EyeTracker::calculateJoined(const cv::Point2f &pupil_pixel_position, cv::Point2f *glints_pixel_positions) {
    if (new_setup_layout_needed_) {
        mtx_setup_to_change_.lock();
        setup_layout_ = new_setup_layout_;
        new_setup_layout_needed_ = false;
        mtx_setup_to_change_.unlock();
    }

    std::optional<cv::Vec3d> cornea_curvature{}, pupil{}, eye_centre{};

    cv::Vec3d v11{setup_layout_.led_positions[0] - setup_layout_.camera_nodal_point_position};
    cv::normalize(v11, v11);
    cv::Vec3d glint_position1{ICStoWCS(glints_pixel_positions[0])};
    cv::Vec3d v12{glint_position1 - setup_layout_.camera_nodal_point_position};
    cv::normalize(v12, v12);
    cv::Vec3d nn1{v11.cross(v12)};
    cv::normalize(nn1, nn1);

    cv::Vec3d v21{setup_layout_.led_positions[1] - setup_layout_.camera_nodal_point_position};
    cv::normalize(v21, v21);
    cv::Vec3d glint_position2{ICStoWCS(glints_pixel_positions[1])};
    cv::Vec3d v22{glint_position2 - setup_layout_.camera_nodal_point_position};
    cv::normalize(v22, v22);
    cv::Vec3d nn2{v21.cross(v22)};
    cv::normalize(nn2, nn2);

    cv::Vec3d bnorm{nn1.cross(nn2)};
    cv::normalize(bnorm, bnorm);
    bnorm = -bnorm;

    double data[]{200, 200};

    cv::Mat mat_data{1, 2, CV_64F, data};

    ray_point_minimizer_->setParameters(bnorm, glint_position1, setup_layout_.led_positions[0]);
    solver_->minimize(mat_data);
    cv::Vec3d c1{setup_layout_.camera_nodal_point_position + bnorm * RayPointMinimizer::kk_};
    cv::Vec3d light1_cornea_glint{RayPointMinimizer::pp_};

    ray_point_minimizer_->setParameters(bnorm, glint_position2, setup_layout_.led_positions[1]);
    solver_->minimize(mat_data);
    cv::Vec3d c2{setup_layout_.camera_nodal_point_position + bnorm * RayPointMinimizer::kk_};
    cv::Vec3d light2_cornea_glint{RayPointMinimizer::pp_};
    cornea_curvature = (c1 + c2) / 2;

    double t{};
    cv::Vec3d pupil_image_position{ICStoWCS(pupil_pixel_position)};
    cv::Vec3d pupil_dir{setup_layout_.camera_nodal_point_position - pupil_image_position};
    cv::normalize(pupil_dir, pupil_dir);
    bool intersected{getRaySphereIntersection(setup_layout_.camera_nodal_point_position, pupil_dir, *cornea_curvature,
                                              EyeProperties::cornea_curvature_radius, t)};

    if (intersected) {
        cv::Vec3d pupil_on_cornea{setup_layout_.camera_nodal_point_position + t * pupil_dir};
        cv::Vec3d nv{pupil_on_cornea - *cornea_curvature};
        cv::Vec3d mdir{setup_layout_.camera_nodal_point_position - pupil_image_position};
        cv::Vec3d direction{getRefractedRay(mdir, nv, EyeProperties::refraction_index)};
        intersected = getRaySphereIntersection(pupil_on_cornea, direction, *cornea_curvature,
                                               EyeProperties::pupil_cornea_distance, t);
        if (intersected) {
            pupil = pupil_on_cornea + t * direction;
            cv::Vec3d pupil_direction{*cornea_curvature - *pupil};
            cv::normalize(pupil_direction, pupil_direction);
            eye_centre = *pupil + EyeProperties::pupil_eye_centre_distance * pupil_direction;
        }
    }

    EyePosition eye_position{cornea_curvature, pupil, eye_centre};
    return eye_position;

    //    kalman_.correct((KFMat(3, 1) << (*cornea_curvature)(0), (*cornea_curvature)(1), (*cornea_curvature)(2)));
    //    mtx_eye_position_.lock();
    //    eye_position_ = {cornea_curvature, pupil, eye_centre};
    //    mtx_eye_position_.unlock();
}

void EyeTracker::getCorneaCurvaturePosition(cv::Vec3d &eye_centre) {
    mtx_eye_position_.lock();
    if (eye_position_) {
        eye_centre = *eye_position_.cornea_curvature;
    } else {
        eye_centre = cv::Vec3d(0.0, 0.0, 0.0);
    }
    mtx_eye_position_.unlock();
}

void EyeTracker::getGazeDirection(cv::Vec3d &gaze_direction) {
    mtx_eye_position_.lock();
    if (eye_position_) {
        gaze_direction = *eye_position_.pupil - *eye_position_.cornea_curvature;
    } else {
        gaze_direction = cv::Vec3d(1.0, 0.0, 0.0);
    }
    mtx_eye_position_.unlock();
}

cv::Point2d EyeTracker::getCorneaCurvaturePixelPosition() {
    if (eye_position_) {
        return unproject(*eye_position_.cornea_curvature);
    }
    return {0.0, 0.0};
}

void EyeTracker::setNewSetupLayout(SetupLayout &setup_layout) {
    mtx_setup_to_change_.lock();
    new_setup_layout_ = setup_layout;
    new_setup_layout_needed_ = true;
    mtx_setup_to_change_.unlock();
}

void EyeTracker::initializeKalmanFilter(float framerate) {
    kalman_ = makeKalmanFilter(framerate);
}

cv::Vec3d EyeTracker::project(const cv::Vec3d &point) const {
    return setup_layout_.camera_nodal_point_position
        + setup_layout_.camera_eye_projection_factor * (setup_layout_.camera_nodal_point_position - point);
}

cv::Point2d EyeTracker::unproject(const cv::Vec3d &point) const {
    return WCStoICS(
        (point - (1 + setup_layout_.camera_eye_projection_factor) * setup_layout_.camera_nodal_point_position)
        / -setup_layout_.camera_eye_projection_factor);
}

cv::Vec3d EyeTracker::ICStoCCS(const cv::Point2d &point) const {
    const double pixel_pitch = image_provider_->getPixelPitch();
    cv::Size2i resolution = image_provider_->getResolution();
    const double x = pixel_pitch * (point.x - resolution.width / 2.0);
    const double y = pixel_pitch * (point.y - resolution.height / 2.0);
    return {x, y, -setup_layout_.camera_lambda};
}

cv::Vec3d EyeTracker::CCStoWCS(const cv::Vec3d &point) const {
    // std::cout << point + setup_layout_.camera_nodal_point_position << std::endl;
    return point + setup_layout_.camera_nodal_point_position;
}

cv::Vec3d EyeTracker::WCStoCCS(const cv::Vec3d &point) const {
    cv::Vec3d ret{};
    /* Warning: return value of cv::solve is not checked;
         * if there is no solution, ret won't be set by the line below! */
    cv::solve(cv::Matx33d::eye(), point - setup_layout_.camera_nodal_point_position, ret);
    return ret;
}

cv::Point2d EyeTracker::CCStoICS(cv::Vec3d point) const {
    const double pixel_pitch = image_provider_->getPixelPitch();
    cv::Size2i resolution = image_provider_->getResolution();
    return static_cast<cv::Point2d>(resolution) / 2 + cv::Point2d(point(0), point(1)) / pixel_pitch;
}

std::vector<cv::Vec3d> EyeTracker::lineSphereIntersections(const cv::Vec3d &sphere_centre, float radius,
                                                           const cv::Vec3d &line_point,
                                                           const cv::Vec3d &line_direction) {
    /* We are looking for points of the form line_point + k*line_direction, which are also radius away
         * from sphere_centre. This can be expressed as a quadratic in k: ak² + bk + c = radius². */
    const double a{cv::norm(line_direction, cv::NORM_L2SQR)};
    const double b{2 * line_direction.dot(line_point - sphere_centre)};
    const double c{cv::norm(line_point, cv::NORM_L2SQR) + cv::norm(sphere_centre, cv::NORM_L2SQR)
                   - 2 * line_point.dot(sphere_centre)};
    const double DISCRIMINANT{std::pow(b, 2) - 4 * a * (c - std::pow(radius, 2))};
    if (std::abs(DISCRIMINANT) < 1e-6) {
        return {line_point - line_direction * b / (2 * a)};// One solution
    } else if (DISCRIMINANT < 0) {
        return {};// No solutions
    } else {      // Two solutions
        const double sqrtDISCRIMINANT{std::sqrt(DISCRIMINANT)};
        return {line_point + line_direction * (-b + sqrtDISCRIMINANT) / (2 * a),
                line_point + line_direction * (-b - sqrtDISCRIMINANT) / (2 * a)};
    }
}

cv::KalmanFilter EyeTracker::makeKalmanFilter(float framerate) {
    constexpr static double VELOCITY_DECAY = 0.9;
    const static cv::Mat TRANSITION_MATRIX =
        (KFMat(6, 6) << 1, 0, 0, 1.0f / framerate, 0, 0, 0, 1, 0, 0, 1.0f / framerate, 0, 0, 0, 1, 0, 0,
         1.0f / framerate, 0, 0, 0, VELOCITY_DECAY, 0, 0, 0, 0, 0, 0, VELOCITY_DECAY, 0, 0, 0, 0, 0, 0, VELOCITY_DECAY);
    const static cv::Mat MEASUREMENT_MATRIX = cv::Mat::eye(3, 6, CV_64F);
    const static cv::Mat PROCESS_NOISE_COV = cv::Mat::eye(6, 6, CV_64F) * 100;
    const static cv::Mat MEASUREMENT_NOISE_COV = cv::Mat::eye(3, 3, CV_64F) * 50;
    const static cv::Mat ERROR_COV_POST = cv::Mat::eye(6, 6, CV_64F) * 0.1;
    const static cv::Mat STATE_POST = cv::Mat::zeros(6, 1, CV_64F);

    cv::KalmanFilter KF(6, 3);
    // clone() is needed as, otherwise, the matrices will be used by reference, and all the filters will be the same
    KF.transitionMatrix = TRANSITION_MATRIX.clone();
    KF.measurementMatrix = MEASUREMENT_MATRIX.clone();
    KF.processNoiseCov = PROCESS_NOISE_COV.clone();
    KF.measurementNoiseCov = MEASUREMENT_NOISE_COV.clone();
    KF.errorCovPost = ERROR_COV_POST.clone();
    KF.statePost = STATE_POST.clone();
    KF.predict();// Without this line, OpenCV complains about incorrect matrix dimensions
    return KF;
}
bool EyeTracker::getRaySphereIntersection(const cv::Vec3d &ray_pos, const cv::Vec3d &ray_dir,
                                          const cv::Vec3d &sphere_pos, double sphere_radius, double &t) {
    double A{ray_dir.dot(ray_dir)};
    cv::Vec3d v{ray_pos - sphere_pos};
    double B{2 * v.dot(ray_dir)};
    double C{v.dot(v) - sphere_radius * sphere_radius};
    double delta{B * B - 4 * A * C};
    if (delta > 0) {
        double t1{-B - std::sqrt(delta) / (2 * A)};
        double t2{-B + std::sqrt(delta) / (2 * A)};
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
cv::Vec3d EyeTracker::getRefractedRay(const cv::Vec3d &direction, const cv::Vec3d &normal, double refraction_index) {
    double nr{1 / refraction_index};
    double mcos{(-direction).dot(normal)};
    double msin{nr * nr * (1 - mcos * mcos)};
    cv::Vec3d t{nr * direction + (nr * mcos - std::sqrt(1 - msin)) * normal};
    cv::normalize(t, t);
    return t;
}
void EyeTracker::findBestCameraParameters(const cv::Vec3d &ground_truth, const cv::Point2f &pupil_pixel_position,
                                          cv::Point2f *glints_pixel_positions) {
    cv::Ptr<cv::DownhillSolver::Function> minimizer_function = cv::Ptr<cv::DownhillSolver::Function>{
        new EyePositionFunction(this, pupil_pixel_position, glints_pixel_positions, ground_truth)};
    cv::Ptr<cv::DownhillSolver> solver = cv::Ptr<cv::DownhillSolver>{cv::DownhillSolver::create()};
    solver->setFunction(minimizer_function);

    double step[10];
    for (double &i : step) {
        i = 100.0f;
    }
    cv::Mat step_data{1, 9, CV_64F, step};

    solver->setInitStep(step_data);
    double data[10];
    for (int i = 0; i < 3; i++) {
        data[i] = setup_layout_.camera_nodal_point_position[i];
        data[i + 3] = setup_layout_.led_positions[0][i];
        data[i + 6] = setup_layout_.led_positions[1][i];
    }
    data[9] = setup_layout_.camera_lambda;

    cv::Mat mat_data{1, 9, CV_64F, data};
    solver->minimize(mat_data);
    setNewSetupLayout(EyePositionFunction::setup_layout_);

    EyePosition eye_position = calculateJoined(pupil_pixel_position, glints_pixel_positions);
    std::cout << EyePositionFunction::setup_layout_.camera_nodal_point_position
              << EyePositionFunction::setup_layout_.led_positions[0]
              << EyePositionFunction::setup_layout_.led_positions[1] << *eye_position.cornea_curvature << "\n";
}
}// namespace et