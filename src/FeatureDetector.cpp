#include "FeatureDetector.hpp"
#include "Utils.hpp"

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>

#include <cmath>
#include <filesystem>
#include <iostream>

using KFMatD = cv::Mat_<double>;
using KFMatF = cv::Mat_<float>;

namespace fs = std::filesystem;

namespace et {

void FeatureDetector::initialize(const std::string &settings_path, bool kalman_filtering_enabled,
                                 bool template_matching_enabled, bool distorted, int camera_id) {
    pupil_kalman_ = makePxKalmanFilter(et::Settings::parameters.camera_params[camera_id].region_of_interest,
                                       et::Settings::parameters.camera_params[camera_id].framerate);
    glints_kalman_ = makePxKalmanFilter(et::Settings::parameters.camera_params[camera_id].region_of_interest,
                                        et::Settings::parameters.camera_params[camera_id].framerate);
    pupil_radius_kalman_ = makeRadiusKalmanFilter(et::Settings::parameters.detection_params[camera_id].min_pupil_radius,
                                                  et::Settings::parameters.detection_params[camera_id].max_pupil_radius,
                                                  et::Settings::parameters.camera_params[camera_id].framerate);
    glint_ellipse_kalman_ =
        makeEllipseKalmanFilter(et::Settings::parameters.camera_params[camera_id].region_of_interest,
                                et::Settings::parameters.camera_params[camera_id].framerate);
    glint_locations_.resize(Settings::parameters.leds_positions[camera_id].size());
    glint_locations_undistorted_.resize(Settings::parameters.leds_positions[camera_id].size());

    region_of_interest_ = &Settings::parameters.camera_params[camera_id].region_of_interest;
    pupil_threshold_ = &Settings::parameters.user_params[camera_id]->pupil_threshold;
    glint_threshold_ = &Settings::parameters.user_params[camera_id]->glint_threshold;
    pupil_search_centre_ = &Settings::parameters.detection_params[camera_id].pupil_search_centre;
    pupil_search_radius_ = &Settings::parameters.detection_params[camera_id].pupil_search_radius;
    min_pupil_radius_ = &Settings::parameters.detection_params[camera_id].min_pupil_radius;
    max_pupil_radius_ = &Settings::parameters.detection_params[camera_id].max_pupil_radius;
    min_glint_radius_ = &Settings::parameters.detection_params[camera_id].min_glint_radius;
    max_glint_radius_ = &Settings::parameters.detection_params[camera_id].max_glint_radius;

    intrinsic_matrix_ = &Settings::parameters.camera_params[camera_id].intrinsic_matrix;
    capture_offset_ = &Settings::parameters.camera_params[camera_id].capture_offset;
    distortion_coefficients_ = &Settings::parameters.camera_params[camera_id].distortion_coefficients;

    detection_params_ = &Settings::parameters.detection_params[camera_id];

    kalman_filtering_enabled_ = kalman_filtering_enabled;
    template_matching_enabled_ = template_matching_enabled;
    distorted_ = distorted;

    bound_ellipse_semi_major_ = &Settings::parameters.detection_params[camera_id].max_hor_glint_pupil_distance;
    bound_ellipse_semi_minor_ = &Settings::parameters.detection_params[camera_id].max_vert_glint_pupil_distance;

    bayes_minimizer_ = new BayesMinimizer();
    bayes_minimizer_func_ = cv::Ptr<cv::DownhillSolver::Function>{bayes_minimizer_};
    bayes_solver_ = cv::DownhillSolver::create();
    bayes_solver_->setFunction(bayes_minimizer_func_);
    cv::Mat step = (cv::Mat_<double>(1, 3) << 100, 100, 100);
    bayes_solver_->setInitStep(step);

    auto template_path = fs::path(settings_path) / ("template_" + std::to_string(camera_id) + ".png");

    cv::Mat glints_template_cpu = cv::imread(template_path, cv::IMREAD_GRAYSCALE);
    glints_template_.upload(glints_template_cpu);
    template_matcher_ = cv::cuda::createTemplateMatching(CV_8UC1, cv::TM_CCOEFF);
    template_crop_ = (KFMatF(2, 3) << 1, 0, glints_template_.cols / 2, 0, 1, glints_template_.rows / 2);

    led_count_ = (int) Settings::parameters.leds_positions[camera_id].size();
}

cv::Point2f FeatureDetector::getPupil() {
    return pupil_location_undistorted_;
}

cv::Point2f FeatureDetector::getDistortedPupil() {
    return pupil_location_;
}

void FeatureDetector::getPupil(cv::Point2f &pupil) {
    mtx_features_.lock();
    pupil = pupil_location_undistorted_;
    mtx_features_.unlock();
}

void FeatureDetector::getPupilFiltered(cv::Point2f &pupil) {
    mtx_features_.lock();
    pupil = pupil_location_filtered_;
    mtx_features_.unlock();
}

int FeatureDetector::getPupilRadius() const {
    return pupil_radius_undistorted_;
}

int FeatureDetector::getDistortedPupilRadius() const {
    return pupil_radius_;
}

std::vector<cv::Point2f> *FeatureDetector::getGlints() {
    return &glint_locations_undistorted_;
}

std::vector<cv::Point2f> *FeatureDetector::getDistortedGlints() {
    return &glint_locations_;
}

void FeatureDetector::preprocessImage(const ImageToProcess &image) {
    gpu_image_.upload(image.pupil);

    cv::cuda::threshold(gpu_image_, pupil_thresholded_image_gpu_, *pupil_threshold_, 255, cv::THRESH_BINARY_INV);

    gpu_image_.upload(image.glints);

    if (template_matching_enabled_) {
        // Finds the correlation of the glint template to every area in the image.
        template_matcher_->match(gpu_image_, glints_template_, glints_thresholded_image_gpu_);

        cv::cuda::threshold(glints_thresholded_image_gpu_, glints_thresholded_image_gpu_, *glint_threshold_ * 2e3, 255,
                            cv::THRESH_BINARY);
        glints_thresholded_image_gpu_.convertTo(glints_thresholded_image_gpu_, CV_8UC1);
    } else {
        cv::cuda::threshold(gpu_image_, glints_thresholded_image_gpu_, *glint_threshold_, 255, cv::THRESH_BINARY);
    }

    pupil_thresholded_image_gpu_.download(pupil_thresholded_image_);
    glints_thresholded_image_gpu_.download(glints_thresholded_image_);

    if (template_matching_enabled_) {
        // Moves the correlation map so that it is centered in the image.
        cv::warpAffine(glints_thresholded_image_, glints_thresholded_image_, template_crop_, *region_of_interest_);
    }
}

bool FeatureDetector::findPupil() {
    cv::findContours(pupil_thresholded_image_, contours_, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::Point2f best_centre{};
    float best_radius{};
    float best_rating{0};

    cv::Point2f image_centre{*pupil_search_centre_};
    auto max_distance = (float) (*pupil_search_radius_);

    // All the contours are analyzed
    for (const std::vector<cv::Point> &contour : contours_) {
        cv::Point2f centre;
        float radius;

        cv::Rect bound_rect = cv::boundingRect(contour);
        centre = 0.5 * (bound_rect.tl() + bound_rect.br());
        radius = (float) std::max(bound_rect.width, bound_rect.height) / 2;

        // Contours forming too small or too large pupils are rejected.
        if (radius < *min_pupil_radius_ or radius > *max_pupil_radius_)
            continue;

        // Contours outside the hole in the view piece are rejected.
        float distance = euclideanDistance(centre, image_centre);
        if (distance > max_distance)
            continue;

        // Contours are rated according to their similarity to the circle.
        const float contour_area = static_cast<float>(cv::contourArea(contour));
        const float circle_area = 3.1415926f * powf(radius, 2);
        float rating = contour_area / circle_area;
        if (rating >= best_rating) {
            best_centre = centre;
            best_rating = rating;
            best_radius = radius;
        }
    }

    if (best_rating == 0) {
        return false;
    }

    if (kalman_filtering_enabled_) {
        pupil_kalman_.correct((KFMatD(2, 1) << best_centre.x, best_centre.y));
        pupil_radius_kalman_.correct((KFMatD(1, 1) << best_radius));
        pupil_location_ = toPoint(pupil_kalman_.predict());
        pupil_radius_ = (int) toValue(pupil_radius_kalman_.predict());
        //        pupil_location_ = best_centre;
        //        pupil_radius_ = (int) best_radius;
    } else {
        pupil_location_ = best_centre;
        pupil_radius_ = (int) best_radius;
    }

    auto centre_undistorted = undistort(best_centre);
    cv::Point2f pupil_left_side = best_centre;
    pupil_left_side.x -= best_radius;
    cv::Point2f pupil_right_side = best_centre;
    pupil_right_side.x += best_radius;

    auto left_side_undistorted = undistort(pupil_left_side);
    auto right_side_undistorted = undistort(pupil_right_side);
    float radius_undistorted = (right_side_undistorted.x - left_side_undistorted.x) * 0.5f;
    mtx_features_.lock();
    pupil_location_undistorted_ = centre_undistorted;
    pupil_radius_undistorted_ = (int) radius_undistorted;
    mtx_features_.unlock();

    return true;
}

bool FeatureDetector::findGlints() {

    cv::findContours(glints_thresholded_image_, contours_, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Point2f image_centre{*pupil_search_centre_};
    auto max_distance = (float) (*pupil_search_radius_);

    std::vector<GlintCandidate> glint_candidates{};
    glint_candidates.reserve(contours_.size());

    // All the contours are analyzed
    for (const std::vector<cv::Point> &contour : contours_) {
        cv::Point2f centre;
        float radius;
        cv::Rect bound_rect = cv::boundingRect(contour);
        centre = 0.5 * (bound_rect.tl() + bound_rect.br());
        radius = (float) std::max(bound_rect.width, bound_rect.height) / 2;

        // Contours forming too small or too large glints are rejected.
        if (radius > *max_glint_radius_ || radius < *min_glint_radius_)
            continue;

        // Contours outside the hole in the view piece are rejected.
        float distance = euclideanDistance(centre, image_centre);
        if (distance > max_distance)
            continue;

        // Contours outside the expected eye area are rejected.
        if (!isInEllipse(centre, pupil_location_)) {
            continue;
        }

        // Contours are rated according to their similarity to the circle.
        const float contour_area = static_cast<float>(cv::contourArea(contour));
        const float circle_area = 3.1415926f * powf(radius, 2);
        float rating = contour_area / circle_area;
        GlintCandidate glint_candidate{};
        glint_candidate.location = centre;
        glint_candidate.rating = rating;
        glint_candidate.right_neighbour = nullptr;
        glint_candidate.bottom_neighbour = nullptr;
        glint_candidates.push_back(glint_candidate);
    }

    if (glint_candidates.empty()) {
        return false;
    }

    // Glints are sorted according to their rating.
    std::sort(glint_candidates.begin(), glint_candidates.end(), [](const GlintCandidate &a, const GlintCandidate &b) {
        return a.rating > b.rating;
    });

    // Determines neighbouring glints of every glint
    findPotentialNeighbours(glint_candidates);
    // Depending on the available neighbours, determines the potential glint
    // position in the 3x2 reflection grid.
    determineGlintTypes(glint_candidates);

    for (auto &found_glint : found_glints_) {
        found_glint = false;
    }

    cv::Point2f distance{};
    bool glint_found{false};
    // Selects final set of candidates. Glints are added as correct from the highest
    // rating to the lowest along with their neighbours. If any glint in the grid
    // has already been chosen, any further changes to it are rejected.
    for (int i = 0; i < 2; i++) {
        for (auto &glint : glint_candidates) {
            // In the first iteration (i = 0) we check only for centre LEDs, as they are the most reliable
            if (i == 0 && glint.glint_type != GlintType::UpperCentre && glint.glint_type != GlintType::BottomCentre) {
                continue;
            }
            if (glint.glint_type == GlintType::Unknown) {
                continue;
            }

            if (glint_found) {
                if (found_glints_[glint.glint_type] && selected_glints_[glint.glint_type].rating >= glint.rating) {
                    continue;
                }

                if (glint.glint_type % 3 != 0 && !isLeftNeighbour(glint, selected_glints_[glint.glint_type - 1])) {
                    continue;
                }
                if (glint.glint_type % 3 != 2 && !isRightNeighbour(glint, selected_glints_[glint.glint_type + 1])) {
                    continue;
                }
                if (glint.glint_type / 3 == 0 && !isBottomNeighbour(glint, selected_glints_[glint.glint_type + 3])) {
                    continue;
                }
                if (glint.glint_type / 3 == 1 && !isUpperNeighbour(glint, selected_glints_[glint.glint_type - 3])) {
                    continue;
                }
            }

            glint_found = true;
            selected_glints_[glint.glint_type] = glint;
            found_glints_[glint.glint_type] = true;
            // Adds glint neighbours as well.
            identifyNeighbours(&glint);
            // Approximates the positions of all glints that have not been found yet.
            approximatePositions();
        }
    }

    int found_glints_count{0};
    for (auto &found_glint : found_glints_) {
        if (found_glint) {
            found_glints_count++;
        }
    }

    if (found_glints_count == 0) {
        return false;
    }

    // If left most glints are not detected, all other glints are moved
    // one spot to the left. This is needed due to lower brightness of left most
    // glints.
    if (!found_glints_[0] && !found_glints_[3]) {
        selected_glints_[0] = selected_glints_[1];
        selected_glints_[1] = selected_glints_[2];
        selected_glints_[3] = selected_glints_[4];
        selected_glints_[4] = selected_glints_[5];
        found_glints_[2] = false;
        found_glints_[5] = false;
        approximatePositions();
    }

    cv::Point2f glints_centre{};
    cv::Point2f new_glints_centre{};

    for (auto &selected_glint : selected_glints_) {
        glints_centre += selected_glint.location;
    }

    // Calculates the glints centre.
    glints_centre.x /= (float) led_count_;
    glints_centre.y /= (float) led_count_;

    if (kalman_filtering_enabled_) {
        glints_kalman_.correct((KFMatD(2, 1) << glints_centre.x, glints_centre.y));
        new_glints_centre = toPoint(glints_kalman_.predict());
        for (int i = 0; i < led_count_; i++) {
            glint_locations_[i] = selected_glints_[i].location + (new_glints_centre - glints_centre);
        }
    } else {
        for (int i = 0; i < led_count_; i++) {
            glint_locations_[i] = selected_glints_[i].location;
        }
    }

    mtx_features_.lock();
    for (int i = 0; i < led_count_; i++) {
        glint_locations_undistorted_[i] = undistort(glint_locations_[i]);
    }
    glint_represent_undistorted_ = glint_locations_undistorted_[0];
    mtx_features_.unlock();

    return true;
}

bool FeatureDetector::findEllipse() {
    cv::findContours(glints_thresholded_image_, contours_, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Point2f> ellipse_points{};
    cv::Point2f im_centre{*region_of_interest_ / 2};

    static unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    // Calculates the central point of each contour.
    for (const auto &contour : contours_) {
        cv::Point2d mean_point{};
        for (const auto &point : contour) {
            mean_point.x += point.x;
            mean_point.y += point.y;
        }
        mean_point.x /= (double) contour.size();
        mean_point.y /= (double) contour.size();
        ellipse_points.push_back(mean_point);
    }

    // Limits the number of glints to 20 which are the closest to the previously
    // estimated circle.
    std::sort(ellipse_points.begin(), ellipse_points.end(), [this](auto const &a, auto const &b) {
        float distance_a = euclideanDistance(a, circle_centre_);
        float distance_b = euclideanDistance(b, circle_centre_);
        return distance_a < distance_b;
    });

    ellipse_points.resize(std::min((int) ellipse_points.size(), 20));
    if (ellipse_points.size() < 5) {
        return false;
    }

    static int frame_counter = 0;
    frame_counter++;

    std::string bitmask(3, 1);
    std::vector<cv::Point2f> circle_points{};
    circle_points.resize(3);
    int best_counter = 0;
    cv::Point2d best_circle_centre{};
    double best_circle_radius{};
    bitmask.resize(ellipse_points.size() - 3, 0);
    cv::Point2d ellipse_centre{};
    double ellipse_radius;
    if (kalman_filtering_enabled_) {
        // Loop on every possible triplet of glints.
        do {
            int counter = 0;
            for (int i = 0; counter < 3; i++) {
                if (bitmask[i]) {
                    circle_points[counter] = ellipse_points[i];
                    counter++;
                }
            }
            bayes_minimizer_->setParameters(circle_points, circle_centre_, circle_radius_);

            cv::Mat x = (cv::Mat_<double>(1, 3) << im_centre.x, im_centre.y, circle_radius_);
            // Find the most likely position of the circle based on the previous
            // circle position and triplet of glints that should lie on it.
            bayes_solver_->minimize(x);
            ellipse_centre.x = x.at<double>(0, 0);
            ellipse_centre.y = x.at<double>(0, 1);
            ellipse_radius = x.at<double>(0, 2);

            counter = 0;
            // Counts all glints that lie close to the circle
            for (auto &ellipse_point : ellipse_points) {
                double value{0.0};
                value += (ellipse_centre.x - ellipse_point.x) * (ellipse_centre.x - ellipse_point.x);
                value += (ellipse_centre.y - ellipse_point.y) * (ellipse_centre.y - ellipse_point.y);
                if (std::abs(std::sqrt(value) - ellipse_radius) <= 3.0) {
                    counter++;
                }
            }
            // Finds the circle which contains the most glints.
            if (counter > best_counter) {
                best_counter = counter;
                best_circle_centre = ellipse_centre;
                best_circle_radius = ellipse_radius;
            }
        } while (std::prev_permutation(bitmask.begin(), bitmask.end()));

        circle_centre_ = best_circle_centre;
        circle_radius_ = best_circle_radius;

        // Remove all glints that are too far from the estimated circle.
        ellipse_points.erase(std::remove_if(ellipse_points.begin(), ellipse_points.end(),
                                            [this](auto const &p) {
                                                double value{0.0};
                                                value += (circle_centre_.x - p.x) * (circle_centre_.x - p.x);
                                                value += (circle_centre_.y - p.y) * (circle_centre_.y - p.y);
                                                return std::abs(std::sqrt(value) - circle_radius_) > 3.0;
                                            }),
                             ellipse_points.end());

        if (ellipse_points.size() < 5) {
            return false;
        }
    }

    // All the remaining points are used to estimate the ellipse.
    glint_ellipse_ = cv::fitEllipse(ellipse_points);

    glint_locations_[0] = glint_ellipse_.center;
    glint_locations_[1] = glint_ellipse_.center;

    for (auto &point : ellipse_points) {
        if (point.x < glint_ellipse_.center.x && point.y < glint_locations_[0].y) {
            glint_locations_[0] = point;
        } else if (point.x > glint_ellipse_.center.x && point.y > glint_locations_[1].y) {
            glint_locations_[1] = point;
        }
    }

    for (auto &point : ellipse_points) {
        point = undistort(point);
    }

    cv::RotatedRect ellipse_undistorted = cv::fitEllipse(ellipse_points);

    if (kalman_filtering_enabled_) {
        glint_ellipse_kalman_.correct((KFMatF(5, 1) << ellipse_undistorted.center.x, ellipse_undistorted.center.y,
                                       ellipse_undistorted.size.width, ellipse_undistorted.size.height,
                                       ellipse_undistorted.angle));

        cv::Mat predicted_ellipse = glint_ellipse_kalman_.predict();
        glint_ellipse_undistorted_.center.x = predicted_ellipse.at<float>(0, 0);
        glint_ellipse_undistorted_.center.y = predicted_ellipse.at<float>(1, 0);
        glint_ellipse_undistorted_.size.width = predicted_ellipse.at<float>(2, 0);
        glint_ellipse_undistorted_.size.height = predicted_ellipse.at<float>(3, 0);
        glint_ellipse_undistorted_.angle = predicted_ellipse.at<float>(4, 0);
        //        glint_ellipse_undistorted_ = ellipse_undistorted;
    } else {
        glint_ellipse_undistorted_ = ellipse_undistorted;
    }

    mtx_features_.lock();
    for (int i = 0; i < 2; i++) {
        glint_locations_undistorted_[i] =
            glint_ellipse_undistorted_.center - glint_ellipse_.center + glint_locations_[i];
    }
    glint_represent_undistorted_ = glint_ellipse_undistorted_.center;
    mtx_features_.unlock();

    return true;
}

cv::KalmanFilter FeatureDetector::makePxKalmanFilter(const cv::Size2i &resolution, float framerate) {
    // Expected average saccade time.
    double saccade_length_sec = 0.1;
    double saccade_per_frame = std::fmin(saccade_length_sec / (1.0f / framerate), 1.0f);
    double velocity_decay = 1.0f - saccade_per_frame;
    cv::Mat transition_matrix{(KFMatD(4, 4) << 1, 0, 1.0f / framerate, 0, 0, 1, 0, 1.0f / framerate, 0, velocity_decay,
                               0, 0, 0, 0, 0, velocity_decay)};
    cv::Mat measurement_matrix{(KFMatD(2, 4) << 1, 0, 0, 0, 0, 1, 0, 0)};
    cv::Mat process_noise_cov{cv::Mat::eye(4, 4, CV_64F) * 2};
    cv::Mat measurement_noise_cov{cv::Mat::eye(2, 2, CV_64F) * 5};
    cv::Mat error_cov_post{cv::Mat::eye(4, 4, CV_64F)};
    cv::Mat state_post{(KFMatD(4, 1) << resolution.width / 2, resolution.height / 2, 0, 0)};

    cv::KalmanFilter KF(4, 2);
    KF.transitionMatrix = transition_matrix;
    KF.measurementMatrix = measurement_matrix;
    KF.processNoiseCov = process_noise_cov;
    KF.measurementNoiseCov = measurement_noise_cov;
    KF.errorCovPost = error_cov_post;
    KF.statePost = state_post;
    // Without this line, OpenCV complains about incorrect matrix dimensions.
    KF.predict();
    return KF;
}

cv::KalmanFilter FeatureDetector::makeRadiusKalmanFilter(const float &min_radius, const float &max_radius,
                                                         float framerate) {

    double radius_change_time_sec = 1.0;
    double radius_change_per_frame = std::fmin(radius_change_time_sec / (1.0f / framerate), 1.0f);
    double velocity_decay = 1.0f - radius_change_per_frame;
    cv::Mat transition_matrix{(KFMatD(2, 2) << 1, 1.0f / framerate, 0, velocity_decay)};
    cv::Mat measurement_matrix{(KFMatD(1, 2) << 1, 0)};
    cv::Mat process_noise_cov{cv::Mat::eye(2, 2, CV_64F) * 2};
    cv::Mat measurement_noise_cov{(KFMatD(1, 1) << 5)};
    cv::Mat error_cov_post{cv::Mat::eye(2, 2, CV_64F)};
    cv::Mat state_post{(KFMatD(2, 1) << (max_radius - min_radius) / 2, 0)};

    cv::KalmanFilter KF(2, 1);
    KF.transitionMatrix = transition_matrix;
    KF.measurementMatrix = measurement_matrix;
    KF.processNoiseCov = process_noise_cov;
    KF.measurementNoiseCov = measurement_noise_cov;
    KF.errorCovPost = error_cov_post;
    KF.statePost = state_post;
    // Without this line, OpenCV complains about incorrect matrix dimensions.
    KF.predict();
    return KF;
}

cv::KalmanFilter FeatureDetector::makeEllipseKalmanFilter(const cv::Size2i &resolution, float framerate) {
    // Expected average saccade time.
    double saccade_length_sec = 0.1;
    double saccade_per_frame = std::fmin(saccade_length_sec / (1.0f / framerate), 1.0f);
    double velocity_decay = 1.0f - saccade_per_frame;
    cv::Mat transition_matrix{(KFMatF(10, 10) << 1, 0, 0, 0, 0, 1.0f / framerate, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                               1.0f / framerate, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1.0f / framerate, 0, 0, 0, 0, 0, 1, 0, 0,
                               0, 0, 1.0f / framerate, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1.0f / framerate, 0, 0, 0, 0, 0,
                               velocity_decay, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, velocity_decay, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, velocity_decay, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, velocity_decay, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, velocity_decay)};
    cv::Mat measurement_matrix{(KFMatF(5, 10) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0)};
    cv::Mat process_noise_cov{cv::Mat::eye(10, 10, CV_32F) * 2};
    cv::Mat measurement_noise_cov{cv::Mat::eye(5, 5, CV_32F) * 5};
    cv::Mat error_cov_post{cv::Mat::eye(10, 10, CV_32F)};
    cv::Mat state_post{
        (KFMatF(10, 1) << resolution.width / 2, resolution.height / 2, resolution.width / 4, resolution.height / 4, 0)};

    cv::KalmanFilter KF(10, 5);
    KF.transitionMatrix = transition_matrix;
    KF.measurementMatrix = measurement_matrix;
    KF.processNoiseCov = process_noise_cov;
    KF.measurementNoiseCov = measurement_noise_cov;
    KF.errorCovPost = error_cov_post;
    KF.statePost = state_post;
    return KF;
}

cv::Mat FeatureDetector::getThresholdedPupilImage() {
    cv::Mat image_ = pupil_thresholded_image_.clone();
    return image_;
}

cv::Mat FeatureDetector::getThresholdedGlintsImage() {
    cv::Mat image_ = glints_thresholded_image_.clone();
    return image_;
}

void FeatureDetector::findPotentialNeighbours(std::vector<GlintCandidate> &glint_candidates) {
    // Checks every pair of glints if there are at the distance making them
    // neighbours.
    for (auto &first : glint_candidates) {
        for (auto &second : glint_candidates) {
            if (&first == &second) {
                continue;
            }
            double distance{cv::norm(first.location - second.location)};

            if (isBottomNeighbour(first, second)) {
                if (!first.bottom_neighbour) {
                    first.bottom_neighbour = &second;
                    second.upper_neighbour = &first;
                } else {
                    double old_distance{cv::norm(first.location - first.bottom_neighbour->location)};
                    if (distance < old_distance) {
                        first.bottom_neighbour->upper_neighbour = nullptr;
                        first.bottom_neighbour = &second;
                        second.upper_neighbour = &first;
                    }
                }
            }
            if (isRightNeighbour(first, second)) {
                if (!first.right_neighbour) {
                    first.right_neighbour = &second;
                    second.left_neighbour = &first;
                } else {
                    double old_distance{cv::norm(first.location - first.right_neighbour->location)};
                    if (distance < old_distance) {
                        first.right_neighbour->left_neighbour = nullptr;
                        first.right_neighbour = &second;
                        second.left_neighbour = &first;
                    }
                }
            }
        }
    }
}

void FeatureDetector::determineGlintTypes(std::vector<GlintCandidate> &glint_candidates) {
    // Glint position in the 3x2 grid is determined based on the neighbours it has.
    for (auto &glint_candidate : glint_candidates) {
        int mask = 0;
        mask |= (glint_candidate.right_neighbour != nullptr);
        mask |= (glint_candidate.bottom_neighbour != nullptr) << 1;
        mask |= (glint_candidate.left_neighbour != nullptr) << 2;
        mask |= (glint_candidate.upper_neighbour != nullptr) << 3;
        switch (mask) {
        case 3:
            glint_candidate.glint_type = GlintType::UpperLeft;
            break;
        case 6:
            glint_candidate.glint_type = GlintType::UpperRight;
            break;
        case 7:
            glint_candidate.glint_type = GlintType::UpperCentre;
            break;
        case 9:
            glint_candidate.glint_type = GlintType::BottomLeft;
            break;
        case 12:
            glint_candidate.glint_type = GlintType::BottomRight;
            break;
        case 13:
            glint_candidate.glint_type = GlintType::BottomCentre;
            break;
        default:
            glint_candidate.glint_type = GlintType::Unknown;
            break;
        }
    }
}

bool FeatureDetector::isLeftNeighbour(GlintCandidate &reference, GlintCandidate &compared) {
    return isRightNeighbour(compared, reference);
}

bool FeatureDetector::isRightNeighbour(GlintCandidate &reference, GlintCandidate &compared) {
    cv::Point2f distance{compared.location - reference.location};

    return (distance.y < detection_params_->max_glint_right_vert_distance
            && distance.y > detection_params_->min_glint_right_vert_distance
            && distance.x < detection_params_->max_glint_right_hor_distance
            && distance.x > detection_params_->min_glint_right_hor_distance);
}

bool FeatureDetector::isUpperNeighbour(GlintCandidate &reference, GlintCandidate &compared) {
    return isBottomNeighbour(compared, reference);
}

bool FeatureDetector::isBottomNeighbour(GlintCandidate &reference, GlintCandidate &compared) {
    cv::Point2f distance{compared.location - reference.location};

    return (distance.y < detection_params_->max_glint_bottom_vert_distance
            && distance.y > detection_params_->min_glint_bottom_vert_distance
            && distance.x < detection_params_->max_glint_bottom_hor_distance
            && distance.x > detection_params_->min_glint_bottom_hor_distance);
}

void FeatureDetector::approximatePositions() {
    // Computes relative positions of all glints to fill the missing spots.
    cv::Point2f mean_right_move{15, 0};
    cv::Point2f mean_down_move{0, 40};
    cv::Point2f total_right_move{0};
    int total_right_found{0};
    cv::Point2f total_down_move{0};
    int total_down_found{0};

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            if (found_glints_[i * 3 + j] && found_glints_[i * 3 + j + 1]) {
                total_right_move += selected_glints_[i * 3 + j + 1].location - selected_glints_[i * 3 + j].location;
                total_right_found++;
            }
        }
    }
    for (int i = 0; i < 3; i++) {
        if (found_glints_[i] && found_glints_[i + 3]) {
            total_down_move += selected_glints_[i + 3].location - selected_glints_[i].location;
            total_down_found++;
        }
    }

    if (total_right_found > 0) {
        mean_right_move = total_right_move / total_right_found;
    }
    if (total_down_found > 0) {
        mean_down_move = total_down_move / total_down_found;
    }

    for (int i = 0; i < 6; i++) {
        if (!found_glints_[i]) {
            for (int j = 0; j < 6; j++) {
                if (found_glints_[j]) {
                    selected_glints_[i].location = selected_glints_[j].location - (j % 3 - i % 3) * mean_right_move
                        - (j / 3 - i / 3) * mean_down_move;
                    selected_glints_[i].rating = 0;
                    break;
                }
            }
        }
    }
}

cv::Point2f FeatureDetector::undistort(cv::Point2f point) {
    if (!distorted_) {
        return point;
    }
    // Convert from 0-based C++ coordinates to 1-based Matlab coordinates and add offset.
    cv::Point2f new_point{point.x + 1 + capture_offset_->width, point.y + 1 + capture_offset_->height};

    std::vector<cv::Point2f> points{point};
    std::vector<cv::Point2f> new_points{new_point};

    cv::undistortPoints(points, new_points, *intrinsic_matrix_, *distortion_coefficients_);

    new_point = new_points[0];
    // Remove normalization
    new_point.x *= intrinsic_matrix_->at<double>(0, 0);
    new_point.y *= intrinsic_matrix_->at<double>(1, 1);
    new_point.x += intrinsic_matrix_->at<double>(0, 2);
    new_point.y += intrinsic_matrix_->at<double>(1, 2);

    // Convert back to 0-based C++ coordinates and subtract offset.
    new_point.x -= 1 + capture_offset_->width;
    new_point.y -= 1 + capture_offset_->height;

    return new_point;
}

void FeatureDetector::getPupilGlintVector(cv::Vec2f &pupil_glint_vector) {
    mtx_features_.lock();
    pupil_glint_vector = pupil_location_undistorted_ - glint_represent_undistorted_;
    mtx_features_.unlock();
}

void FeatureDetector::getPupilGlintVectorFiltered(cv::Vec2f &pupil_glint_vector) {
    mtx_features_.lock();
    pupil_glint_vector = pupil_location_filtered_ - glint_location_filtered_;
    mtx_features_.unlock();
}

cv::RotatedRect FeatureDetector::getEllipse() {
    return glint_ellipse_undistorted_;
}

cv::RotatedRect FeatureDetector::getDistortedEllipse() {
    return glint_ellipse_;
}

void FeatureDetector::setGazeBufferSize(uint8_t value) {
    buffer_size_ = value;
}

void FeatureDetector::updateGazeBuffer() {
    if (pupil_location_buffer_.size() != buffer_size_ || glint_location_buffer_.size() != buffer_size_) {
        pupil_location_buffer_.resize(buffer_size_);
        glint_location_buffer_.resize(buffer_size_);
        buffer_idx_ = 0;
        buffer_summed_count_ = 0;
        pupil_location_summed_.x = 0.0f;
        pupil_location_summed_.y = 0.0f;
        glint_location_summed_.x = 0.0f;
        glint_location_summed_.y = 0.0f;
    }

    if (buffer_summed_count_ == buffer_size_) {
        pupil_location_summed_ -= pupil_location_buffer_[buffer_idx_];
        glint_location_summed_ -= glint_location_buffer_[buffer_idx_];
    }

    pupil_location_buffer_[buffer_idx_] = pupil_location_undistorted_;
    glint_location_buffer_[buffer_idx_] = glint_represent_undistorted_;
    pupil_location_summed_ += pupil_location_buffer_[buffer_idx_];
    glint_location_summed_ += glint_location_buffer_[buffer_idx_];

    if (buffer_summed_count_ != buffer_size_) {
        buffer_summed_count_++;
    }

    mtx_features_.lock();
    pupil_location_filtered_ = pupil_location_summed_ / buffer_summed_count_;
    glint_location_filtered_ = glint_location_summed_ / buffer_summed_count_;
    mtx_features_.unlock();

    buffer_idx_ = (buffer_idx_ + 1) % buffer_size_;
}

void FeatureDetector::identifyNeighbours(GlintCandidate *glint_candidate) {
    if (glint_candidate->upper_neighbour && glint_candidate->glint_type >= 3
        && !found_glints_[glint_candidate->glint_type - 3]) {
        selected_glints_[glint_candidate->glint_type - 3] = *glint_candidate->upper_neighbour;
        found_glints_[glint_candidate->glint_type - 3] = true;
        glint_candidate->upper_neighbour->glint_type = (GlintType) (glint_candidate->glint_type - 3);
        identifyNeighbours(glint_candidate->upper_neighbour);
    }
    if (glint_candidate->bottom_neighbour && glint_candidate->glint_type < 3
        && !found_glints_[glint_candidate->glint_type + 3]) {
        selected_glints_[glint_candidate->glint_type + 3] = *glint_candidate->bottom_neighbour;
        found_glints_[glint_candidate->glint_type + 3] = true;
        glint_candidate->bottom_neighbour->glint_type = (GlintType) (glint_candidate->glint_type + 3);
        identifyNeighbours(glint_candidate->bottom_neighbour);
    }
    if (glint_candidate->right_neighbour && glint_candidate->glint_type % 3 < 2
        && !found_glints_[glint_candidate->glint_type + 1]) {
        selected_glints_[glint_candidate->glint_type + 1] = *glint_candidate->right_neighbour;
        found_glints_[glint_candidate->glint_type + 1] = true;
        glint_candidate->right_neighbour->glint_type = (GlintType) (glint_candidate->glint_type + 1);
        identifyNeighbours(glint_candidate->right_neighbour);
    }
    if (glint_candidate->left_neighbour && glint_candidate->glint_type % 3 > 0
        && !found_glints_[glint_candidate->glint_type - 1]) {
        selected_glints_[glint_candidate->glint_type - 1] = *glint_candidate->left_neighbour;
        found_glints_[glint_candidate->glint_type - 1] = true;
        glint_candidate->left_neighbour->glint_type = (GlintType) (glint_candidate->glint_type - 1);
        identifyNeighbours(glint_candidate->left_neighbour);
    }
}

FeatureDetector::~FeatureDetector() {
    if (!bayes_solver_->empty()) {
        bayes_solver_.release();
    }
    if (!bayes_minimizer_func_.empty()) {
        bayes_minimizer_func_.release();
    }
}

bool FeatureDetector::setKalmanFiltering(bool enable) {
    bool previous_state = kalman_filtering_enabled_;
    kalman_filtering_enabled_ = enable;
    return previous_state;
}
} // namespace et