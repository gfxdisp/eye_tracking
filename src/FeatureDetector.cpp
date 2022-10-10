#include "FeatureDetector.hpp"

#include "Utils.hpp"

#include <opencv2/cudaarithm.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <opencv2/cudaimgproc.hpp>
#include <random>
#include <set>

using KFMatD = cv::Mat_<double>;
using KFMatF = cv::Mat_<float>;

namespace et {

void FeatureDetector::initialize(int camera_id) {
    pupil_kalman_ = makePxKalmanFilter(
        et::Settings::parameters.camera_params[camera_id].region_of_interest,
        et::Settings::parameters.camera_params[camera_id].framerate);
    leds_kalman_ = makePxKalmanFilter(
        et::Settings::parameters.camera_params[camera_id].region_of_interest,
        et::Settings::parameters.camera_params[camera_id].framerate);
    pupil_radius_kalman_ = makeRadiusKalmanFilter(
        et::Settings::parameters.detection_params.min_pupil_radius[camera_id],
        et::Settings::parameters.detection_params.max_pupil_radius[camera_id],
        et::Settings::parameters.camera_params[camera_id].framerate);
    glint_ellipse_kalman_ = makeEllipseKalmanFilter(
        et::Settings::parameters.camera_params[camera_id].region_of_interest,
        et::Settings::parameters.detection_params.min_pupil_radius[camera_id],
        et::Settings::parameters.detection_params.max_pupil_radius[camera_id],
        et::Settings::parameters.camera_params[camera_id].framerate);
    glint_locations_.resize(
        Settings::parameters.leds_positions[camera_id].size());

    region_if_interest_ = &Settings::parameters.camera_params[camera_id].region_of_interest;
    pupil_threshold_ = &Settings::parameters.user_params->pupil_threshold[camera_id];
    glint_threshold_ = &Settings::parameters.user_params->glint_threshold[camera_id];
    pupil_search_centre_ =
        &Settings::parameters.detection_params.pupil_search_centre[camera_id];
    pupil_search_radius_ =
        &Settings::parameters.detection_params.pupil_search_radius[camera_id];
    min_pupil_radius_ =
        &Settings::parameters.detection_params.min_pupil_radius[camera_id];
    max_pupil_radius_ =
        &Settings::parameters.detection_params.max_pupil_radius[camera_id];

    bayes_minimizer_ = new BayesMinimizer();
    bayes_minimizer_func_ =
        cv::Ptr<cv::DownhillSolver::Function>{bayes_minimizer_};
    bayes_solver_ = cv::DownhillSolver::create();
    bayes_solver_->setFunction(bayes_minimizer_func_);
    cv::Mat step = (cv::Mat_<double>(1, 3) << 100, 100, 100);
    bayes_solver_->setInitStep(step);

    cv::Mat morphology_element{};
    int size{};

    size = Settings::parameters.detection_params.pupil_close_size;
    if (size) {
        morphology_element =
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(size, size));
        pupil_close_filter_ = cv::cuda::createMorphologyFilter(
            cv::MORPH_CLOSE, CV_8UC1, morphology_element);
    }
    size = Settings::parameters.detection_params.pupil_dilate_size;
    if (size) {
        morphology_element =
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(size, size));
        pupil_dilate_filter_ = cv::cuda::createMorphologyFilter(
            cv::MORPH_DILATE, CV_8UC1, morphology_element);
    }
    size = Settings::parameters.detection_params.pupil_erode_size;
    if (size) {
        morphology_element =
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(size, size));
        pupil_erode_filter_ = cv::cuda::createMorphologyFilter(
            cv::MORPH_ERODE, CV_8UC1, morphology_element);
    }

    size = Settings::parameters.detection_params.glint_close_size;
    if (size) {
        morphology_element =
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(size, size));
        glints_close_filter_ = cv::cuda::createMorphologyFilter(
            cv::MORPH_CLOSE, CV_8UC1, morphology_element);
    }
    size = Settings::parameters.detection_params.glint_dilate_size;
    if (size) {
        morphology_element =
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(size, size));
        glints_dilate_filter_ = cv::cuda::createMorphologyFilter(
            cv::MORPH_DILATE, CV_8UC1, morphology_element);
    }
    size = Settings::parameters.detection_params.glint_erode_size;
    if (size) {
        morphology_element =
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(size, size));
        glints_erode_filter_ = cv::cuda::createMorphologyFilter(
            cv::MORPH_ERODE, CV_8UC1, morphology_element);
    }

    glints_template_ = cv::imread("template.png", cv::IMREAD_GRAYSCALE);
    template_crop_ = (KFMatF(2, 3) << 1, 0, glints_template_.cols / 2,
                         0, 1, glints_template_.rows / 2);
}

cv::Point2f FeatureDetector::getPupil() {
    return pupil_location_;
}

void FeatureDetector::getPupil(cv::Point2f &pupil) {
    mtx_features_.lock();
    pupil = pupil_location_;
    mtx_features_.unlock();
}

void FeatureDetector::getPupilFiltered(cv::Point2f &pupil) {
    mtx_features_.lock();
    pupil = pupil_location_filtered_;
    mtx_features_.unlock();
}

int FeatureDetector::getPupilRadius() const {
    return pupil_radius_;
}

std::vector<cv::Point2f> *FeatureDetector::getGlints() {
    return &glint_locations_;
}

void FeatureDetector::getGlints(std::vector<cv::Point2f> &glint_locations) {
    mtx_features_.lock();
    glint_locations = glint_locations_;
    mtx_features_.unlock();
}

bool FeatureDetector::findPupil(const cv::Mat &image) {
    cv::threshold(
        image, pupil_thresholded_image_, *pupil_threshold_, 255,
        cv::THRESH_BINARY_INV);
    pupil_thresholded_image_.convertTo(
        pupil_thresholded_image_, CV_8UC1);

    cv::findContours(pupil_thresholded_image_, contours_, hierarchy_,
                     cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::Point2f best_centre{};
    float best_radius{};
    float best_rating{0};

    cv::Point2f image_centre{*pupil_search_centre_};
    int max_distance{*pupil_search_radius_};

    for (const std::vector<cv::Point> &contour : contours_) {
        cv::Point2f centre;
        float radius;

        cv::minEnclosingCircle(contour, centre, radius);
        if (radius < *min_pupil_radius_
            or radius > *max_pupil_radius_)
            continue;

        float distance = euclideanDistance(centre, image_centre);
        if (distance > max_distance)
            continue;

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

    pupil_kalman_.correct(
        (KFMatD(2, 1) << best_centre.x, best_centre.y));
    pupil_radius_kalman_.correct((KFMatD(1, 1) << best_radius));
    mtx_features_.lock();
    pupil_location_ = toPoint(pupil_kalman_.predict());
    pupil_radius_ =
        (int) toValue(pupil_radius_kalman_.predict());
    //    pupil_location_[camera_id] = best_centre; // Disables Kalman filtering
    //    pupil_radius_[camera_id] = best_radius;   // Disables Kalman filtering
    mtx_features_.unlock();
    return true;
}

bool FeatureDetector::findGlints(const cv::Mat &image) {
    cv::threshold(
        image, glints_thresholded_image_,
        *glint_threshold_, 255,
        cv::THRESH_BINARY);
    glints_thresholded_image_.convertTo(
        glints_thresholded_image_, CV_8UC1);

    cv::findContours(glints_thresholded_image_, contours_,
                     hierarchy_, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Point2f image_centre{*pupil_search_centre_};
    int max_distance{*pupil_search_radius_};

    std::vector<GlintCandidate> glint_candidates{};
    glint_candidates.reserve(contours_.size());

    for (const std::vector<cv::Point> &contour : contours_) {
        cv::Point2f centre;
        float radius;
        cv::minEnclosingCircle(contour, centre, radius);
        if (radius > Settings::parameters.detection_params.max_glint_radius
            || radius < Settings::parameters.detection_params.min_glint_radius)
            continue;

        float distance = euclideanDistance(centre, image_centre);
        if (distance > max_distance)
            continue;

        if (!isInEllipse(centre, pupil_location_)) {
            continue;
        }

        const float contour_area = static_cast<float>(cv::contourArea(contour));
        const float circle_area = 3.1415926f * powf(radius, 2);
        float rating = contour_area / circle_area;
        GlintCandidate glint_candidate{};
        glint_candidate.location = centre;
        glint_candidate.rating = rating;
        glint_candidate.found = false;
        glint_candidate.right_neighbour = nullptr;
        glint_candidate.bottom_neighbour = nullptr;
        glint_candidate.neighbour_count = 0;
        glint_candidates.push_back(glint_candidate);
    }

    if (glint_candidates.empty()) {
        return false;
    }

    std::sort(glint_candidates.begin(), glint_candidates.end(),
              [](const GlintCandidate &a, const GlintCandidate &b) {
                  return a.rating > b.rating;
              });
    findBestGlintPair(glint_candidates);
    determineGlintTypes(glint_candidates);

    for (int i = 0; i < 6; i++) {
        selected_glints_[i].found = false;
    }

    cv::Point2f distance{};
    bool glint_found{false};
    for (int i = 0; i < 2; i++) {
        for (auto &glint : glint_candidates) {
            // In the first iteration (i = 0) we check only for centre LEDs, as they are the most reliable
            if (i == 0 && glint.glint_type != GlintType::UpperCentre
                && glint.glint_type != GlintType::BottomCentre) {
                continue;
            }
            if (glint.glint_type == GlintType::Unknown) {
                continue;
            }

            if (glint_found) {
                if (selected_glints_[glint.glint_type].found
                    && selected_glints_[glint.glint_type].rating
                        >= glint.rating) {
                    continue;
                }

                if (glint.glint_type % 3 != 0
                    && !isLeftNeighbour(
                        glint,
                        selected_glints_[glint.glint_type - 1])) {
                    continue;
                }
                if (glint.glint_type % 3 != 2
                    && !isRightNeighbour(
                        glint,
                        selected_glints_[glint.glint_type + 1])) {
                    continue;
                }
                if (glint.glint_type / 3 == 0
                    && !isBottomNeighbour(
                        glint,
                        selected_glints_[glint.glint_type + 3])) {
                    continue;
                }
                if (glint.glint_type / 3 == 1
                    && !isUpperNeighbour(
                        glint,
                        selected_glints_[glint.glint_type - 3])) {
                    continue;
                }
            }

            glint_found = true;
            selected_glints_[glint.glint_type] = glint;
            selected_glints_[glint.glint_type].found = true;
            identifyNeighbours(&glint);
            approximatePositions();
        }
    }

    int found_glints{0};
    for (int i = 0; i < 6; i++) {
        if (selected_glints_[i].found) {
            found_glints++;
        }
    }

    if (found_glints == 0) {
        return false;
    }

    if (!selected_glints_[0].found
        && !selected_glints_[3].found) {
        selected_glints_[0] = selected_glints_[1];
        selected_glints_[1] = selected_glints_[2];
        selected_glints_[3] = selected_glints_[4];
        selected_glints_[4] = selected_glints_[5];
        selected_glints_[2].found = false;
        selected_glints_[5].found = false;
        approximatePositions();
    }

    cv::Point2f glints_centre{};
    cv::Point2f new_glints_centre{};

    for (int i = 0; i < 6; i++) {
        glints_centre += selected_glints_[i].location;
    }

    glints_centre.x /= led_count_;
    glints_centre.y /= led_count_;

    leds_kalman_.correct(
        (KFMatD(2, 1) << glints_centre.x, glints_centre.y));
    new_glints_centre = toPoint(leds_kalman_.predict());
    for (int i = 0; i < led_count_; i++) {
        mtx_features_.lock();
        glint_locations_[i] = selected_glints_[i].location
            + (new_glints_centre - glints_centre);
        mtx_features_.unlock();
    }

    return true;
}

cv::KalmanFilter
FeatureDetector::makePxKalmanFilter(const cv::Size2i &resolution,
                                    float framerate) {

    double saccade_length_sec = 0.1; // expected average time
    double saccade_per_frame =
        std::fmin(saccade_length_sec / (1.0f / framerate), 1.0f);
    double velocity_decay = 1.0f - saccade_per_frame;
    cv::Mat transition_matrix{(KFMatD(4, 4) << 1, 0, 1.0f / framerate, 0, 0, 1,
                               0, 1.0f / framerate, 0, velocity_decay, 0, 0, 0,
                               0, 0, velocity_decay)};
    cv::Mat measurement_matrix{(KFMatD(2, 4) << 1, 0, 0, 0, 0, 1, 0, 0)};
    cv::Mat process_noise_cov{cv::Mat::eye(4, 4, CV_64F) * 2};
    cv::Mat measurement_noise_cov{cv::Mat::eye(2, 2, CV_64F) * 5};
    cv::Mat error_cov_post{cv::Mat::eye(4, 4, CV_64F)};
    cv::Mat state_post{
        (KFMatD(4, 1) << resolution.width / 2, resolution.height / 2, 0, 0)};

    cv::KalmanFilter KF(4, 2);
    KF.transitionMatrix = transition_matrix;
    KF.measurementMatrix = measurement_matrix;
    KF.processNoiseCov = process_noise_cov;
    KF.measurementNoiseCov = measurement_noise_cov;
    KF.errorCovPost = error_cov_post;
    KF.statePost = state_post;
    KF.predict(); // Without this line, OpenCV complains about incorrect matrix dimensions
    return KF;
}

cv::KalmanFilter FeatureDetector::makeRadiusKalmanFilter(
    const float &min_radius, const float &max_radius, float framerate) {

    double radius_change_time_sec = 1.0;
    double radius_change_per_frame =
        std::fmin(radius_change_time_sec / (1.0f / framerate), 1.0f);
    double velocity_decay = 1.0f - radius_change_per_frame;
    cv::Mat transition_matrix{
        (KFMatD(2, 2) << 1, 1.0f / framerate, 0, velocity_decay)};
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
    KF.predict(); // Without this line, OpenCV complains about incorrect matrix dimensions
    return KF;
}

cv::KalmanFilter FeatureDetector::makeEllipseKalmanFilter(
    const cv::Size2i &resolution, const float &min_axis, const float &max_axis,
    float framerate) {
    double saccade_length_sec = 0.1; // expected average time
    double saccade_per_frame =
        std::fmin(saccade_length_sec / (1.0f / framerate), 1.0f);
    double velocity_decay = 1.0f - saccade_per_frame;
    cv::Mat transition_matrix{
        (KFMatF(10, 10) << 1, 0, 0, 0, 0, 1.0f / framerate, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 0, 1.0f / framerate, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
         1.0f / framerate, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1.0f / framerate, 0, 0,
         0, 0, 0, 1, 0, 0, 0, 0, 1.0f / framerate, 0, 0, 0, 0, 0,
         velocity_decay, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, velocity_decay, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, velocity_decay, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         velocity_decay, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, velocity_decay)};
    cv::Mat measurement_matrix{(KFMatF(5, 10) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 1, 0, 0, 0, 0, 0)};
    cv::Mat process_noise_cov{cv::Mat::eye(10, 10, CV_32F) * 2};
    cv::Mat measurement_noise_cov{cv::Mat::eye(5, 5, CV_32F) * 5};
    cv::Mat error_cov_post{cv::Mat::eye(10, 10, CV_32F)};
    cv::Mat state_post{(KFMatF(10, 1) << resolution.width / 2,
                        resolution.height / 2, resolution.width / 4,
                        resolution.height / 4, 0)};

    cv::KalmanFilter KF(10, 5);
    KF.transitionMatrix = transition_matrix;
    KF.measurementMatrix = measurement_matrix;
    KF.processNoiseCov = process_noise_cov;
    KF.measurementNoiseCov = measurement_noise_cov;
    KF.errorCovPost = error_cov_post;
    KF.statePost = state_post;
    //    KF.predict(); // Without this line, OpenCV complains about incorrect matrix dimensions
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

void FeatureDetector::findBestGlintPair(
    std::vector<GlintCandidate> &glint_candidates) {
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
                    double old_distance{cv::norm(
                        first.location - first.bottom_neighbour->location)};
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
                    double old_distance{cv::norm(
                        first.location - first.right_neighbour->location)};
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

void FeatureDetector::determineGlintTypes(
    std::vector<GlintCandidate> &glint_candidates) {
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

bool FeatureDetector::isLeftNeighbour(GlintCandidate &reference,
                                      GlintCandidate &compared) {
    return isRightNeighbour(compared, reference);
}

bool FeatureDetector::isRightNeighbour(GlintCandidate &reference,
                                       GlintCandidate &compared) {
    cv::Point2f distance{compared.location - reference.location};
    DetectionParams *params{&Settings::parameters.detection_params};

    return (distance.y < params->glint_right_vert_distance[1]
            && distance.y > params->glint_right_vert_distance[0]
            && distance.x < params->glint_right_hor_distance[1]
            && distance.x > params->glint_right_hor_distance[0]);
}

bool FeatureDetector::isUpperNeighbour(GlintCandidate &reference,
                                       GlintCandidate &compared) {
    return isBottomNeighbour(compared, reference);
}

bool FeatureDetector::isBottomNeighbour(GlintCandidate &reference,
                                        GlintCandidate &compared) {
    cv::Point2f distance{compared.location - reference.location};
    DetectionParams *params{&Settings::parameters.detection_params};

    return (distance.y < params->glint_bottom_vert_distance[1]
            && distance.y > params->glint_bottom_vert_distance[0]
            && distance.x < params->glint_bottom_hor_distance[1]
            && distance.x > params->glint_bottom_hor_distance[0]);
}

void FeatureDetector::approximatePositions() {
    cv::Point2f mean_right_move{15, 0};
    cv::Point2f mean_down_move{0, 40};
    cv::Point2f total_right_move{0};
    int total_right_found{0};
    cv::Point2f total_down_move{0};
    int total_down_found{0};

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            if (selected_glints_[i * 3 + j].found
                && selected_glints_[i * 3 + j + 1].found) {
                total_right_move +=
                    selected_glints_[i * 3 + j + 1].location
                    - selected_glints_[i * 3 + j].location;
                total_right_found++;
            }
        }
    }
    for (int i = 0; i < 3; i++) {
        if (selected_glints_[i].found
            && selected_glints_[i + 3].found) {
            total_down_move += selected_glints_[i + 3].location
                - selected_glints_[i].location;
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
        if (!selected_glints_[i].found) {
            for (int j = 0; j < 6; j++) {
                if (selected_glints_[j].found) {
                    selected_glints_[i].location =
                        selected_glints_[j].location
                        - (j % 3 - i % 3) * mean_right_move
                        - (j / 3 - i / 3) * mean_down_move;
                    selected_glints_[i].rating = 0;
                    break;
                }
            }
        }
    }
}

void FeatureDetector::getPupilGlintVector(cv::Vec2f &pupil_glint_vector) {
    mtx_features_.lock();
    pupil_glint_vector =
        pupil_location_ - glint_locations_[0];
    mtx_features_.unlock();
}

void FeatureDetector::getPupilGlintVectorFiltered(cv::Vec2f &pupil_glint_vector) {
    mtx_features_.lock();
    pupil_glint_vector = pupil_location_filtered_
        - glint_location_filtered_;
    mtx_features_.unlock();
}

bool FeatureDetector::findEllipse(const cv::Mat &image,
                                  const cv::Point2f &pupil) {

    cv::matchTemplate(image, glints_template_,
                      glints_thresholded_image_, cv::TM_CCOEFF);
    cv::warpAffine(glints_thresholded_image_,
                   glints_thresholded_image_,
                   template_crop_,
                   cv::Size(image.cols, image.rows));
    cv::threshold(glints_thresholded_image_,
                  glints_thresholded_image_, *glint_threshold_ * 2e3, 255,
                  CV_8UC1);
    glints_thresholded_image_.convertTo(
        glints_thresholded_image_, CV_8UC1);

    cv::findContours(glints_thresholded_image_, contours_,
                     hierarchy_, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Point2f> ellipse_points{};
    cv::Point2f im_centre{
        *region_if_interest_ / 2};

    static unsigned seed =
        std::chrono::system_clock::now().time_since_epoch().count();

    for (const auto &contour : contours_) {
        cv::Point2d mean_point{};
        for (const auto &point : contour) {
            mean_point.x += point.x;
            mean_point.y += point.y;
        }
        mean_point.x /= contour.size();
        mean_point.y /= contour.size();
        ellipse_points.push_back(mean_point);
    }

    ellipse_points.erase(
        std::remove_if(ellipse_points.begin(), ellipse_points.end(),
                       [&pupil, &im_centre](auto const &p) {
                           float distance_a = euclideanDistance(p, pupil);
                           float distance_b = euclideanDistance(p, im_centre);
                           return distance_a > 300 || distance_b > 300;
                       }),
        ellipse_points.end());

    std::sort(ellipse_points.begin(), ellipse_points.end(),
              [this](auto const &a, auto const &b) {
                  float distance_a =
                      euclideanDistance(a, ellipse_centre_);
                  float distance_b =
                      euclideanDistance(b, ellipse_centre_);
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
    double ellipse_radius{};
    do {
        int counter = 0;
        for (int i = 0; counter < 3; i++) {
            if (bitmask[i]) {
                circle_points[counter] = ellipse_points[i];
                counter++;
            }
        }
        bayes_minimizer_->setParameters(circle_points,
                                        ellipse_centre_,
                                        ellipse_radius_);

        cv::Mat x = (cv::Mat_<double>(1, 3) << im_centre.x, im_centre.y,
                     ellipse_radius_);
        bayes_solver_->minimize(x);
        ellipse_centre.x = x.at<double>(0, 0);
        ellipse_centre.y = x.at<double>(0, 1);
        ellipse_radius = x.at<double>(0, 2);

        counter = 0;
        for (auto &ellipse_point : ellipse_points) {
            double value{0.0};
            value += (ellipse_centre.x - ellipse_point.x)
                * (ellipse_centre.x - ellipse_point.x);
            value += (ellipse_centre.y - ellipse_point.y)
                * (ellipse_centre.y - ellipse_point.y);
            if (std::abs(std::sqrt(value) - ellipse_radius) <= 3.0) {
                counter++;
            }
        }
        if (counter > best_counter) {
            best_counter = counter;
            best_circle_centre = ellipse_centre;
            best_circle_radius = ellipse_radius;
        }
    } while (std::prev_permutation(bitmask.begin(), bitmask.end()));

    ellipse_centre_ = best_circle_centre;
    ellipse_radius_ = best_circle_radius;

    ellipse_points.erase(
        std::remove_if(ellipse_points.begin(), ellipse_points.end(),
                       [this](auto const &p) {
                           double value{0.0};
                           value += (ellipse_centre_.x - p.x)
                               * (ellipse_centre_.x - p.x);
                           value += (ellipse_centre_.y - p.y)
                               * (ellipse_centre_.y - p.y);
                           return std::abs(std::sqrt(value)
                                           - ellipse_radius_)
                               > 3.0;
                       }),
        ellipse_points.end());

    if (ellipse_points.size() < 5) {
        return false;
    }

    cv::RotatedRect ellipse = cv::fitEllipse(ellipse_points);

    mtx_features_.lock();
    glint_locations_[0] = ellipse.center;
    mtx_features_.unlock();

    glint_ellipse_kalman_.correct(
        (KFMatF(5, 1) << ellipse.center.x, ellipse.center.y, ellipse.size.width,
         ellipse.size.height, ellipse.angle));

    cv::Mat predicted_ellipse = glint_ellipse_kalman_.predict();
    glint_ellipse_.center.x = predicted_ellipse.at<float>(0, 0);
    glint_ellipse_.center.y = predicted_ellipse.at<float>(1, 0);
    glint_ellipse_.size.width = predicted_ellipse.at<float>(2, 0);
    glint_ellipse_.size.height = predicted_ellipse.at<float>(3, 0);
    glint_ellipse_.angle = predicted_ellipse.at<float>(4, 0);
//    glint_ellipse_[camera_id] = ellipse; // Disables Kalman filtering

    return true;
}

cv::RotatedRect FeatureDetector::getEllipse() {
    return glint_ellipse_;
}

void FeatureDetector::setGazeBufferSize(uint8_t value) {
    buffer_size_ = value;
}

void FeatureDetector::updateGazeBuffer() {
    if (pupil_location_buffer_.size() != buffer_size_
        || glint_location_buffer_.size() != buffer_size_) {
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

    pupil_location_buffer_[buffer_idx_] = pupil_location_;
    glint_location_buffer_[buffer_idx_] = glint_locations_[0];
    pupil_location_summed_ += pupil_location_buffer_[buffer_idx_];
    glint_location_summed_ += glint_location_buffer_[buffer_idx_];

    if (buffer_summed_count_ != buffer_size_) {
        buffer_summed_count_++;
    }

    mtx_features_.lock();
    pupil_location_filtered_ =
        pupil_location_summed_ / buffer_summed_count_;
    glint_location_filtered_ =
        glint_location_summed_ / buffer_summed_count_;
    mtx_features_.unlock();

    buffer_idx_ = (buffer_idx_ + 1) % buffer_size_;
}

void FeatureDetector::identifyNeighbours(GlintCandidate *glint_candidate) {
    if (glint_candidate->upper_neighbour && glint_candidate->glint_type >= 3
        && !selected_glints_[glint_candidate->glint_type - 3]
                .found) {
        selected_glints_[glint_candidate->glint_type - 3] =
            *glint_candidate->upper_neighbour;
        selected_glints_[glint_candidate->glint_type - 3].found =
            true;
        glint_candidate->upper_neighbour->glint_type =
            (GlintType) (glint_candidate->glint_type - 3);
        identifyNeighbours(glint_candidate->upper_neighbour);
    }
    if (glint_candidate->bottom_neighbour && glint_candidate->glint_type < 3
        && !selected_glints_[glint_candidate->glint_type + 3]
                .found) {
        selected_glints_[glint_candidate->glint_type + 3] =
            *glint_candidate->bottom_neighbour;
        selected_glints_[glint_candidate->glint_type + 3].found =
            true;
        glint_candidate->bottom_neighbour->glint_type =
            (GlintType) (glint_candidate->glint_type + 3);
        identifyNeighbours(glint_candidate->bottom_neighbour);
    }
    if (glint_candidate->right_neighbour && glint_candidate->glint_type % 3 < 2
        && !selected_glints_[glint_candidate->glint_type + 1]
                .found) {
        selected_glints_[glint_candidate->glint_type + 1] =
            *glint_candidate->right_neighbour;
        selected_glints_[glint_candidate->glint_type + 1].found =
            true;
        glint_candidate->right_neighbour->glint_type =
            (GlintType) (glint_candidate->glint_type + 1);
        identifyNeighbours(glint_candidate->right_neighbour);
    }
    if (glint_candidate->left_neighbour && glint_candidate->glint_type % 3 > 0
        && !selected_glints_[glint_candidate->glint_type - 1]
                .found) {
        selected_glints_[glint_candidate->glint_type - 1] =
            *glint_candidate->left_neighbour;
        selected_glints_[glint_candidate->glint_type - 1].found =
            true;
        glint_candidate->left_neighbour->glint_type =
            (GlintType) (glint_candidate->glint_type - 1);
        identifyNeighbours(glint_candidate->left_neighbour);
    }
}

FeatureDetector::~FeatureDetector() {
    bayes_solver_.release();
    bayes_minimizer_func_.release();
}
} // namespace et