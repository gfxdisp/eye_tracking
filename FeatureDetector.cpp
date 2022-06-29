#include "FeatureDetector.hpp"

#include <opencv2/cudaarithm.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>

using KFMat = cv::Mat_<double>;

namespace et {

void FeatureDetector::initializeKalmanFilters(const cv::Size2i &resolution,
                                              float framerate) {
    pupil_kalman_ = makeKalmanFilter(resolution, framerate);
    leds_kalman_ = makeKalmanFilter(resolution, framerate);
    glint_locations_.resize(Settings::parameters.leds_positions.size());
}

bool FeatureDetector::findImageFeatures(const cv::Mat &image) {
    gpu_image_.upload(image);
    cv::cuda::threshold(gpu_image_, pupil_thresholded_image_,
                        Settings::parameters.user_params->pupil_threshold, 255,
                        cv::THRESH_BINARY_INV);
    cv::cuda::threshold(gpu_image_, glints_thresholded_image_,
                        Settings::parameters.user_params->glint_threshold, 255,
                        cv::THRESH_BINARY);
    return findPupil() & findGlints();
}

cv::Point2f FeatureDetector::getPupil() {
    return pupil_location_;
}

void FeatureDetector::getPupil(cv::Point2f &pupil) {
    mtx_features_.lock();
    pupil = pupil_location_;
    mtx_features_.unlock();
}

int FeatureDetector::getPupilRadius() const {
    return pupil_radius_;
}

void FeatureDetector::getPupilRadius(int &pupil_radius) {
    mtx_features_.lock();
    pupil_radius = pupil_radius_;
    mtx_features_.unlock();
}

std::vector<cv::Point2f> *FeatureDetector::getGlints() {
    return &glint_locations_;
}

void FeatureDetector::getGlints(std::vector<cv::Point2f> &glint_locations) {
    mtx_features_.lock();
    glint_locations = glint_locations_;
    mtx_features_.unlock();
}

bool FeatureDetector::findPupil() {
    pupil_thresholded_image_.download(cpu_image_);

    cv::findContours(cpu_image_, contours_, hierarchy_, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);
    cv::Point2f best_centre{};
    float best_radius{};
    float best_rating{0};

    static cv::Size2f image_size{pupil_thresholded_image_.size()};
    static cv::Point2f image_centre{
        cv::Point2f(image_size.width / 2, image_size.height / 2)};
    static float max_distance{std::max(image_size.width, image_size.height)
                              / 2.0f};

    for (const std::vector<cv::Point> &contour : contours_) {
        cv::Point2f centre;
        float radius;

        cv::minEnclosingCircle(contour, centre, radius);
        if (static_cast<int>(radius)
                < Settings::parameters.user_params->min_pupil_radius
            or static_cast<int>(radius)
                > Settings::parameters.user_params->max_pupil_radius)
            continue;

        float distance = euclideanDistance(centre, image_centre);
        //        if (distance > max_distance)
        //            continue;

        const float contour_area = static_cast<float>(cv::contourArea(contour));
        //        if (contour_area <= 0)
        //            continue;
        const float circle_area = 3.1415926f * powf(radius, 2);
        float rating =
            contour_area / circle_area * (1.0f - distance / max_distance);
        if (rating >= best_rating) {
            best_centre = centre;
            best_rating = rating;
            best_radius = radius;
        }
    }

    if (best_rating == 0) {
        return false;
    }

    pupil_kalman_.correct((KFMat(2, 1) << best_centre.x, best_centre.y));
    mtx_features_.lock();
    pupil_location_ = toPoint(pupil_kalman_.predict());
    pupil_radius_ = best_radius;
    mtx_features_.unlock();
    return true;
}

bool FeatureDetector::findGlints() {
    glints_thresholded_image_.download(cpu_image_);
    float best_rating{0};

    cv::findContours(cpu_image_, contours_, hierarchy_, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);

    static cv::Size2f image_size{glints_thresholded_image_.size()};
    static cv::Point2f image_centre{
        cv::Point2f(image_size.width / 2, image_size.height / 2)};
    static float max_distance{std::max(image_size.width, image_size.height)
                              / 2.0f};

    std::vector<GlintCandidate> glint_candidates{};
    glint_candidates.reserve(contours_.size());

    for (const std::vector<cv::Point> &contour : contours_) {
        cv::Point2f centre;
        float radius;
        cv::minEnclosingCircle(contour, centre, radius);
        if (radius > Settings::parameters.user_params->max_glint_radius
            || radius < Settings::parameters.user_params->min_glint_radius)
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

    if (glint_candidates.size() < Settings::parameters.leds_positions.size()) {
        return false;
    }
    size_t led_count{Settings::parameters.leds_positions.size()};
    size_t leds_per_row{led_count / 2};

    GlintCandidate *best_glint{};
    findBestGlintPair(glint_candidates);
    for (auto &glint : glint_candidates) {
        float rating{0};
        int neighbour_count{0};
        GlintCandidate *current_glint = &glint;
        while (current_glint) {
            rating += current_glint->rating;
            neighbour_count++;
            current_glint = current_glint->right_neighbour;
        }
        current_glint = glint.bottom_neighbour;
        while (current_glint) {
            rating += current_glint->rating;
            neighbour_count++;
            current_glint = current_glint->right_neighbour;
        }
        glint.rating = rating;
        glint.neighbour_count = neighbour_count;
        if (!best_glint || neighbour_count > best_glint->neighbour_count
            || (neighbour_count == best_glint->neighbour_count
                && rating > best_glint->rating)) {
            best_glint = &glint;
        }
    }

    std::vector<cv::Vec2f> glints{};
    GlintCandidate *current_glint = best_glint;
    int added_glints{0};
    while (current_glint && added_glints < leds_per_row) {
        glints.push_back(current_glint->location);
        current_glint = current_glint->right_neighbour;
        added_glints++;
    }
    current_glint = best_glint->bottom_neighbour;
    added_glints = 0;
    while (current_glint && added_glints < leds_per_row) {
        glints.push_back(current_glint->location);
        current_glint = current_glint->right_neighbour;
        added_glints++;
    }

    cv::Vec2f glints_centre{};
    cv::Vec2f new_glints_centre{};

    for (int i = 0; i < glints.size(); i++) {
        glints_centre += glints[i];
    }

    glints_centre(0) /= glints.size();
    glints_centre(1) /= glints.size();

    leds_kalman_.correct((KFMat(2, 1) << glints_centre(0), glints_centre(1)));
    new_glints_centre = toPoint(leds_kalman_.predict());
    for (int i = 0; i < glints.size(); i++) {
        mtx_features_.lock();
        glint_locations_[i] = glints[i] + (new_glints_centre - glints_centre);
        mtx_features_.unlock();
    }

    return true;
}

cv::KalmanFilter FeatureDetector::makeKalmanFilter(const cv::Size2i &resolution,
                                                   float framerate) {
    constexpr static double VELOCITY_DECAY = 1.0;
    const static cv::Mat TRANSITION_MATRIX =
        (KFMat(4, 4) << 1, 0, 1.0f / framerate, 0, 0, 1, 0, 1.0f / framerate, 0,
         0, VELOCITY_DECAY, 0, 0, 0, 0, VELOCITY_DECAY);
    const static cv::Mat MEASUREMENT_MATRIX =
        (KFMat(2, 4) << 1, 0, 0, 0, 0, 1, 0, 0);
    const static cv::Mat PROCESS_NOISE_COV = cv::Mat::eye(4, 4, CV_64F) * 1000;
    const static cv::Mat MEASUREMENT_NOISE_COV =
        cv::Mat::eye(2, 2, CV_64F) * 0.1;
    const static cv::Mat ERROR_COV_POST = cv::Mat::eye(4, 4, CV_64F) * 0.1;
    const static cv::Mat STATE_POST =
        (KFMat(4, 1) << resolution.width / 2.0, resolution.height / 2.0, 0, 0);

    cv::KalmanFilter KF(4, 2);
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

cv::Mat FeatureDetector::getThresholdedPupilImage() {
    cv::Mat image_{};
    pupil_thresholded_image_.download(image_);
    return image_;
}

cv::Mat FeatureDetector::getThresholdedGlintsImage() {
    cv::Mat image_{};
    glints_thresholded_image_.download(image_);
    return image_;
}

void FeatureDetector::findBestGlintPair(
    std::vector<GlintCandidate> &glint_candidates) {
    FeaturesParams *params = Settings::parameters.user_params;
    for (auto &first : glint_candidates) {
        for (auto &second : glint_candidates) {
            if (&first == &second) {
                continue;
            }
            if (first.bottom_neighbour && first.right_neighbour) {
                break;
            }

            cv::Point2f distance = second.location - first.location;
            if (distance.y < params->glint_bottom_vert_distance[1]
                && distance.y > params->glint_bottom_vert_distance[0]
                && distance.x < params->glint_bottom_hor_distance[1]
                && distance.x > params->glint_bottom_hor_distance[0]) {
                first.bottom_neighbour = &second;
            }
            if (distance.y < params->glint_right_vert_distance[1]
                && distance.y > params->glint_right_vert_distance[0]
                && distance.x < params->glint_right_hor_distance[1]
                && distance.x > params->glint_right_hor_distance[0]) {
                first.right_neighbour = &second;
            }
        }
    }
}

} // namespace et