#include "FeatureDetector.hpp"

#include <opencv2/cudaarithm.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>

using KFMat = cv::Mat_<double>;

namespace et {

void FeatureDetector::initialize(const cv::Size2i &resolution,
                                 float framerate) {
    pupil_kalman_ = makeKalmanFilter(resolution, framerate);
    leds_kalman_ = makeKalmanFilter(resolution, framerate);
    glint_locations_.resize(Settings::parameters.leds_positions.size());

    cv::Mat morphology_element{};
    int size{};

    size = pupil_close_size_;
    if (size) {
        morphology_element =
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(size, size));
        pupil_close_filter_ = cv::cuda::createMorphologyFilter(
            cv::MORPH_CLOSE, CV_8UC1, morphology_element);
    }
    size = pupil_dilate_size_;
    if (size) {
        morphology_element =
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(size, size));
        pupil_dilate_filter_ = cv::cuda::createMorphologyFilter(
            cv::MORPH_DILATE, CV_8UC1, morphology_element);
    }
    size = pupil_erode_size_;
    if (size) {
        morphology_element =
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(size, size));
        pupil_erode_filter_ = cv::cuda::createMorphologyFilter(
            cv::MORPH_ERODE, CV_8UC1, morphology_element);
    }

    size = glints_close_size_;
    if (size) {
        morphology_element =
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(size, size));
        glints_close_filter_ = cv::cuda::createMorphologyFilter(
            cv::MORPH_CLOSE, CV_8UC1, morphology_element);
    }
    size = glints_dilate_size_;
    if (size) {
        morphology_element =
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(size, size));
        glints_dilate_filter_ = cv::cuda::createMorphologyFilter(
            cv::MORPH_DILATE, CV_8UC1, morphology_element);
    }
    size = glints_erode_size_;
    if (size) {
        morphology_element =
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(size, size));
        glints_erode_filter_ = cv::cuda::createMorphologyFilter(
            cv::MORPH_ERODE, CV_8UC1, morphology_element);
    }
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

bool FeatureDetector::findPupil(const cv::Mat &image) {
    gpu_image_.upload(image);
    cv::cuda::threshold(gpu_image_, pupil_thresholded_image_,
                        Settings::parameters.user_params->pupil_threshold, 255,
                        cv::THRESH_BINARY_INV);
    if (pupil_erode_filter_) {
        pupil_erode_filter_->apply(pupil_thresholded_image_,
                                   pupil_thresholded_image_);
    }
    if (pupil_dilate_filter_) {
        pupil_dilate_filter_->apply(pupil_thresholded_image_,
                                    pupil_thresholded_image_);
    }
    if (pupil_close_filter_) {
        pupil_close_filter_->apply(pupil_thresholded_image_,
                                   pupil_thresholded_image_);
    }
    pupil_thresholded_image_.download(cpu_image_);

    cv::findContours(cpu_image_, contours_, hierarchy_, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);
    cv::Point2f best_centre{};
    float best_radius{};
    float best_rating{0};

    static cv::Size2f image_size{pupil_thresholded_image_.size()};
    static cv::Point2f image_centre{
        cv::Point2f(image_size.width / 2, image_size.height / 2)};
    static float max_distance{std::min(image_size.width, image_size.height)
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

    pupil_kalman_.correct((KFMat(2, 1) << best_centre.x, best_centre.y));
    mtx_features_.lock();
    pupil_location_ = toPoint(pupil_kalman_.predict());
    pupil_radius_ = best_radius;
    pupil_location_ = best_centre; //Only for testing
    mtx_features_.unlock();
    return true;
}

bool FeatureDetector::findGlints(const cv::Mat &image) {
    gpu_image_.upload(image);
    cv::cuda::threshold(gpu_image_, glints_thresholded_image_,
                        Settings::parameters.user_params->glint_threshold, 255,
                        cv::THRESH_BINARY);

    if (glints_erode_filter_) {
        glints_erode_filter_->apply(glints_thresholded_image_,
                                    glints_thresholded_image_);
    }
    if (glints_dilate_filter_) {
        glints_dilate_filter_->apply(glints_thresholded_image_,
                                     glints_thresholded_image_);
    }
    if (glints_close_filter_) {
        glints_close_filter_->apply(glints_thresholded_image_,
                                    glints_thresholded_image_);
    }
    glints_thresholded_image_.download(cpu_image_);

    float best_rating{0};

    cv::findContours(cpu_image_, contours_, hierarchy_, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);

    static cv::Size2f image_size{glints_thresholded_image_.size()};
    static cv::Point2f image_centre{
        cv::Point2f(image_size.width / 2, image_size.height / 2)};
    static float max_distance{std::min(image_size.width, image_size.height)
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

        float distance = euclideanDistance(centre, image_centre);
//        if (distance > max_distance)
//            continue;

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
    size_t led_count{Settings::parameters.leds_positions.size()};
    size_t leds_per_row{led_count / 2};

    GlintCandidate *best_glint{};
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
    for (auto &glint : glint_candidates) {
        if (glint.glint_type == GlintType::Unknown) {
            continue;
        }

        if (glint_found) {
            if (selected_glints_[glint.glint_type].found
                && selected_glints_[glint.glint_type].rating > glint.rating) {
                continue;
            }

            if (glint.glint_type % 3 != 0
                && !isLeftNeighbour(glint,
                                    selected_glints_[glint.glint_type - 1])) {
                continue;
            }
            if (glint.glint_type % 3 != 2
                && !isRightNeighbour(glint,
                                     selected_glints_[glint.glint_type + 1])) {
                continue;
            }
            if (glint.glint_type / 3 == 0
                && !isBottomNeighbour(glint,
                                      selected_glints_[glint.glint_type + 3])) {
                continue;
            }
            if (glint.glint_type / 3 == 1
                && !isUpperNeighbour(glint,
                                     selected_glints_[glint.glint_type - 3])) {
                continue;
            }
        }

        glint_found = true;
        selected_glints_[glint.glint_type] = glint;
        selected_glints_[glint.glint_type].found = true;

        if (glint.upper_neighbour
            && !selected_glints_[glint.glint_type - 3].found) {
            selected_glints_[glint.glint_type - 3] = *glint.upper_neighbour;
            selected_glints_[glint.glint_type - 3].found = true;
        }
        if (glint.bottom_neighbour
            && !selected_glints_[glint.glint_type + 3].found) {
            selected_glints_[glint.glint_type + 3] = *glint.bottom_neighbour;
            selected_glints_[glint.glint_type + 3].found = true;
        }
        if (glint.right_neighbour
            && !selected_glints_[glint.glint_type + 1].found) {
            selected_glints_[glint.glint_type + 1] = *glint.right_neighbour;
            selected_glints_[glint.glint_type + 1].found = true;
        }
        if (glint.left_neighbour
            && !selected_glints_[glint.glint_type - 1].found) {
            selected_glints_[glint.glint_type - 1] = *glint.left_neighbour;
            selected_glints_[glint.glint_type - 1].found = true;
        }

        approximatePositions();
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

    if (!selected_glints_[0].found && !selected_glints_[3].found) {
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

    for (auto &selected_glint : selected_glints_) {
        glints_centre += selected_glint.location;
    }

    glints_centre.x /= 6;
    glints_centre.y /= 6;

    leds_kalman_.correct((KFMat(2, 1) << glints_centre.x, glints_centre.y));
    new_glints_centre = toPoint(leds_kalman_.predict());
    for (int i = 0; i < 6; i++) {
        mtx_features_.lock();
        glint_locations_[i] =
            selected_glints_[i].location + (new_glints_centre - glints_centre);
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
                }
                else {
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
                }
                else {
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
    FeaturesParams *params{Settings::parameters.user_params};

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
    FeaturesParams *params{Settings::parameters.user_params};

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
                total_right_move += selected_glints_[i * 3 + j + 1].location
                    - selected_glints_[i * 3 + j].location;
                total_right_found++;
            }
        }
    }
    for (int i = 0; i < 3; i++) {
        if (selected_glints_[i].found && selected_glints_[i + 3].found) {
            total_down_move +=
                selected_glints_[i + 3].location - selected_glints_[i].location;
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
                    selected_glints_[i].location = selected_glints_[j].location
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
    pupil_glint_vector = pupil_location_ - glint_locations_[0];
    mtx_features_.unlock();
}

bool FeatureDetector::findEllipse(const cv::Mat &image) {
    gpu_image_.upload(image);
    cv::cuda::threshold(gpu_image_, glints_thresholded_image_,
                        Settings::parameters.user_params->glint_threshold, 255,
                        cv::THRESH_BINARY);

    if (glints_erode_filter_) {
        glints_erode_filter_->apply(glints_thresholded_image_,
                                    glints_thresholded_image_);
    }
    if (glints_dilate_filter_) {
        glints_dilate_filter_->apply(glints_thresholded_image_,
                                     glints_thresholded_image_);
    }
    if (glints_close_filter_) {
        glints_close_filter_->apply(glints_thresholded_image_,
                                    glints_thresholded_image_);
    }
    glints_thresholded_image_.download(cpu_image_);

    cv::findContours(cpu_image_, contours_, hierarchy_, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Point2f> glints{};
    glints.reserve(contours_.size());

    for (const std::vector<cv::Point> &contour : contours_) {
        for (const cv::Point &point : contour) {
            glints.push_back(point);
        }
    }
    if (glints.size() < 5) {
        return false;
    }
    glint_ellipse_ = cv::fitEllipse(glints);
    return true;
}

cv::RotatedRect FeatureDetector::getEllipse() {
    return glint_ellipse_;
}

} // namespace et