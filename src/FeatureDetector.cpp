#include "FeatureDetector.hpp"

#include <opencv2/cudaarithm.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>

using KFMat = cv::Mat_<double>;

namespace et {

void FeatureDetector::initialize() {
    for (int i = 0; i < 2; i++) {
        pupil_kalman_[i] = makeKalmanFilter(
            et::Settings::parameters.camera_params[i].region_of_interest,
            et::Settings::parameters.camera_params[i].framerate);
        leds_kalman_[i] = makeKalmanFilter(
            et::Settings::parameters.camera_params[i].region_of_interest,
            et::Settings::parameters.camera_params[i].framerate);
        glint_locations_[i].resize(Settings::parameters.leds_positions[i].size());
    }

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
}

cv::Point2f FeatureDetector::getPupil(int camera_id) {
    return pupil_location_[camera_id];
}

void FeatureDetector::getPupil(cv::Point2f &pupil, int camera_id) {
    mtx_features_.lock();
    pupil = pupil_location_[camera_id];
    mtx_features_.unlock();
}

void FeatureDetector::getPupilFiltered(cv::Point2f &pupil, int camera_id) {
    mtx_features_.lock();
    pupil = pupil_location_filtered_[camera_id];
    mtx_features_.unlock();
}

int FeatureDetector::getPupilRadius(int camera_id) const {
    return pupil_radius_[camera_id];
}

std::vector<cv::Point2f> *FeatureDetector::getGlints(int camera_id) {
    return &glint_locations_[camera_id];
}

void FeatureDetector::getGlints(std::vector<cv::Point2f> &glint_locations,
                                int camera_id) {
    mtx_features_.lock();
    glint_locations = glint_locations_[camera_id];
    mtx_features_.unlock();
}

bool FeatureDetector::findPupil(const cv::Mat &image, int camera_id) {
    gpu_image_.upload(image);
    cv::cuda::threshold(
        gpu_image_, pupil_thresholded_image_[camera_id],
        Settings::parameters.user_params->pupil_threshold[camera_id], 255,
        cv::THRESH_BINARY_INV);
    if (pupil_erode_filter_) {
        pupil_erode_filter_->apply(pupil_thresholded_image_[camera_id],
                                   pupil_thresholded_image_[camera_id]);
    }
    if (pupil_dilate_filter_) {
        pupil_dilate_filter_->apply(pupil_thresholded_image_[camera_id],
                                    pupil_thresholded_image_[camera_id]);
    }
    if (pupil_close_filter_) {
        pupil_close_filter_->apply(pupil_thresholded_image_[camera_id],
                                   pupil_thresholded_image_[camera_id]);
    }
    pupil_thresholded_image_[camera_id].download(cpu_image_);

    cv::findContours(cpu_image_, contours_, hierarchy_, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);
    cv::Point2f best_centre{};
    float best_radius{};
    float best_rating{0};

    cv::Point2f image_centre{
        Settings::parameters.detection_params.pupil_search_centre[camera_id]};
    int max_distance{
        Settings::parameters.detection_params.pupil_search_radius[camera_id]};

    for (const std::vector<cv::Point> &contour : contours_) {
        cv::Point2f centre;
        float radius;

        cv::minEnclosingCircle(contour, centre, radius);
        if (radius < Settings::parameters.detection_params.min_pupil_radius
            or radius > Settings::parameters.detection_params.max_pupil_radius)
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

    pupil_kalman_[camera_id].correct(
        (KFMat(2, 1) << best_centre.x, best_centre.y));
    mtx_features_.lock();
    pupil_location_[camera_id] = toPoint(pupil_kalman_[camera_id].predict());
    pupil_radius_[camera_id] = best_radius;
    //    pupil_location_ = best_centre; //Only for testing
    mtx_features_.unlock();
    return true;
}

bool FeatureDetector::findGlints(const cv::Mat &image, int camera_id) {
    gpu_image_.upload(image);
    cv::cuda::threshold(
        gpu_image_, glints_thresholded_image_[camera_id],
        Settings::parameters.user_params->glint_threshold[camera_id], 255,
        cv::THRESH_BINARY);

    if (glints_erode_filter_) {
        glints_erode_filter_->apply(glints_thresholded_image_[camera_id],
                                    glints_thresholded_image_[camera_id]);
    }
    if (glints_dilate_filter_) {
        glints_dilate_filter_->apply(glints_thresholded_image_[camera_id],
                                     glints_thresholded_image_[camera_id]);
    }
    if (glints_close_filter_) {
        glints_close_filter_->apply(glints_thresholded_image_[camera_id],
                                    glints_thresholded_image_[camera_id]);
    }
    glints_thresholded_image_[camera_id].download(cpu_image_);

    cv::findContours(cpu_image_, contours_, hierarchy_, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);

    cv::Point2f image_centre{
        Settings::parameters.detection_params.pupil_search_centre[camera_id]};
    int max_distance{
        Settings::parameters.detection_params.pupil_search_radius[camera_id]};

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

        if (!isInEllipse(centre, pupil_location_[camera_id])) {
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
    size_t led_count{Settings::parameters.leds_positions[camera_id].size()};

    std::sort(glint_candidates.begin(), glint_candidates.end(),
              [](const GlintCandidate &a, const GlintCandidate &b) {
                  return a.rating > b.rating;
              });
    findBestGlintPair(glint_candidates);
    determineGlintTypes(glint_candidates);

    for (int i = 0; i < 6; i++) {
        selected_glints_[camera_id][i].found = false;
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
                if (selected_glints_[camera_id][glint.glint_type].found
                    && selected_glints_[camera_id][glint.glint_type].rating
                        >= glint.rating) {
                    continue;
                }

                if (glint.glint_type % 3 != 0
                    && !isLeftNeighbour(
                        glint,
                        selected_glints_[camera_id][glint.glint_type - 1])) {
                    continue;
                }
                if (glint.glint_type % 3 != 2
                    && !isRightNeighbour(
                        glint,
                        selected_glints_[camera_id][glint.glint_type + 1])) {
                    continue;
                }
                if (glint.glint_type / 3 == 0
                    && !isBottomNeighbour(
                        glint,
                        selected_glints_[camera_id][glint.glint_type + 3])) {
                    continue;
                }
                if (glint.glint_type / 3 == 1
                    && !isUpperNeighbour(
                        glint,
                        selected_glints_[camera_id][glint.glint_type - 3])) {
                    continue;
                }
            }

            glint_found = true;
            selected_glints_[camera_id][glint.glint_type] = glint;
            selected_glints_[camera_id][glint.glint_type].found = true;
            identifyNeighbours(&glint, camera_id);
            approximatePositions(camera_id);
        }
    }

    int found_glints{0};
    for (int i = 0; i < 6; i++) {
        if (selected_glints_[camera_id][i].found) {
            found_glints++;
        }
    }

    if (found_glints == 0) {
        return false;
    }

    if (!selected_glints_[camera_id][0].found
        && !selected_glints_[camera_id][3].found) {
        selected_glints_[camera_id][0] = selected_glints_[camera_id][1];
        selected_glints_[camera_id][1] = selected_glints_[camera_id][2];
        selected_glints_[camera_id][3] = selected_glints_[camera_id][4];
        selected_glints_[camera_id][4] = selected_glints_[camera_id][5];
        selected_glints_[camera_id][2].found = false;
        selected_glints_[camera_id][5].found = false;
        approximatePositions(camera_id);
    }

    cv::Point2f glints_centre{};
    cv::Point2f new_glints_centre{};

    for (int i = 0; i < 6; i++) {
        glints_centre += selected_glints_[camera_id][i].location;
    }

    glints_centre.x /= led_count;
    glints_centre.y /= led_count;

    leds_kalman_[camera_id].correct(
        (KFMat(2, 1) << glints_centre.x, glints_centre.y));
    new_glints_centre = toPoint(leds_kalman_[camera_id].predict());
    for (int i = 0; i < led_count; i++) {
        mtx_features_.lock();
        glint_locations_[camera_id][i] = selected_glints_[camera_id][i].location
            + (new_glints_centre - glints_centre);
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

cv::Mat FeatureDetector::getThresholdedPupilImage(int camera_id) {
    cv::Mat image_{};
    pupil_thresholded_image_[camera_id].download(image_);
    return image_;
}

cv::Mat FeatureDetector::getThresholdedGlintsImage(int camera_id) {
    cv::Mat image_{};
    glints_thresholded_image_[camera_id].download(image_);
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

void FeatureDetector::approximatePositions(int camera_id) {
    cv::Point2f mean_right_move{15, 0};
    cv::Point2f mean_down_move{0, 40};
    cv::Point2f total_right_move{0};
    int total_right_found{0};
    cv::Point2f total_down_move{0};
    int total_down_found{0};

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            if (selected_glints_[camera_id][i * 3 + j].found
                && selected_glints_[camera_id][i * 3 + j + 1].found) {
                total_right_move +=
                    selected_glints_[camera_id][i * 3 + j + 1].location
                    - selected_glints_[camera_id][i * 3 + j].location;
                total_right_found++;
            }
        }
    }
    for (int i = 0; i < 3; i++) {
        if (selected_glints_[camera_id][i].found
            && selected_glints_[camera_id][i + 3].found) {
            total_down_move += selected_glints_[camera_id][i + 3].location
                - selected_glints_[camera_id][i].location;
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
        if (!selected_glints_[camera_id][i].found) {
            for (int j = 0; j < 6; j++) {
                if (selected_glints_[camera_id][j].found) {
                    selected_glints_[camera_id][i].location =
                        selected_glints_[camera_id][j].location
                        - (j % 3 - i % 3) * mean_right_move
                        - (j / 3 - i / 3) * mean_down_move;
                    selected_glints_[camera_id][i].rating = 0;
                    break;
                }
            }
        }
    }
}

void FeatureDetector::getPupilGlintVector(cv::Vec2f &pupil_glint_vector,
                                          int camera_id) {
    mtx_features_.lock();
    pupil_glint_vector =
        pupil_location_[camera_id] - glint_locations_[camera_id][0];
    mtx_features_.unlock();
}

void FeatureDetector::getPupilGlintVectorFiltered(cv::Vec2f &pupil_glint_vector,
                                                  int camera_id) {
    mtx_features_.lock();
    pupil_glint_vector = pupil_location_filtered_[camera_id]
        - glint_location_filtered_[camera_id];
    mtx_features_.unlock();
}

bool FeatureDetector::findEllipse(const cv::Mat &image, int camera_id) {
    gpu_image_.upload(image);
    cv::cuda::threshold(
        gpu_image_, glints_thresholded_image_[camera_id],
        Settings::parameters.user_params->glint_threshold[camera_id], 255,
        cv::THRESH_BINARY);

    if (glints_erode_filter_) {
        glints_erode_filter_->apply(glints_thresholded_image_[camera_id],
                                    glints_thresholded_image_[camera_id]);
    }
    if (glints_dilate_filter_) {
        glints_dilate_filter_->apply(glints_thresholded_image_[camera_id],
                                     glints_thresholded_image_[camera_id]);
    }
    if (glints_close_filter_) {
        glints_close_filter_->apply(glints_thresholded_image_[camera_id],
                                    glints_thresholded_image_[camera_id]);
    }
    glints_thresholded_image_[camera_id].download(cpu_image_);

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
    glint_ellipse_[camera_id] = cv::fitEllipse(glints);
    return true;
}

cv::RotatedRect FeatureDetector::getEllipse(int camera_id) {
    return glint_ellipse_[camera_id];
}

void FeatureDetector::setGazeBufferSize(uint8_t value) {
    buffer_size_ = value;
}

void FeatureDetector::updateGazeBuffer() {
    for (int i = 0; i < 2; i++) {
        if (pupil_location_buffer_[i].size() != buffer_size_
            || glint_location_buffer_[i].size() != buffer_size_) {
            pupil_location_buffer_[i].resize(buffer_size_);
            glint_location_buffer_[i].resize(buffer_size_);
            buffer_idx_ = 0;
            buffer_summed_count_ = 0;
            pupil_location_summed_[i].x = 0.0f;
            pupil_location_summed_[i].y = 0.0f;
            glint_location_summed_[i].x = 0.0f;
            glint_location_summed_[i].y = 0.0f;
        }

        if (buffer_summed_count_ == buffer_size_) {
            pupil_location_summed_[i] -= pupil_location_buffer_[i][buffer_idx_];
            glint_location_summed_[i] -= glint_location_buffer_[i][buffer_idx_];
        }

        pupil_location_buffer_[i][buffer_idx_] = pupil_location_[i];
        glint_location_buffer_[i][buffer_idx_] = glint_locations_[i][0];
        pupil_location_summed_[i] += pupil_location_buffer_[i][buffer_idx_];
        glint_location_summed_[i] += glint_location_buffer_[i][buffer_idx_];
    }
 
    if (buffer_summed_count_ != buffer_size_) {
        buffer_summed_count_++;
    }
    
    for (int i = 0; i < 2; i++) {
        mtx_features_.lock();
        pupil_location_filtered_[i] =
            pupil_location_summed_[i] / buffer_summed_count_;
        glint_location_filtered_[i] =
            glint_location_summed_[i] / buffer_summed_count_;
        mtx_features_.unlock();
    }

    buffer_idx_ = (buffer_idx_ + 1) % buffer_size_;
}

void FeatureDetector::identifyNeighbours(GlintCandidate *glint_candidate,
                                         int camera_id) {
    if (glint_candidate->upper_neighbour && glint_candidate->glint_type >= 3
        && !selected_glints_[camera_id][glint_candidate->glint_type - 3]
                .found) {
        selected_glints_[camera_id][glint_candidate->glint_type - 3] =
            *glint_candidate->upper_neighbour;
        selected_glints_[camera_id][glint_candidate->glint_type - 3].found =
            true;
        glint_candidate->upper_neighbour->glint_type =
            (GlintType) (glint_candidate->glint_type - 3);
        identifyNeighbours(glint_candidate->upper_neighbour, camera_id);
    }
    if (glint_candidate->bottom_neighbour && glint_candidate->glint_type < 3
        && !selected_glints_[camera_id][glint_candidate->glint_type + 3]
                .found) {
        selected_glints_[camera_id][glint_candidate->glint_type + 3] =
            *glint_candidate->bottom_neighbour;
        selected_glints_[camera_id][glint_candidate->glint_type + 3].found =
            true;
        glint_candidate->bottom_neighbour->glint_type =
            (GlintType) (glint_candidate->glint_type + 3);
        identifyNeighbours(glint_candidate->bottom_neighbour, camera_id);
    }
    if (glint_candidate->right_neighbour && glint_candidate->glint_type % 3 < 2
        && !selected_glints_[camera_id][glint_candidate->glint_type + 1]
                .found) {
        selected_glints_[camera_id][glint_candidate->glint_type + 1] =
            *glint_candidate->right_neighbour;
        selected_glints_[camera_id][glint_candidate->glint_type + 1].found =
            true;
        glint_candidate->right_neighbour->glint_type =
            (GlintType) (glint_candidate->glint_type + 1);
        identifyNeighbours(glint_candidate->right_neighbour, camera_id);
    }
    if (glint_candidate->left_neighbour && glint_candidate->glint_type % 3 > 0
        && !selected_glints_[camera_id][glint_candidate->glint_type - 1]
                .found) {
        selected_glints_[camera_id][glint_candidate->glint_type - 1] =
            *glint_candidate->left_neighbour;
        selected_glints_[camera_id][glint_candidate->glint_type - 1].found =
            true;
        glint_candidate->left_neighbour->glint_type =
            (GlintType) (glint_candidate->glint_type - 1);
        identifyNeighbours(glint_candidate->left_neighbour, camera_id);
    }
}
} // namespace et