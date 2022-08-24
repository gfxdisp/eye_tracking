#ifndef FEATURE_DETECTOR_H
#define FEATURE_DETECTOR_H

#include "Settings.hpp"

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/opencv.hpp>

#include <mutex>
#include <vector>

namespace et {

enum GlintType {
    UpperLeft = 0,
    UpperCentre,
    UpperRight,
    BottomLeft,
    BottomCentre,
    BottomRight,
    Unknown
};

struct GlintCandidate {
    cv::Point2f location{};
    float rating{};
    int neighbour_count{};
    GlintCandidate *left_neighbour{};
    GlintCandidate *right_neighbour{};
    GlintCandidate *bottom_neighbour{};
    GlintCandidate *upper_neighbour{};
    GlintType glint_type{};
    bool found{};
};

class FeatureDetector {
public:
    void initialize(const cv::Size2i &resolution, float framerate);

    bool findPupil(const cv::Mat &image);

    bool findGlints(const cv::Mat &image);

    bool findEllipse(const cv::Mat &image);

    cv::Point2f getPupil();

    void getPupilGlintVector(cv::Vec2f &pupil_glint_vector);

    void getPupil(cv::Point2f &pupil);

    void getPupilGlintVectorFiltered(cv::Vec2f &pupil_glint_vector);

    void getPupilFiltered(cv::Point2f &pupil);

    [[nodiscard]] int getPupilRadius() const;

    void getPupilRadius(int &pupil_radius);

    std::vector<cv::Point2f> *getGlints();

    cv::RotatedRect getEllipse();

    void getGlints(std::vector<cv::Point2f> &glint_locations);

    cv::Mat getThresholdedPupilImage();

    cv::Mat getThresholdedGlintsImage();

    void setGazeBufferSize(uint8_t value);
    void updateGazeBuffer();

private:


    static cv::KalmanFilter makeKalmanFilter(const cv::Size2i &resolution,
                                             float framerate);

    void findBestGlintPair(std::vector<GlintCandidate> &glint_candidates);
    void determineGlintTypes(std::vector<GlintCandidate> &glint_candidates);

    std::mutex mtx_features_{};

    int pupil_radius_{0};
    cv::KalmanFilter pupil_kalman_{};
    cv::KalmanFilter leds_kalman_{};
    cv::RotatedRect glint_ellipse_{};

    cv::Point2f pupil_location_{};
    std::vector<cv::Point2f> glint_locations_{};

    int buffer_size_{8};
    int buffer_idx_{0};
    int buffer_summed_count_{0};
    std::vector<cv::Point2f> pupil_location_buffer_{};
    std::vector<cv::Point2f> glint_location_buffer_{};
    cv::Point2f pupil_location_summed_{};
    cv::Point2f glint_location_summed_{};
    cv::Point2f pupil_location_filtered_{};
    cv::Point2f glint_location_filtered_{};

    cv::Mat cpu_image_{};
    cv::cuda::GpuMat gpu_image_{};
    cv::cuda::GpuMat pupil_thresholded_image_{};
    cv::cuda::GpuMat glints_thresholded_image_{};

    cv::Ptr<cv::cuda::Filter> glints_dilate_filter_{};
    cv::Ptr<cv::cuda::Filter> glints_erode_filter_{};
    cv::Ptr<cv::cuda::Filter> glints_close_filter_{};
    int glints_dilate_size_{3};
    int glints_erode_size_{3};
    int glints_close_size_{0};

    cv::Ptr<cv::cuda::Filter> pupil_dilate_filter_{};
    cv::Ptr<cv::cuda::Filter> pupil_erode_filter_{};
    cv::Ptr<cv::cuda::Filter> pupil_close_filter_{};
    int pupil_dilate_size_{3};
    int pupil_erode_size_{3};
    int pupil_close_size_{7};

    static bool isLeftNeighbour(GlintCandidate &reference, GlintCandidate &compared);
    static bool isRightNeighbour(GlintCandidate &reference, GlintCandidate &compared);
    static bool isUpperNeighbour(GlintCandidate &reference, GlintCandidate &compared);
    static bool isBottomNeighbour(GlintCandidate &reference, GlintCandidate &compared);
    void approximatePositions();

    std::vector<std::vector<cv::Point>> contours_{};
    std::vector<cv::Vec4i> hierarchy_{}; // Unused output

    GlintCandidate selected_glints_[6]{};

    bool rotated_video_{false};

    static inline cv::Point2f toPoint(cv::Mat m) {
        return {(float) m.at<double>(0, 0), (float) m.at<double>(0, 1)};
    }

    static inline float euclideanDistance(cv::Point2f &p, cv::Point2f &q) {
        cv::Point2f diff = p - q;
        return cv::sqrt(diff.x * diff.x + diff.y * diff.y);
    }

    static inline bool isInEllipse(cv::Point2f &point, cv::Point2f &centre) {
        float semi_major =
            Settings::parameters.user_params->max_hor_glint_pupil_distance;
        float semi_minor =
            Settings::parameters.user_params->max_vert_glint_pupil_distance;
        float major = ((point.x - centre.x) * (point.x - centre.x))
            / (semi_major * semi_major);
        float minor = ((point.y - centre.y) * (point.y - centre.y))
            / (semi_minor * semi_minor);
        return major + minor <= 1;
    }
};
} // namespace et

#endif