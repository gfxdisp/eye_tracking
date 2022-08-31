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
    void initialize();

    bool findPupil(const cv::Mat &image, int camera_id);

    bool findGlints(const cv::Mat &image, int camera_id);

    bool findEllipse(const cv::Mat &image, int camera_id);

    cv::Point2f getPupil(int camera_id);

    void getPupilGlintVector(cv::Vec2f &pupil_glint_vector, int camera_id);

    void getPupil(cv::Point2f &pupil, int camera_id);

    void getPupilGlintVectorFiltered(cv::Vec2f &pupil_glint_vector,
                                     int camera_id);

    void getPupilFiltered(cv::Point2f &pupil, int camera_id);

    [[nodiscard]] int getPupilRadius(int camera_id) const;

    std::vector<cv::Point2f> *getGlints(int camera_id);

    cv::RotatedRect getEllipse(int camera_id);

    void getGlints(std::vector<cv::Point2f> &glint_locations, int camera_id);

    cv::Mat getThresholdedPupilImage(int camera_id);

    cv::Mat getThresholdedGlintsImage(int camera_id);

    void setGazeBufferSize(uint8_t value);
    void updateGazeBuffer();

private:
    static cv::KalmanFilter makeKalmanFilter(const cv::Size2i &resolution,
                                             float framerate);

    void findBestGlintPair(std::vector<GlintCandidate> &glint_candidates);
    void determineGlintTypes(std::vector<GlintCandidate> &glint_candidates);
    void identifyNeighbours(GlintCandidate *glint_candidate, int camera_id);

    std::mutex mtx_features_{};

    int pupil_radius_[2]{0};
    cv::KalmanFilter pupil_kalman_[2]{};
    cv::KalmanFilter leds_kalman_[2]{};
    cv::RotatedRect glint_ellipse_[2]{};

    cv::Point2f pupil_location_[2]{};
    std::vector<cv::Point2f> glint_locations_[2]{};

    int buffer_size_{16};
    int buffer_idx_{0};
    int buffer_summed_count_{0};
    std::vector<cv::Point2f> pupil_location_buffer_[2]{};
    std::vector<cv::Point2f> glint_location_buffer_[2]{};
    cv::Point2f pupil_location_summed_[2]{};
    cv::Point2f glint_location_summed_[2]{};
    cv::Point2f pupil_location_filtered_[2]{};
    cv::Point2f glint_location_filtered_[2]{};

    cv::Mat cpu_image_{};
    cv::cuda::GpuMat gpu_image_{};
    cv::cuda::GpuMat pupil_thresholded_image_[2];
    cv::cuda::GpuMat glints_thresholded_image_[2];

    cv::Ptr<cv::cuda::Filter> glints_dilate_filter_{};
    cv::Ptr<cv::cuda::Filter> glints_erode_filter_{};
    cv::Ptr<cv::cuda::Filter> glints_close_filter_{};

    cv::Ptr<cv::cuda::Filter> pupil_dilate_filter_{};
    cv::Ptr<cv::cuda::Filter> pupil_erode_filter_{};
    cv::Ptr<cv::cuda::Filter> pupil_close_filter_{};

    static bool isLeftNeighbour(GlintCandidate &reference,
                                GlintCandidate &compared);
    static bool isRightNeighbour(GlintCandidate &reference,
                                 GlintCandidate &compared);
    static bool isUpperNeighbour(GlintCandidate &reference,
                                 GlintCandidate &compared);
    static bool isBottomNeighbour(GlintCandidate &reference,
                                  GlintCandidate &compared);
    void approximatePositions(int camera_id);

    std::vector<std::vector<cv::Point>> contours_{};
    std::vector<cv::Vec4i> hierarchy_{}; // Unused output

    GlintCandidate selected_glints_[2][6]{};

    static inline cv::Point2f toPoint(cv::Mat m) {
        return {(float) m.at<double>(0, 0), (float) m.at<double>(0, 1)};
    }

    static inline float euclideanDistance(cv::Point2f &p, cv::Point2f &q) {
        cv::Point2f diff = p - q;
        return cv::sqrt(diff.x * diff.x + diff.y * diff.y);
    }

    static inline bool isInEllipse(cv::Point2f &point, cv::Point2f &centre) {
        float semi_major =
            Settings::parameters.detection_params.max_hor_glint_pupil_distance;
        float semi_minor =
            Settings::parameters.detection_params.max_vert_glint_pupil_distance;
        float major = ((point.x - centre.x) * (point.x - centre.x))
            / (semi_major * semi_major);
        float minor = ((point.y - centre.y) * (point.y - centre.y))
            / (semi_minor * semi_minor);
        return major + minor <= 1;
    }
};
} // namespace et

#endif