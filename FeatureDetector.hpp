#ifndef FEATURE_DETECTOR_H
#define FEATURE_DETECTOR_H

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/opencv.hpp>

#include <mutex>
#include <vector>

namespace et {
struct GlintCandidate {
    cv::Point2f location;
    float rating;
    bool found;
};

class FeatureDetector {
public:
    void initializeKalmanFilters(const cv::Size2i &resolution, float framerate);

    bool findImageFeatures(const cv::Mat &image);

    cv::Point2f getPupil();

    void getPupil(cv::Point2f &pupil);

    [[nodiscard]] int getPupilRadius() const;

    void getPupilRadius(int &pupil_radius);

    std::vector<cv::Point2f> *getGlints();

    void getGlints(std::vector<cv::Point2f> &glint_locations);

    cv::Mat getThresholdedPupilImage();

    cv::Mat getThresholdedGlintsImage();

private:
    bool findPupil();

    bool findGlints();

    static cv::KalmanFilter makeKalmanFilter(const cv::Size2i &resolution,
                                             float framerate);

    void findBestGlintPair(std::vector<GlintCandidate> &glint_candidates,
                           std::pair<cv::Point2f, cv::Point2f> &best_pair);

    std::mutex mtx_features_{};

    int pupil_radius_{0};
    cv::Point2f pupil_location_{};
    cv::KalmanFilter pupil_kalman_{};
    std::vector<cv::Point2f> glint_locations_{};
    std::vector<cv::KalmanFilter> led_kalmans_{};

    cv::Mat cpu_image_{};
    cv::cuda::GpuMat gpu_image_{};
    cv::cuda::GpuMat pupil_thresholded_image_{};
    cv::cuda::GpuMat glints_thresholded_image_{};

    std::vector<std::vector<cv::Point>> contours_{};
    std::vector<cv::Vec4i> hierarchy_{};// Unused output

    bool rotated_video_{false};

    static inline cv::Point2f toPoint(cv::Mat m) {
        return {(float) m.at<double>(0, 0), (float) m.at<double>(0, 1)};
    }

    static inline float euclideanDistance(cv::Point2f &p, cv::Point2f &q) {
        cv::Point2f diff = p - q;
        return cv::sqrt(diff.x * diff.x + diff.y * diff.y);
    }
};
}// namespace et

#endif