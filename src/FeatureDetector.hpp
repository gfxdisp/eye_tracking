#ifndef FEATURE_DETECTOR_H
#define FEATURE_DETECTOR_H

#include "BayesMinimizer.hpp"
#include "PolynomialFit.hpp"
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
    virtual ~FeatureDetector();
    void initialize(int camera_id);

    void preprocessGlintEllipse(const cv::Mat& image);
    
    void preprocessIndivGlints(const cv::Mat& image);

    bool findPupil();

    bool findGlints();

    bool findEllipse(const cv::Point2f &pupil);

    cv::Point2f getPupil();

    void getPupilGlintVector(cv::Vec2f &pupil_glint_vector);

    void getPupil(cv::Point2f &pupil);

    void getPupilGlintVectorFiltered(cv::Vec2f &pupil_glint_vector);

    void getPupilFiltered(cv::Point2f &pupil);

    [[nodiscard]] int getPupilRadius() const;

    std::vector<cv::Point2f> *getGlints();

    cv::RotatedRect getEllipse();


    void getGlints(std::vector<cv::Point2f> &glint_locations);

    cv::Mat getThresholdedPupilImage();

    cv::Mat getThresholdedGlintsImage();

    void setGazeBufferSize(uint8_t value);
    void updateGazeBuffer();

private:
    static cv::KalmanFilter makePxKalmanFilter(const cv::Size2i &resolution,
                                               float framerate);
    static cv::KalmanFilter makeRadiusKalmanFilter(const float &min_radius,
                                                   const float &max_radius,
                                                   float framerate);

    static cv::KalmanFilter
    makeEllipseKalmanFilter(const cv::Size2i &resolution, const float &min_axis,
                            const float &max_axis, float framerate);

    void findBestGlintPair(std::vector<GlintCandidate> &glint_candidates);
    void determineGlintTypes(std::vector<GlintCandidate> &glint_candidates);
    void identifyNeighbours(GlintCandidate *glint_candidate);

    std::mutex mtx_features_{};
    int pupil_radius_{0};
    cv::KalmanFilter pupil_kalman_{};
    cv::KalmanFilter leds_kalman_{};
    cv::KalmanFilter pupil_radius_kalman_{};
    cv::KalmanFilter glint_ellipse_kalman_{};

    cv::RotatedRect glint_ellipse_{};
    cv::Point2f pupil_location_{};
    std::vector<cv::Point2f> glint_locations_{};

    BayesMinimizer *bayes_minimizer_{};
    cv::Ptr<cv::DownhillSolver::Function> bayes_minimizer_func_{};
    cv::Ptr<cv::DownhillSolver> bayes_solver_{};
    cv::Point2d ellipse_centre_{};
    double ellipse_radius_{};

    cv::Size2i *region_if_interest_{};
    int *pupil_threshold_{};
    int *glint_threshold_{};
    cv::Point2f *pupil_search_centre_{};
    int *pupil_search_radius_{};
    float *min_pupil_radius_{};
    float *max_pupil_radius_{};

    cv::cuda::GpuMat glints_template_;
    cv::Ptr<cv::cuda::TemplateMatching> template_matcher_{};
    cv::Mat template_crop_{};

    int buffer_size_{16};
    int buffer_idx_{0};
    int buffer_summed_count_{0};
    std::vector<cv::Point2f> pupil_location_buffer_{};
    std::vector<cv::Point2f> glint_location_buffer_{};
    cv::Point2f pupil_location_summed_{};
    cv::Point2f glint_location_summed_{};
    cv::Point2f pupil_location_filtered_{};
    cv::Point2f glint_location_filtered_{};

    cv::cuda::GpuMat gpu_image_{};
    cv::cuda::GpuMat pupil_thresholded_image_gpu_{};
    cv::cuda::GpuMat glints_thresholded_image_gpu_{};
    cv::Mat cpu_image_{};
    cv::Mat pupil_thresholded_image_;
    cv::Mat glints_thresholded_image_;

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
    void approximatePositions();

    std::vector<std::vector<cv::Point>> contours_{};
    std::vector<cv::Vec4i> hierarchy_{}; // Unused output

    int led_count_{};
    GlintCandidate selected_glints_[6]{};

    static inline cv::Point2f toPoint(cv::Mat m) {
        return {(float) m.at<double>(0, 0), (float) m.at<double>(0, 1)};
    }

    static inline float toValue(cv::Mat m) {
        return (float) m.at<double>(0, 0);
    }

    static inline float euclideanDistance(const cv::Point2f &p,
                                          const cv::Point2f &q) {
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

    static inline bool isInEllipse(cv::Point2f &point, cv::RotatedRect &ellipse,
                                   float acceptable_error) {
        float a = (point.x - ellipse.center.x) * std::cos(ellipse.angle);
        float b = (point.y - ellipse.center.y) * std::sin(ellipse.angle);
        float c = (point.x - ellipse.center.x) * std::sin(ellipse.angle);
        float d = (point.y - ellipse.center.y) * std::cos(ellipse.angle);
        float e =
            ((a + b) * (a + b)) / (ellipse.size.width * ellipse.size.width);
        float f =
            ((c + d) * (c + d)) / (ellipse.size.height * ellipse.size.height);
        return std::abs(e + f - 1.0f) < acceptable_error;
    }
};
} // namespace et

#endif