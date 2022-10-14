#ifndef HDRMFS_EYE_TRACKER_FEATURE_DETECTOR_HPP
#define HDRMFS_EYE_TRACKER_FEATURE_DETECTOR_HPP

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

/**
 * Location of glint in 3x2 setup.
 */
enum GlintType {
    UpperLeft = 0,
    UpperCentre,
    UpperRight,
    BottomLeft,
    BottomCentre,
    BottomRight,
    Unknown
};

/**
 * All gathered information about a specific glint.
 */
struct GlintCandidate {
    // Glint location.
    cv::Point2f location{};
    // Glint rating. The higher it is, the more likely it is to be a primary
    // LED reflection.
    float rating{};
    // Pointer to the glint located on the left (if it exists) that lies at
    // an expected distance.
    GlintCandidate *left_neighbour{};
    // Pointer to the glint located on the right (if it exists) that lies at
    // an expected distance.
    GlintCandidate *right_neighbour{};
    // Pointer to the glint located below (if it exists) that lies at
    // an expected distance.
    GlintCandidate *bottom_neighbour{};
    // Pointer to the glint located above (if it exists) that lies at
    // an expected distance.
    GlintCandidate *upper_neighbour{};
    // Location of the glint in the 3x2 setup.
    GlintType glint_type{};
};

/**
 * Detects various eye features on the image directly in the image space.
 */
class FeatureDetector {
public:
    /**
     * Releases the resources used by BayesMinimizer.
     */
    virtual ~FeatureDetector();
    /**
     * Creates Kalman filters, initializes BayesMinimizer, and loads glint template.
     * @param settings_path Path to a folder containing all settings files.
     * @param kalman_filtering_enabled True if kalman filtering is enabled,
     * false otherwise.
     * @param template_matching_enabled True if glints are detected using
     * template matching, false otherwise.
     * @param camera_id An id of the camera to which the object corresponds.
     */
    void initialize(const std::string &settings_path,
                    bool kalman_filtering_enabled,
                    bool template_matching_enabled, int camera_id);

    /**
     * Uploads an image to GPU, thresholds it for glints and pupil detection,
     * and saves to CPU.
     * @param image Image to be preprocessed.
     */
    void preprocessImage(const cv::Mat &image);

    /**
     * Detects a pupil in the image preprocessed using preprocessImage().
     * @return True if the pupil was found. False otherwise.
     */
    bool findPupil();

    /**
     * Detects all glints in the image preprocessed using preprocessImage().
     * @return True if the glints were found. False otherwise.
     */
    bool findGlints();

    /**
     * Detects an ellipse formed from glints in the image preprocessed using
     * preprocessImage().
     * @return True if the glint ellipse was found. False otherwise.
     */
    bool findEllipse();

    /**
     * Retrieves a vector between a specific glint and pupil position
     * based on parameters that were previously calculated in findPupil()
     * and findGlints(). If the ellipse fitting was used through findEllipse(),
     * the vector is calculated between pupil position and ellipse centre.
     * @param pupil_glint_vector Variable that will contain the vector.
     */
    void getPupilGlintVector(cv::Vec2f &pupil_glint_vector);

    /**
     * Retrieves pupil position in image space that was previously
     * calculated in findPupil().
     * @param pupil Variable that will contain the pupil position.
     */
    void getPupil(cv::Point2f &pupil);

    /**
     * Retrieves a vector between a specific glint and pupil position
     * based on parameters that were previously calculated in findPupil()
     * and findGlints(). If the ellipse fitting was used through findEllipse(),
     * the vector is calculated between pupil position and ellipse centre.
     * Value is averaged across a number of frames set using
     * the setGazeBufferSize() method.
     * @param pupil_glint_vector Variable that will contain the vector.
     */
    void getPupilGlintVectorFiltered(cv::Vec2f &pupil_glint_vector);

    /**
     * Retrieves pupil position in image space that was previously
     * calculated in findPupil(). Value is averaged across a number of frames
     * set using the setGazeBufferSize() method.
     * @param pupil Variable that will contain the pupil position.
     */
    void getPupilFiltered(cv::Point2f &pupil);

    /**
     * Retrieves pupil position in image space that was previously
     * calculated in findPupil().
     * @return Pupil position.
     */
    cv::Point2f getPupil();

    /**
     * Retrieves distorted pupil position in image space that was previously
     * calculated in findPupil().
     * @return Pupil position.
     */
    cv::Point2f getDistortedPupil();

    /**
     * Retrieves a radius of a pupil.
     * @return Pupil radius.
     */
    [[nodiscard]] int getPupilRadius() const;

    /**
     * Retrieves a radius of a distorted pupil.
     * @return Pupil radius.
     */
    [[nodiscard]] int getDistortedPupilRadius() const;

    /**
     * Retrieves a pointer to the vector of glints detected using findGlints().
     * @return A vector of glints.
     */
    std::vector<cv::Point2f> *getGlints();

    /**
     * Retrieves a pointer to the vector of distorted glints detected using findGlints().
     * @return A vector of glints.
     */
    std::vector<cv::Point2f> *getDistortedGlints();

    /**
     * Retrieves an ellipse detected using findEllipse().
     * @return A detected ellipse.
     */
    cv::RotatedRect getEllipse();

    /**
     * Retrieves a distorted ellipse detected using findEllipse().
     * @return A detected ellipse.
     */
    cv::RotatedRect getDistortedEllipse();

    /**
     * Retrieves a thresholded image generated using preprocessImage()
     * used for pupil detection.
     * @return A thresholded image.
     */
    cv::Mat getThresholdedPupilImage();

    /**
     * Retrieves a thresholded image generated using preprocessImage()
     * used for glint detection.
     * @return A thresholded image.
     */
    cv::Mat getThresholdedGlintsImage();

    /**
     * Used to set the number of frames, across which getPupilFiltered() and
     * getPupilGlintVectorFiltered() are calculated.
     * @param value Number of frames.
     */
    void setGazeBufferSize(uint8_t value);

    /**
     * Updates the pupil and pupil-glint vectors based on the newest data
     * from findPupil(), findGlints() and findEllipse() averaged across a number
     * of frames set using setGazeBufferSize().
     */
    void updateGazeBuffer();

private:
    /**
     * Creates a 4x4 Kalman Filter assuming its input vector consists of
     * XY pixel position and XY velocity.
     * @param resolution Dimensions of the image space.
     * @param framerate Estimated system framerate used to calculate velocity.
     * @return Kalman filter.
     */
    static cv::KalmanFilter makePxKalmanFilter(const cv::Size2i &resolution,
                                               float framerate);

    /**
     * Creates a 2x2 Kalman Filter assuming its input vector consists of
     * a pixel pupil radius and velocity.
     * @param min_radius Minimum expected radius of the pupil in pixels.
     * @param max_radius Maximum expected radius of the pupil in pixels.
     * @param framerate Estimated system framerate used to calculate velocity.
     * @return Kalman filter.
     */
    static cv::KalmanFilter makeRadiusKalmanFilter(const float &min_radius,
                                                   const float &max_radius,
                                                   float framerate);

    /**
     * Creates a 10x10 Kalman Filter assuming its input vector consists of
     * a XY ellipse centre, ellipse width, ellipse height, ellipse angle,
     * and velocity of all mentioned parameters.
     * @param resolution Dimensions of the image space.
     * @param framerate Estimated system framerate used to calculate velocity.
     * @return Kalman filter.
     */
    static cv::KalmanFilter
    makeEllipseKalmanFilter(const cv::Size2i &resolution, float framerate);

    /**
     * Finds bottom, upper, left, and right neighbours of all glints.
     * @param glint_candidates A vector of all glints that are expected to be
     * primary reflections from LEDs.
     */
    void findPotentialNeighbours(std::vector<GlintCandidate> &glint_candidates);
    /**
     * Estimates the position in the 3x2 LED grid from which the glints come.
     * @param glint_candidates A vector of all glints that are expected to be
     * primary reflections from LEDs.
     */
    static void
    determineGlintTypes(std::vector<GlintCandidate> &glint_candidates);
    /**
     * Finds all known neighbours of the selected glint.
     * @param glint_candidate A pointer to a glint that is known to be primary
     * reflection.
     */
    void identifyNeighbours(GlintCandidate *glint_candidate);

    // Synchronization variable between feature detection and socket server.
    std::mutex mtx_features_{};

    // Kalman filter used to correct noisy pupil position.
    cv::KalmanFilter pupil_kalman_{};
    // Kalman filter used to correct noise glint positions.
    cv::KalmanFilter glints_kalman_{};
    // Kalman filter used to correct noisy pupil radius.
    cv::KalmanFilter pupil_radius_kalman_{};
    // Kalman filter used to correct noisy glint ellipse parameters.
    cv::KalmanFilter glint_ellipse_kalman_{};

    // Pupil location estimated using findPupil().
    cv::Point2f pupil_location_{};
    // Undisorted pupil location estimated using findPupil().
    cv::Point2f pupil_location_undistorted_{};
    // Glint locations estimated using findGlints() or findEllipse().
    // Radius of the pupil in pixels.
    int pupil_radius_{0};
    // Radius of the undistorted pupil in pixels.
    int pupil_radius_undistorted_{0};
    std::vector<cv::Point2f> glint_locations_{};
    // Undistorted glint locations estimated using findGlints() or findEllipse().
    std::vector<cv::Point2f> glint_locations_undistorted_{};
    // Ellipse parameters estimated using findEllipse().
    cv::RotatedRect glint_ellipse_{};
    // Undisorted ellipse parameters estimated using findEllipse().
    cv::RotatedRect glint_ellipse_undistorted_{};

    // Function used to find a circle based on a set of glints and its
    // expected position.
    BayesMinimizer *bayes_minimizer_{};
    // Pointer to a BayesMinimizer function.
    cv::Ptr<cv::DownhillSolver::Function> bayes_minimizer_func_{};
    // Downhill solver optimizer used to find a glint circle.
    cv::Ptr<cv::DownhillSolver> bayes_solver_{};
    // Expected position of a circle on which glints lie.
    cv::Point2d circle_centre_{};
    // Expected radius of a circle on which glints lie.
    double circle_radius_{};

    // True if kalman filtering is enabled, false otherwise.
    bool kalman_filtering_enabled_{};
    // True if glints are detected using template matching, false otherwise.
    bool template_matching_enabled_{};

    // Size of the region-of-interest extracted from the full image.
    cv::Size2i *region_of_interest_{};
    // Threshold value for pupil detection used in preprocessImage().
    int *pupil_threshold_{};
    // Threshold value for glints detection used in preprocessImage().
    int *glint_threshold_{};
    // Centre of the circle aligned with the hole in the view piece in the image.
    cv::Point2f *pupil_search_centre_{};
    // Radius of the circle aligned with the hole in the view piece in the image.
    int *pupil_search_radius_{};
    // Minimal radius of the pupil in pixels.
    float *min_pupil_radius_{};
    // Maximal radius of the pupil in pixels.
    float *max_pupil_radius_{};
    // Minimal radius of the glint in pixels.
    float *min_glint_radius_{};
    // Maximal radius of the glint in pixels.
    float *max_glint_radius_{};
    // Parameters used to increase the eye features detection precision.
    DetectionParams *detection_params_{};
    // Template uploaded to the GPU used to detect glints with preprocessImage().
    cv::cuda::GpuMat glints_template_;
    // Template matching algorithm used to detect glints with preprocessImage().
    cv::Ptr<cv::cuda::TemplateMatching> template_matcher_{};
    // Affine warping matrix used to translate correlation matrix to align with
    // the original image.
    cv::Mat template_crop_{};

    // Intrinsic matrix of the camera.
    cv::Mat *intrinsic_matrix_{};
    // Distance from top-left corner of the region-of-interest to the top-left
    // corner of the full image, measured in pixels separately for every axis.
    cv::Size2i *capture_offset_{};
    // Distortion coefficients of the camera.
    std::vector<float> *distortion_coefficients_{};

    // Size of the buffer used to average pupil position and pupil-glint vector
    // across multiple frames.
    int buffer_size_{4};
    // Index in the buffer of the most recent pupil and glint positions.
    int buffer_idx_{0};
    // Number of total pupil and glint positions.
    int buffer_summed_count_{0};
    // Buffer of the most recent pupil positions.
    std::vector<cv::Point2f> pupil_location_buffer_{};
    // Buffer of the most recent glint positions.
    std::vector<cv::Point2f> glint_location_buffer_{};
    // Summed values from the pupil_location_buffer_ vector.
    cv::Point2f pupil_location_summed_{};
    // Summed values from the glint_location_buffer_ vector.
    cv::Point2f glint_location_summed_{};
    // Mean pupil position averaged across whole pupil_location_buffer_ vector.
    cv::Point2f pupil_location_filtered_{};
    // Mean pupil position averaged across whole glint_location_buffer_ vector.
    cv::Point2f glint_location_filtered_{};

    // GPU matrix to which the camera image is directly uploaded.
    cv::cuda::GpuMat gpu_image_{};
    // GPU matrix which contains the thresholded image used for pupil detection
    // with findPupil().
    cv::cuda::GpuMat pupil_thresholded_image_gpu_{};
    // GPU matrix which contains the thresholded image used for glints detection
    // with findGlints() or findEllipse().
    cv::cuda::GpuMat glints_thresholded_image_gpu_{};
    // pupil_thresholded_image_gpu_ downloaded to CPU.
    cv::Mat pupil_thresholded_image_;
    // glints_thresholded_image_gpu_ downloaded to CPU.
    cv::Mat glints_thresholded_image_;

    // Semi major axis of the ellipse around the pupil where the glints
    // could be found.
    float *bound_ellipse_semi_major_;
    // Semi minor axis of the ellipse around the pupil where the glints
    // could be found.
    float *bound_ellipse_semi_minor_;

    /**
     * Checks if the compared glint is the left neighbour of the reference glint.
     * @param reference Glint expected to be a primary reflection from an LED.
     * @param compared Checked left-neighbour of the reference glint expected
     * to be a primary reflection from an LED.
     * @return True, if compared is the left neighbour of reference. False otherwise.
     */
    bool isLeftNeighbour(GlintCandidate &reference, GlintCandidate &compared);
    /**
     * Checks if the compared glint is the right neighbour of the reference glint.
     * @param reference Glint expected to be a primary reflection from an LED.
     * @param compared Checked right-neighbour of the reference glint expected
     * to be a primary reflection from an LED.
     * @return True, if compared is the right neighbour of reference. False otherwise.
     */
    bool isRightNeighbour(GlintCandidate &reference, GlintCandidate &compared);
    /**
     * Checks if the compared glint is the upper neighbour of the reference glint.
     * @param reference Glint expected to be a primary reflection from an LED.
     * @param compared Checked upper-neighbour of the reference glint expected
     * to be a primary reflection from an LED.
     * @return True, if compared is the upper neighbour of reference. False otherwise.
     */
    bool isUpperNeighbour(GlintCandidate &reference, GlintCandidate &compared);
    /**
     * Checks if the compared glint is the bottom neighbour of the reference glint.
     * @param reference Glint expected to be a primary reflection from an LED.
     * @param compared Checked bottom-neighbour of the reference glint expected
     * to be a primary reflection from an LED.
     * @return True, if compared is the bottom neighbour of reference. False otherwise.
     */
    bool isBottomNeighbour(GlintCandidate &reference, GlintCandidate &compared);
    /**
     * Calculates the estimated positions of the glints in the 2x3 grid that
     * have not been found.
     */
    void approximatePositions();

    /**
     * Computes the location of the pixel after undistorting it.
     * @param point Position in the region-of-interest image space.
     * @return Position in image space without distortions.
     */
    [[nodiscard]] cv::Point2f undistort(cv::Point2f point);

    // Vectors of all contours that are expected to be pupil or glints in
    // findPupil(), findGlints(), and findEllipse().
    std::vector<std::vector<cv::Point>> contours_{};

    // Number of LEDs used in findGlints().
    int led_count_{};
    // Primary reflections from LED 3x2 grid found using findEllipse().
    GlintCandidate selected_glints_[6]{};
    // Boolean informing whether LED 3x2 grid reflections have been found in
    // the image (true), or using estimation (false).
    bool found_glints_[6]{};

    /**
     * Converts cv::Mat to cv::Point2f
     * @param m Matrix to be converted.
     * @return Converted point.
     */
    static inline cv::Point2f toPoint(cv::Mat m) {
        return {(float) m.at<double>(0, 0), (float) m.at<double>(0, 1)};
    }

    /**
     * Extracts a single value from cv::Mat.
     * @param m Matrix with a value to extract.
     * @return Extracted value.
     */
    static inline float toValue(cv::Mat m) {
        return (float) m.at<double>(0, 0);
    }

    /**
     * Computes the Euclidean distance between two points in 2D space.
     * @param p First point.
     * @param q Second point.
     * @return The distance.
     */
    static inline float euclideanDistance(const cv::Point2f &p,
                                          const cv::Point2f &q) {
        cv::Point2f diff = p - q;
        return cv::sqrt(diff.x * diff.x + diff.y * diff.y);
    }

    /**
     * Checks, whether a point is in the expected area around the eye centre.
     * @param point Checked point.
     * @param centre Centre of an ellipse corresponding to the whole eye in
     * the image.
     * @return True if the point is in the expected area. False otherwise.
     */
    inline bool isInEllipse(cv::Point2f &point, cv::Point2f &centre) {
        float major = ((point.x - centre.x) * (point.x - centre.x))
            / (*bound_ellipse_semi_major_ * *bound_ellipse_semi_major_);
        float minor = ((point.y - centre.y) * (point.y - centre.y))
            / (*bound_ellipse_semi_minor_ * *bound_ellipse_semi_minor_);
        return major + minor <= 1;
    }
};
} // namespace et

#endif //HDRMFS_EYE_TRACKER_FEATURE_DETECTOR_HPP