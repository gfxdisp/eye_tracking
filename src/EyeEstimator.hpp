#ifndef HDRMFS_EYE_TRACKER_EYE_ESTIMATOR_HPP
#define HDRMFS_EYE_TRACKER_EYE_ESTIMATOR_HPP

#include "FeatureDetector.hpp"

#include "ImageProvider.hpp"
#include "RayPointMinimizer.hpp"

#include <opencv2/opencv.hpp>

#include <mutex>
#include <optional>
#include <vector>

namespace et {
/**
 * Calculates the position of the eye's centre its other elements.
 */
class EyeEstimator {
public:
    /**
     * Releases the resources used by RayPointMinimizer.
     */
    virtual ~EyeEstimator();

    /**
     * Initializes RayPointMinimizer, loads polynomial coefficients for ellipse
     * fitting, calculates the rotation between visual and optical axis, and
     * calculates projection matrix between image and camera space.
     * @param settings_path Path to a folder containing all settings files.
     * @param kalman_filtering_enabled True if kalman filtering is enabled,
     * false otherwise.
     * @param camera_id An id of the camera to which the object corresponds.
     */
    void initialize(const std::string &settings_path,
                    bool kalman_filtering_enabled, int camera_id);

    /**
     * Calculates eye and cornea centre positions in camera space based on the
     * 2006 Guestrin et al. model.
     * @param pupil_pix_position Pupil position in image space.
     * @param glint_pix_positions Glint positions in image space. The number of
     * glints and their order must correspond to
     * LEDs in Settings::parameters.leds_positions.
     */
    void getEyeFromModel(cv::Point2f pupil_pix_position,
                         std::vector<cv::Point2f> *glint_pix_positions);

    /**
     * Calculates eye and cornea centre positions in camera space based on the
     * polynomial parameters.
     * @param pupil_pix_position Pupil position in image space.
     * @param ellipse Ellipse fit to glints in image space.
     */
    void getEyeFromPolynomial(cv::Point2f pupil_pix_position,
                              cv::RotatedRect ellipse);

    /**
     * Calculates the diameter of the pupil in millimeters.
     * @param pupil_pix_position Centre of the pupil in image space.
     * @param pupil_px_radius Radius of the pupil in image space.
     * @param cornea_centre_position Cornea centre position in camera space.
     */
    void calculatePupilDiameter(cv::Point2f pupil_pix_position,
                                int pupil_px_radius,
                                const cv::Vec3f &cornea_centre_position);

    /**
     * Retrieves eye centre position in camera space previously calculated using
     * getEyeFromModel() or getEyeFromPolynomial().
     * @param eye_centre Variable that will contain the eye centre position.
     */
    void getEyeCentrePosition(cv::Vec3d &eye_centre);

    /**
     * Retrieves cornea centre position in camera space previously calculated
     * using getEyeFromModel() or getEyeFromPolynomial().
     * @param cornea_centre Variable that will contain the cornea centre position.
     */
    void getCorneaCurvaturePosition(cv::Vec3d &cornea_centre);

    /**
     * Calculates the gaze direction in camera space based on parameters that
     * were previously calculated using getEyeFromModel() or
     * getEyeFromPolynomial().
     * @param gaze_direction Variable that will contain the normalized
     * gaze direction.
     */
    void getGazeDirection(cv::Vec3f &gaze_direction);

    /**
     * Retrieves pupil diameter in millimeters that was previously calculated
     * using getEyeFromModel() or getEyeFromPolynomial().
     * @param pupil_diameter Variable that will contain the pupil diameter.
     */
    void getPupilDiameter(float &pupil_diameter);

    /**
     * Retrieves cornea centre position projected to image space based on
     * calculations from getEyeFromModel() or getEyeFromPolynomial().
     * @return Cornea centre position.
     */
    cv::Point2f getCorneaCurvaturePixelPosition();

    /**
     * Retrieves eye centre position projected to image space based on
     * calculations from getEyeFromModel() or getEyeFromPolynomial().
     * @return Eye centre position.
     */
    cv::Point2f getEyeCentrePixelPosition();

    /**
     * Calculates the intersection point between a ray and a sphere.
     * @param ray_pos Position of the ray origin.
     * @param ray_dir Direction of the ray.
     * @param sphere_pos Centre position of the sphere.
     * @param sphere_radius Radius of the sphere.
     * @param t The distance between ray origin and an intersection point.
     * @return True if the intersection is found. False otherwise.
     */
    static bool getRaySphereIntersection(const cv::Vec3f &ray_pos,
                                         const cv::Vec3d &ray_dir,
                                         const cv::Vec3f &sphere_pos,
                                         double sphere_radius, double &t);

    /**
     * Calculates a vector's direction after refraction.
     * @param direction Direction vector.
     * @param normal Normal of the refracting surface.
     * @param refraction_index Refraction index of the surface.
     * @return Direction vector after refraction.
     */
    static cv::Vec3d getRefractedRay(const cv::Vec3d &direction,
                                     const cv::Vec3d &normal,
                                     double refraction_index);

    /**
     * Creates a matrix used to rotate an optical axis to correspond to visual
     * axis based on applied eye parameters.
     */
    static void createVisualAxis();

private:
    // Diameter of the pupil in millimeters.
    float pupil_diameter_{};

    // Kalman filter used to correct noisy cornea centre position.
    cv::KalmanFilter kalman_eye_{};

    // Inverted calculated optical axis of the eye.
    cv::Vec3f inv_optical_axis_{};
    // Synchronization variable between an eye-tracking algorithm and socket server.
    std::mutex mtx_eye_position_{};

    // Polynomial coefficients estimating x-axis position of the eye centre
    // in the getEyeFromPolynomial() method.
    PolynomialFit eye_centre_pos_x_fit_{5, 3};
    // Polynomial coefficients estimating y-axis position of the eye centre
    // in the getEyeFromPolynomial() method.
    PolynomialFit eye_centre_pos_y_fit_{5, 3};
    // Polynomial coefficients estimating z-axis position of the eye centre
    // in the getEyeFromPolynomial() method.
    PolynomialFit eye_centre_pos_z_fit_{5, 3};
    // Polynomial coefficients estimating x-axis position of the cornea centre
    // in the getEyeFromPolynomial() method.
    PolynomialFit cornea_centre_pos_x_fit_{5, 3};
    // Polynomial coefficients estimating y-axis position of the cornea centre
    // in the getEyeFromPolynomial() method.
    PolynomialFit cornea_centre_pos_y_fit_{5, 3};
    // Polynomial coefficients estimating z-axis position of the cornea centre
    // in the getEyeFromPolynomial() method.
    PolynomialFit cornea_centre_pos_z_fit_{5, 3};

    // Function used to optimize cornea centre position in
    // the getEyeFromModel() method.
    RayPointMinimizer *ray_point_minimizer_{};
    // Pointer to a RayPointMinimizer function.
    cv::Ptr<cv::DownhillSolver::Function> minimizer_function_{};
    // Downhill solver optimizer used to find cornea centre position.
    cv::Ptr<cv::DownhillSolver> solver_{};

    // Inverted projection matrix from image to camera space.
    cv::Mat inv_projection_matrix_{};
    // Rotation matrix between optical and visual axis of the eye.
    static cv::Mat visual_axis_rotation_matrix_;

    // LEDs positions in camera space.
    std::vector<cv::Vec3f> *leds_positions_{};
    // Constant shift needed to be added to the vector between eye and cornea centre.
    cv::Vec3f *gaze_shift_{};
    // Intrinsic matrix of the camera.
    cv::Mat *intrinsic_matrix_{};
    // Distance from top-left corner of the region-of-interest to the top-left
    // corner of the full image, measured in pixels separately for every axis.
    cv::Size2i *capture_offset_{};
    // All parameters of the camera.
    CameraParams *camera_params_{};
    // Distortion coefficients of the camera.
    std::vector<float> *distortion_coefficients_{};

    // Cornea centre position in camera space calculated using getEyeFromModel()
    // or getEyeFromPolynomial().
    cv::Vec3f cornea_centre_{};
    // Eye centre position in camera space calculated using getEyeFromModel()
    // or getEyeFromPolynomial().
    cv::Vec3f eye_centre_{};

    // True if kalman filtering is enabled, false otherwise.
    bool kalman_filtering_enabled_{};

    /**
     * Converts cv::Mat to cv::Point3f
     * @param m Matrix to be converted.
     * @return Converted point.
     */
    static inline cv::Point3f toPoint(cv::Mat m) {
        return {(float) m.at<double>(0, 0), (float) m.at<double>(0, 1),
                (float) m.at<double>(0, 2)};
    }

    /**
     * Computes the location of the pixel after undistorting it.
     * @param point Position in the region-of-interest image space.
     * @return Position in image space without distortions.
     */
    [[nodiscard]] cv::Point2f undistort(cv::Point2f point);

    /**
     * Converts point position from camera space to region-of-interest image space.
     * @param point Location in the camera space.
     * @return Pixel location of the point in region-of-interest image space.
     */
    [[nodiscard]] cv::Vec2f unproject(const cv::Vec3f &point) const;

    /**
     * Converts point from image space to camera space.
     * @param point Location of the pixel in image space.
     * @return Location of the pixel in camera space.
     */
    [[nodiscard]] cv::Vec3f ICStoCCS(const cv::Point2f &point) const;

    /**
     * Estimates the camera space position of the point located on the pupil
     * @param pupil_px_position Pixel position of the point located on the pupil.
     * @param cornea_centre Cornea centre position in camera space.
     * @return Camera space position of the pupil point.
     */
    [[nodiscard]] static cv::Vec3f
    calculatePositionOnPupil(const cv::Vec3f &pupil_px_position,
                             const cv::Vec3f &cornea_centre);

    /**
     * Creates a 6x6 Kalman Filter assuming its input vector consists of
     * XYZ position and XYZ velocity.
     * @param framerate Estimated system framerate used to calculate velocity.
     * @return Kalman filter.
     */
    [[nodiscard]] static cv::KalmanFilter makeKalmanFilter(float framerate);

    /**
     * Creates inverted projection matrix from image to camera space.
     */
    void createInvProjectionMatrix();

    /**
     * Converts set of angles to a rotation matrix.
     * @param euler_angles 3 angles in radians around X, Y, and Z-axis.
     * @return Rotation matrix.
     */
    static cv::Mat euler2rot(const float *euler_angles);

    /**
     * Loads coefficients of the polynomial used in the getEyeFromPolynomial()
     * method. If those coefficients cannot be found, there are calculated from
     * the input files.
     * @param coefficients_filename CSV filename with polynomial coefficients.
     * @param eye_data_filename CSV file with ground truth eye and cornea centre
     * positions of the Blender rendered images.
     * @param features_data_filename CSV file pupil and glints position of the
     * Blender rendered images.
     */
    void loadPolynomialCoefficients(const std::string &coefficients_filename,
                                    const std::string &eye_data_filename,
                                    const std::string &features_data_filename);
};

} // namespace et

#endif //HDRMFS_EYE_TRACKER_EYE_ESTIMATOR_HPP