#ifndef EYE_TRACKER_H
#define EYE_TRACKER_H

#include "FeatureDetector.hpp"

#include "ImageProvider.hpp"
#include "RayPointMinimizer.hpp"

#include <opencv2/opencv.hpp>

#include <mutex>
#include <optional>
#include <vector>

namespace et {
struct EyePosition {
    std::optional<cv::Vec3f> cornea_curvature{};
    std::optional<cv::Vec3f> pupil{};
    std::optional<cv::Vec3f> eye_centre{};

    inline explicit operator bool() const {
        return eye_centre and pupil and cornea_curvature;
    }
};

struct EyeData {
    cv::Vec3f cornea_curvature{};
    cv::Vec3f pupil{};
    cv::Vec3f eye_centre{};
};

class EyeEstimator {
public:
    EyeEstimator();

    virtual ~EyeEstimator();

    void initialize(int camera_id);

    void getEyeFromModel(cv::Point2f pupil_pix_position,
                         std::vector<cv::Point2f> *glint_pix_positions,
                         int pupil_radius);

    void getEyeFromPolynomial(cv::Point2f pupil_pix_position,
                             cv::RotatedRect ellipse);

    void getEyeCentrePosition(cv::Vec3d &eye_centre);

    void getCorneaCurvaturePosition(cv::Vec3d &cornea_centre);

    void getGazeDirection(cv::Vec3f &gaze_direction);

    void getPupilDiameter(float &pupil_diameter);

    void getEyeData(EyeData &eye_data);

    cv::Point2f getCorneaCurvaturePixelPosition();

    cv::Point2f getEyeCentrePixelPosition();

    static bool getRaySphereIntersection(const cv::Vec3f &ray_pos,
                                         const cv::Vec3d &ray_dir,
                                         const cv::Vec3f &sphere_pos,
                                         double sphere_radius, double &t);

    static cv::Vec3d getRefractedRay(const cv::Vec3d &direction,
                                     const cv::Vec3d &normal,
                                     double refraction_index);

    bool isSetupUpdated();
    static void createVisualAxis();

private:
    bool setup_updated_{false};

    float pupil_diameter_{};

    cv::KalmanFilter kalman_eye_{};
    cv::KalmanFilter kalman_gaze_{};
    EyePosition eye_position_{};
    cv::Vec3f inv_optical_axis_{};
    std::mutex mtx_eye_position_{};

    PolynomialFit eye_centre_pos_x_fit_{5, 3};
    PolynomialFit eye_centre_pos_y_fit_{5, 3};
    PolynomialFit eye_centre_pos_z_fit_{5, 3};
    PolynomialFit eye_np_pos_x_fit_{5, 3};
    PolynomialFit eye_np_pos_y_fit_{5, 3};
    PolynomialFit eye_np_pos_z_fit_{5, 3};

    RayPointMinimizer *ray_point_minimizer_{};
    cv::Ptr<cv::DownhillSolver::Function> minimizer_function_{};
    cv::Ptr<cv::DownhillSolver> solver_{};

    cv::Point2f pupil_pix_position_{};
    std::vector<cv::Point2f> *glint_pix_positions_{};

    cv::Mat full_projection_matrices_{};
    static cv::Mat visual_axis_rotation_matrix_;

    std::vector<cv::Vec3f> *leds_positions_{};
    cv::Vec3f *gaze_shift_{};
    cv::Mat *intrinsic_matrix_{};
    cv::Size2i *capture_offset_{};
    cv::Size2i *dimensions_{};
    std::vector<float> *distortion_coefficients_{};

    [[nodiscard]] inline cv::Vec3f project(const cv::Vec2f &point) const {
        return project(ICStoWCS(point));
    }

    [[nodiscard]] inline cv::Vec3f ICStoWCS(const cv::Vec2f &point) const {
        return CCStoWCS(ICStoCCS(point));
    }

    [[nodiscard]] inline cv::Vec2f WCStoICS(const cv::Vec3f &point) const {
        return CCStoICS(WCStoCCS(point));
    }

    static inline cv::Point3f toPoint(cv::Mat m) {
        return {(float) m.at<double>(0, 0), (float) m.at<double>(0, 1),
                (float) m.at<double>(0, 2)};
    }

    [[nodiscard]] cv::Point2f undistort(cv::Point2f point);

    [[nodiscard]] cv::Vec3f project(const cv::Vec3f &point) const;

    [[nodiscard]] cv::Vec2f unproject(const cv::Vec3f &point) const;

    [[nodiscard]] cv::Vec3f ICStoCCS(const cv::Point2f &point) const;

    [[nodiscard]] cv::Vec3f CCStoWCS(const cv::Vec3f &point) const;

    [[nodiscard]] cv::Vec3f WCStoCCS(const cv::Vec3f &point) const;

    [[nodiscard]] cv::Vec2f CCStoICS(cv::Vec3f point) const;

    [[nodiscard]] cv::Vec3f
    ICStoEyePosition(const cv::Vec3f &point,
                     const cv::Vec3f &cornea_centre) const;

    static std::vector<cv::Vec3d>
    lineSphereIntersections(const cv::Vec3d &sphere_centre, float radius,
                            const cv::Vec3d &line_point,
                            const cv::Vec3d &line_direction);

    [[nodiscard]] static cv::KalmanFilter makeKalmanFilter(float framerate);

    void createProjectionMatrix();

    static cv::Mat euler2rot(float *euler_angles);
};

} // namespace et

#endif //EYE_TRACKER_H