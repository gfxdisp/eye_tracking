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

class EyeTracker {
public:
    EyeTracker();

    virtual ~EyeTracker();

    void calculateJoined(cv::Point2f pupil_pix_position,
                         std::vector<cv::Point2f> *glint_pix_positions,
                         int pupil_radius, int camera_id);
    void getCorneaCurvaturePosition(cv::Vec3d &eye_centre, int camera_id);

    void getGazeDirection(cv::Vec3f &gaze_direction, int camera_id);

    void getPupilDiameter(float &pupil_diameter, int camera_id);

    void getEyeData(EyeData &eye_data, int camera_id);

    cv::Point2f getCorneaCurvaturePixelPosition(int camera_id);

    cv::Point2f getEyeCentrePixelPosition(int camera_id);

    void initialize();

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

    float pupil_diameter_[2]{};

    cv::KalmanFilter kalman_eye_[2]{};
    cv::KalmanFilter kalman_gaze_[2]{};
    EyePosition eye_position_[2]{};
    cv::Vec3f inv_optical_axis_{};
    std::mutex mtx_eye_position_{};
    std::mutex mtx_setup_to_change_{};

    RayPointMinimizer *ray_point_minimizer_{};
    cv::Ptr<cv::DownhillSolver::Function> minimizer_function_{};
    cv::Ptr<cv::DownhillSolver> solver_{};

    cv::Point2f pupil_pix_position_{};
    std::vector<cv::Point2f> *glint_pix_positions_{};

    cv::Mat full_projection_matrices_[2]{};
    static cv::Mat visual_axis_rotation_matrix_;

    [[nodiscard]] inline cv::Vec3f project(const cv::Vec2f &point, int camera_id) const {
        return project(ICStoWCS(point, camera_id));
    }

    [[nodiscard]] inline cv::Vec3f ICStoWCS(const cv::Vec2f &point, int camera_id) const {
        return CCStoWCS(ICStoCCS(point, camera_id));
    }

    [[nodiscard]] inline cv::Vec2f WCStoICS(const cv::Vec3f &point, int camera_id) const {
        return CCStoICS(WCStoCCS(point), camera_id);
    }

    static inline cv::Point3f toPoint(cv::Mat m) {
        return {(float) m.at<double>(0, 0), (float) m.at<double>(0, 1), (float) m.at<double>(0, 2)};
    }

    [[nodiscard]] cv::Point2f undistort(cv::Point2f point, int camera_id);

    [[nodiscard]] cv::Vec3f project(const cv::Vec3f &point) const;

    [[nodiscard]] cv::Vec2f unproject(const cv::Vec3f &point, int camera_id) const;

    [[nodiscard]] cv::Vec3f ICStoCCS(const cv::Point2f &point, int camera_id) const;

    [[nodiscard]] cv::Vec3f CCStoWCS(const cv::Vec3f &point) const;

    [[nodiscard]] cv::Vec3f WCStoCCS(const cv::Vec3f &point) const;

    [[nodiscard]] cv::Vec2f CCStoICS(cv::Vec3f point, int camera_id) const;

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

}// namespace et

#endif //EYE_TRACKER_H