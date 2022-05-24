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

class EyeTracker {
public:
    EyeTracker(ImageProvider *image_provider);

    virtual ~EyeTracker();

    void calculateJoined(cv::Point2f pupil_pix_position,
                         std::vector<cv::Point2f> &glint_pix_positions,
                         float pupil_radius);
    void getCorneaCurvaturePosition(cv::Vec3d &eye_centre);

    void getGazeDirection(cv::Vec3d &gaze_direction);

    void getPupilDiameter(float &pupil_diameter);

    cv::Point2f getCorneaCurvaturePixelPosition();

    void initializeKalmanFilter(float framerate);

    static bool getRaySphereIntersection(const cv::Vec3f &ray_pos,
                                         const cv::Vec3d &ray_dir,
                                         const cv::Vec3f &sphere_pos,
                                         double sphere_radius, double &t);

    static cv::Vec3d getRefractedRay(const cv::Vec3d &direction,
                                     const cv::Vec3d &normal,
                                     double refraction_index);

    bool isSetupUpdated();

private:
    bool setup_updated_{false};

    float pupil_diameter_{};

    ImageProvider *image_provider_{};
    cv::KalmanFilter kalman_{};
    EyePosition eye_position_{};
    std::mutex mtx_eye_position_{};
    std::mutex mtx_setup_to_change_{};

    RayPointMinimizer *ray_point_minimizer_{};
    cv::Ptr<cv::DownhillSolver::Function> minimizer_function_{};
    cv::Ptr<cv::DownhillSolver> solver_{};

    cv::Mat full_projection_matrix_{};

    [[nodiscard]] inline cv::Vec3f project(const cv::Vec2f &point) const {
        return project(ICStoWCS(point));
    }

    [[nodiscard]] inline cv::Vec3f ICStoWCS(const cv::Vec2f &point) const {
        return CCStoWCS(ICStoCCS(point));
    }

    [[nodiscard]] inline cv::Vec2f WCStoICS(const cv::Vec3f &point) const {
        return CCStoICS(WCStoCCS(point));
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

    cv::Mat euler2rot(double *euler_angles);
};

}// namespace et

#endif