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
    std::optional<cv::Vec3d> cornea_curvature{};
    std::optional<cv::Vec3d> pupil{};
    std::optional<cv::Vec3d> eye_centre{};
    inline explicit operator bool() const {
        return eye_centre and pupil and cornea_curvature;
    }
};

struct EyeProperties {
    static constexpr float cornea_curvature_radius{7.8f};
    static constexpr float pupil_cornea_distance{4.2f};
    static constexpr float refraction_index{1.3375f};
    static constexpr float eye_ball_radius{11.6f};
    static constexpr float pupil_eye_centre_distance{9.5f};
};

struct SetupLayout {
    double camera_lambda{};
    cv::Vec3d camera_nodal_point_position{};
    cv::Vec3d led_positions[FeatureDetector::LED_COUNT]{};
    double camera_eye_distance{};
    double camera_eye_projection_factor{};
    cv::Matx33d rotation{cv::Matx33d::eye()};
    cv::Vec3d translation{};
    double alpha{};
    double beta{};
    cv::Mat visual_axis_rotation{};
};

class EyeTracker {
public:
    EyeTracker(SetupLayout &setup_layout, ImageProvider *image_provider);

    virtual ~EyeTracker();

    void calculateJoined(const cv::Point2f &pupil_pixel_position, cv::Point2f *glints_pixel_positions, float
                                                                                                           pupil_radius);
    void getCorneaCurvaturePosition(cv::Vec3d &eye_centre);

    void getGazeDirection(cv::Vec3d &gaze_direction);

    void getPupilDiameter(float &pupil_diameter);

    cv::Point2d getCorneaCurvaturePixelPosition();

    void setNewSetupLayout(SetupLayout &setup_layout);

    void initializeKalmanFilter(float framerate);

    static bool getRaySphereIntersection(const cv::Vec3d &ray_pos, const cv::Vec3d &ray_dir,
                                         const cv::Vec3d &sphere_pos, double sphere_radius, double &t);

    static cv::Vec3d getRefractedRay(const cv::Vec3d &direction, const cv::Vec3d &normal, double refraction_index);

    bool isSetupUpdated();

private:
    SetupLayout setup_layout_{};
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

    [[nodiscard]] inline cv::Vec3d project(const cv::Point2f &point) const {
        return project(ICStoWCS(point));
    }

    [[nodiscard]] inline cv::Vec3d ICStoWCS(const cv::Point2d &point) const {
        return CCStoWCS(ICStoCCS(point));
    }

    [[nodiscard]] inline cv::Point2d WCStoICS(const cv::Vec3d &point) const {
        return CCStoICS(WCStoCCS(point));
    }

    [[nodiscard]] cv::Vec3d project(const cv::Vec3d &point) const;

    [[nodiscard]] cv::Point2d unproject(const cv::Vec3d &point) const;

    [[nodiscard]] cv::Vec3d ICStoCCS(const cv::Point2d &point) const;

    [[nodiscard]] cv::Vec3d CCStoWCS(const cv::Vec3d &point) const;

    [[nodiscard]] cv::Vec3d WCStoCCS(const cv::Vec3d &point) const;

    [[nodiscard]] cv::Point2d CCStoICS(cv::Vec3d point) const;

    static std::vector<cv::Vec3d> lineSphereIntersections(const cv::Vec3d &sphere_centre, float radius,
                                                          const cv::Vec3d &line_point, const cv::Vec3d &line_direction);

    [[nodiscard]] static cv::KalmanFilter makeKalmanFilter(float framerate);

    cv::Mat euler2rot(double *euler_angles);
};

}// namespace et

#endif