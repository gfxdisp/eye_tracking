#ifndef HDRMFS_EYE_TRACKER_EYE_ESTIMATOR_HPP
#define HDRMFS_EYE_TRACKER_EYE_ESTIMATOR_HPP

#include <eye_tracker/optimizers/NodalPointOptimizer.hpp>
#include <eye_tracker/Settings.hpp>
#include <eye_tracker/optimizers/Polynomial.hpp>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <vector>

namespace et {
    struct EyeInfo {
        cv::Point2d pupil_centre_position_ics;
        double pupil_radius_ics;
        std::vector<cv::Point2d> glint_positions_ics;
        std::vector<bool> glints_valid;
    };

    class EyeEstimator {
    public:
        explicit EyeEstimator(int camera_id);

        bool detectEye(EyeInfo& eye_info, cv::Point3d& eye_centre_position_wcs, cv::Point3d& cornea_centre_position_wcs, cv::Vec2d& gaze_direction_angles_rad) const;

        bool findPupilDiameter(cv::Point2d pupil_centre_position_ics, double pupil_radius_ics, const cv::Vec3d& cornea_centre_position_wcs, double& pupil_diameter_mm) const;

        bool estimateEyePositions(EyeInfo& eye_info, bool add_correction);

        void getEyeCentrePositionWCS(cv::Point3d& eye_centre_position_wcs);

        void getCorneaCentrePositionWCS(cv::Point3d& cornea_centre);

        void getGazeDirection(cv::Vec3d& gaze_direction);

        void getPupilDiameter(double& pupil_diameter_mm);

        void getCorneaCentrePositionICS(cv::Point2d& cornea_centre_position_ics, bool use_offset = true);

        cv::Point2d getNormalizedGazePoint() const;

        void updateFineTuning();

        EyeParams eye_params_{};

        static constexpr double eye_camera_distance_min_mm = 250.0;
        static constexpr double eye_camera_distance_max_mm = 450.0;

        static constexpr double bottom_left_gaze_window_x_mm_ = 130;
        static constexpr double bottom_left_gaze_window_y_mm_ = 50;
        static constexpr double upper_right_gaze_window_x_mm_ = 330;
        static constexpr double upper_right_gaze_window_y_mm_ = 250;

    protected:
        cv::Vec3d ICStoCCS(cv::Point2d point) const;

        cv::Vec3d CCStoWCS(const cv::Vec3d& point) const;

        cv::Vec3d ICStoWCS(cv::Point2d point) const;

        cv::Point2d CCStoICS(const cv::Point3d& point) const;

        cv::Point2d WCStoICS(const cv::Point3d& point) const;

        cv::Point3d WCStoCCS(const cv::Point3d& point) const;

        cv::Vec3d calculatePupilPositionCCS(const cv::Vec3d& pupil_centre_position_at_sensor_ccs, const cv::Vec3d& cornea_centre_position_ccs) const;

        cv::Size2i* capture_offset_{};

        cv::Mat* intrinsic_matrix_{};

        cv::Mat inv_extrinsic_matrix_{};

        cv::Mat extrinsic_matrix_{};

        cv::Size2i* dimensions_{};

        FeaturesParams* features_params_{};

        int camera_id_{};

        double pupil_diameter_mm_{};

        std::mutex mtx_eye_position_{};

        cv::Point3d cornea_centre_position_wcs_{};

        cv::Point2d cornea_centre_position_ics_{};

        cv::Point3d eye_centre_position_wcs_{};

        cv::Point2d eye_centre_position_ics_{};

        cv::Point2d gaze_point_{};

        cv::Point2d normalized_gaze_point_{};

        cv::Vec3d model_gaze_direction_{};

        cv::Vec3d camera_nodal_position_ccs_{};

        std::shared_ptr<Polynomial> theta_polynomial_{};
        std::shared_ptr<Polynomial> phi_polynomial_{};
        cv::Point3d eye_position_offset_{};
        double marker_depth_{};

        constexpr static int GAZE_BUFFER = 10;
        cv::Point3d gaze_point_buffer_[GAZE_BUFFER]{};
        int gaze_point_index_{0};
        bool gaze_point_history_full_{false};
        cv::Point3d gaze_point_sum_{};

        std::shared_ptr<NodalPointOptimizer> cornea_centre_position_ccs_optimizer_{};
    };
} // namespace et

#endif // HDRMFS_EYE_TRACKER_EYE_ESTIMATOR_HPP
