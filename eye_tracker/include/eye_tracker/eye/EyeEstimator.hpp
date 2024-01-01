#ifndef HDRMFS_EYE_TRACKER_EYEESTIMATOR_HPP
#define HDRMFS_EYE_TRACKER_EYEESTIMATOR_HPP

#include "eye_tracker/Settings.hpp"

#include <opencv2/core/types.hpp>
#include <vector>
#include <opencv2/core/mat.hpp>

namespace et
{

    struct EyeInfo
    {
        cv::Point2d pupil;
        double pupil_radius;
        std::vector<cv::Point2d> glints;
        cv::RotatedRect ellipse;
    };

    class EyeEstimator
    {
    public:
        EyeEstimator(int camera_id);

        virtual bool detectEye(EyeInfo &eye_info, cv::Point3d &nodal_point, cv::Point3d &eye_centre, cv::Point3d &visual_axis) = 0;

        virtual bool
        findPupilDiameter(cv::Point2d pupil_pix_position, int pupil_px_radius, const cv::Vec3d &cornea_centre_position,
                          double &diameter);

        virtual void getGazeDirection(cv::Point3d nodal_point, cv::Point3d eye_centre, cv::Vec3d &gaze_direction);


        bool findEye(EyeInfo &eye_info);

        /**
         * Retrieves eye centre position in camera space previously calculated using
         * getEyeFromModel() or getEyeFromPolynomial().
         * @param eye_centre Variable that will contain the eye centre position.
         */
        void getEyeCentrePosition(cv::Point3d &eye_centre);

        /**
         * Retrieves cornea centre position in camera space previously calculated
         * using getEyeFromModel() or getEyeFromPolynomial().
         * @param cornea_centre Variable that will contain the cornea centre position.
         */
        void getCorneaCurvaturePosition(cv::Point3d &cornea_centre);

        /**
         * Calculates the gaze direction in camera space based on parameters that
         * were previously calculated using getEyeFromModel() or
         * getEyeFromPolynomial().
         * @param gaze_direction Variable that will contain the normalized
         * gaze direction.
         */
        void getGazeDirection(cv::Vec3d &gaze_direction);

        /**
         * Retrieves pupil diameter in millimeters that was previously calculated
         * using getEyeFromModel() or getEyeFromPolynomial().
         * @param pupil_diameter Variable that will contain the pupil diameter.
         */
        void getPupilDiameter(double &pupil_diameter);

        /**
         * Retrieves cornea centre position projected to image space based on
         * calculations from getEyeFromModel() or getEyeFromPolynomial().
         * @return Cornea centre position.
         */
        cv::Point2d getCorneaCurvaturePixelPosition();

        /**
         * Retrieves eye centre position projected to image space based on
         * calculations from getEyeFromModel() or getEyeFromPolynomial().
         * @return Eye centre position.
         */
        cv::Point2d getEyeCentrePixelPosition();

    protected:

        /**
         * Creates inverted projection matrix from image to camera space.
         */
        void createInvertedProjectionMatrix();

        /**
         * Converts point from image space to camera space.
         * @param point Location of the pixel in image space.
         * @return Location of the pixel in camera space.
         */
        [[nodiscard]] cv::Vec3d ICStoCCS(const cv::Point2d point);

        [[nodiscard]] cv::Vec3d CCStoWCS(const cv::Vec3d point);

        [[nodiscard]] cv::Vec3d ICStoWCS(const cv::Point2d point);

        [[nodiscard]] cv::Point2d CCStoICS(cv::Point3d point);

        [[nodiscard]] cv::Point2d WCStoICS(cv::Point3d point);

        [[nodiscard]] cv::Point3d WCStoCCS(cv::Point3d point);

        /**
         * Estimates the camera space position of the point located on the pupil
         * @param pupil_px_position Pixel position of the point located on the pupil.
         * @param cornea_centre Cornea centre position in camera space.
         * @return Camera space position of the pupil point.
         */
        [[nodiscard]] cv::Vec3d
        calculatePositionOnPupil(const cv::Vec3d &pupil_px_position, const cv::Vec3d &cornea_centre);

        // Distance from top-left corner of the region-of-interest to the top-left
        // corner of the full image, measured in pixels separately for every axis.
        cv::Size2i *capture_offset_{};
        // Intrinsic matrix of the camera.
        cv::Mat *intrinsic_matrix_{};
        // Inverted projection matrix from image to camera space.
        cv::Mat inv_projection_matrix_{};
        // Inverted extrinsic matrix of the camera.
        cv::Mat inv_extrinsic_matrix_{};

        cv::Mat extrinsic_matrix_{};

        cv::Size2i *dimensions_{};

        SetupVariables *setup_variables_{};

        int camera_id_{};

        double pupil_cornea_distance_{};

        double eye_cornea_distance_{};

        double cornea_radius_{};

        double refraction_index_{};

        // Diameter of the pupil in millimeters.
        double pupil_diameter_{};

        // Synchronization variable between an eye-tracking algorithm and socket server.
        std::mutex mtx_eye_position_{};

        cv::Point3d cornea_centre_{};

        cv::Point2d cornea_centre_pixel_{};

        cv::Point3d eye_centre_{};

        cv::Point2d eye_centre_pixel_{};

        cv::Vec3d gaze_direction_{};

        cv::Vec3d camera_nodal_point_{};
    };

} // et

#endif //HDRMFS_EYE_TRACKER_EYEESTIMATOR_HPP