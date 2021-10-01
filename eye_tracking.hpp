#pragma once
#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp> // cv::KalmanFilter
#include <vector>
namespace EyeTracking {
    /* Three coordinate systems are used in this file (as by Guestrin & Eizenman).
     * The ICS (image coordinate system) is a 2D Cartesian system for positions of pixels in the image.
     * The CCS (camera coordinate system) is a 3D Cartesian system with its origin at nodalPoint, its z axis
     * directed out of the camera, y directed up and x directed left.
     * The WCS (world coordinate system) is a 3D Cartesian system; its relationship to the CCS is determined by
     * positions.nodalPoint and positions.rotation.
     *
     * Unless indicated otherwise, 2D positions and distances are in pixels and in the ICS,
     * while 3D positions and distances are in millimetres and in the WCS. */

    using cv::Matx33d;
    using cv::Point2d;
    using cv::Point2f;
    using cv::Vec3d;
    using KFMat = cv::Mat_<double>;

    /* Given the centre of a sphere, its radius, a position vector for a point on a line, and a direction vector
     * for that line, find the intersections of the line and the sphere (there may be 0, 1 or 2 of them). */
    std::vector<Vec3d> lineSphereIntersections(Vec3d sphereCentre, float radius, Vec3d linePoint, Vec3d lineDirection);

    struct EyePosition {
        /* Type representing the position and orientation of an eye (in the WCS, in mm).
         * We use std::optional as some of these values can be missing. */
        std::optional<Vec3d> corneaCurvatureCentre, pupilCentre, eyeCentre; // c, p, d
        inline operator bool() const {
            return eyeCentre and pupilCentre and corneaCurvatureCentre;
        }
    };

    const static Point2f None = {-1, -1}; // Placeholder value

    // Conversions (for use with cv::KalmanFilter)
    inline Point2d toPoint(cv::Mat m) {
        return {m.at<double>(0, 0), m.at<double>(0, 1)};
    }

    inline cv::Mat toMat(Point2d p) {
        return (cv::Mat_<double>(2, 1) << p.x, p.y);
    }

    struct CircleConstraints {
        // Type representing the parameters of a (dark) circle to be detected in a monochrome image
        uint8_t threshold; // Brightness threshold
        float minRadius, maxRadius; // in pixels
        /* For each contour detected in the image, the minimum enclosing circle is calculated.
         * If the contour fills minRating of the circle's area, it is considered circular. */
        float minRating = 0.78;
    };

    struct RatedCircleCentre {
        // Type representing a detected circle centre with a rating
        Point2f point = {-1, -1};
        float rating = std::numeric_limits<float>::infinity();
        inline bool operator<(const RatedCircleCentre& other) const {
            return rating < other.rating;
        }
        inline bool operator>(const RatedCircleCentre& other) const {
            return rating > other.rating;
        }
    };

    // Detect dark circles in an image
    std::vector<RatedCircleCentre> findCircles(const cv::cuda::GpuMat& frame, CircleConstraints constraints);

    struct CameraProperties {
        double FPS; // Hz
        inline double dt() const { return 1/FPS; } // s
        cv::Size2i resolution; // px
        double pixelPitch; // mm
        double exposureTime; // ms (probably)
        double gain; // range and units depend on camera
    };

    struct EyeProperties {
        /* From Guestrin & Eizenman
         * They provide a calibration procedure, but in fact there is not much variation in these parameters. */
        float R = 7.8; // mm, radius of corneal curvature
        float K = 4.2; // mm, distance between pupil centre and centre of corneal curvature
        float n1 = 1.3375; // Standard Keratometric Index, refractive index of cornea and aqueous humour
        // From Bekerman, Gottlieb & Vaiman
        float D = 10; // mm, distance between pupil centre and centre of eye rotation
    };

    struct ImageProperties {
        cv::Rect ROI;
        CircleConstraints pupil, iris;
        double maxPupilIrisSeparation; // px
        double templateMatchingThreshold = 0.9;
    };

    struct Positions {
        // Represents the layout of the system: locations of the camera, light and eye, relationship between CCS and WCS.
        double lambda; // mm; distance from the nodal point to the image plane; changes as focus is moved
        Vec3d nodalPoint; // o; mm; nodal point of the camera, in the WCS
        Vec3d light; // l; mm; position of the light, in the WCS
        Matx33d rotation; // dimensionless; rotation matrix from CCS to WCS
        double cameraEyeDistance; // mm; Z-axis distance from camera to Purkyně reflection
        double cameraEyeProjectionFactor;
        inline Positions(double lambda,
                         Vec3d nodalPoint,
                         Vec3d light,
                         Matx33d rotation = Matx33d::eye()):
            lambda(lambda),
            nodalPoint(nodalPoint),
            light(light),
            rotation(rotation),
            cameraEyeDistance(-nodalPoint(2)),
            cameraEyeProjectionFactor(cameraEyeDistance/lambda) {};
    };

    class Tracker {
        // Class encapsulating the state of the eye tracker
        protected:
            cv::KalmanFilter KF;
            const EyeProperties eye;
            const CameraProperties camera;
            const Positions positions;
        public:
            // Create Kálmán filters with default settings, for use in the ICS or WCS
            cv::KalmanFilter makeICSKalmanFilter() const;
            cv::KalmanFilter makeWCSKalmanFilter() const;

            // Conversions between coordinate systems (see top of this file)
            Vec3d ICStoCCS(Point2d point) const;
            Vec3d CCStoWCS(Vec3d point) const;
            Vec3d WCStoCCS(Vec3d point) const;
            Point2d CCStoICS(Vec3d point) const;
            inline Vec3d ICStoWCS(Point2d point) const { return CCStoWCS(ICStoCCS(point)); }
            inline Point2d WCStoICS(Vec3d point) const { return CCStoICS(WCStoCCS(point)); }

            // Projecting a point from the camera's image plane onto the cornea
            Vec3d project(Vec3d point) const;
            inline Vec3d project(Point2f point) const { return project(ICStoWCS(point)); }
            Point2d unproject(Vec3d point) const; // inverse of project()

            inline Tracker(EyeProperties eye, CameraProperties camera, Positions positions):
                eye(eye), camera(camera), positions(positions), KF(makeWCSKalmanFilter()) {};

            /* Submit a measurement to the Kálmán filter. The return value bypasses the Kálmán filter, i.e. the eye
             * position is calculated directly from the inputs without considering previous states. */
            EyePosition correct(Point2f reflectionPixel, Point2f pupilPixel);
            // Read a prediction from the Kálmán filter
            EyePosition predict();
    };
}
