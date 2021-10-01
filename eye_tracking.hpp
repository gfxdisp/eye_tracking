#pragma once
#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp> // cv::KalmanFilter
#include <vector>
namespace EyeTracking {
    using cv::Matx33d;
    using cv::Point2d;
    using cv::Point2f;
    using cv::Vec3d;

    struct CircleConstraints { // Type representing a circle to be detected
        uint8_t threshold; // Maximum brightness inside the circle
        float minRadius, maxRadius; // in pixels
        float maxRating = 0.05;
    };

    struct EyePosition {
        std::optional<Vec3d> corneaCurvatureCentre, pupilCentre, eyeCentre; // c, p, d
        inline operator bool() const {
            return eyeCentre and pupilCentre and corneaCurvatureCentre;
        }
    };

    struct PointWithRating {
        Point2f point = {-1, -1};
        float rating = std::numeric_limits<float>::infinity();
        inline bool operator<(const PointWithRating& other) const {
            return rating < other.rating;
        }
    };

    const static Point2f None = {-1, -1};
    using KFMat = cv::Mat_<double>;

    inline Point2d toPoint(cv::Mat m) {
        return {m.at<double>(0, 0), m.at<double>(0, 1)};
    }

    inline cv::Mat toMat(Point2d p) {
        return (cv::Mat_<double>(2, 1) << p.x, p.y);
    }

    struct EyeProperties {
        /* From Guestrin & Eizenman
         * They provide a calibration procedure, but in fact there is not much variation in these parameters. */
        float R = 7.8; // mm, radius of corneal curvature
        float K = 4.2; // mm, distance between pupil centre and centre of corneal curvature
        float n1 = 1.3375; // Standard Keratometric Index, refractive index of cornea and aqueous humour
        // From Bekerman, Gottlieb & Vaiman
        float D = 10; // mm, distance between pupil centre and centre of eye rotation
    };

    struct CameraProperties {
        double FPS; // Hz
        inline double dt() const { return 1/FPS; } // s
        cv::Size2i resolution; // px
        double pixelPitch; // mm
        double exposureTime; // unknown units (ms?)
        double gain;
    };

    struct ImageProperties {
        cv::Rect ROI;
        CircleConstraints pupil, iris;
        double maxPupilIrisSeparation; // px
        double templateMatchingThreshold = 0.9;
    };

    struct Positions {
        double lambda;
        Vec3d nodalPoint; // o; nodal point of the camera
        Vec3d light; // l
        Matx33d rotation; // dimensionless; rotation matrix from CCS to WCS
        double cameraEyeDistance;
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

    std::vector<PointWithRating> findCircles(const cv::cuda::GpuMat& frame, uint8_t thresh, float min_radius, float max_radius, float max_rating);
    inline std::vector<PointWithRating> findCircles(const cv::cuda::GpuMat& frame, CircleConstraints constraints) {
        return findCircles(frame, constraints.threshold, constraints.minRadius, constraints.maxRadius, constraints.maxRating);
    }

    /* Given the centre of a sphere, its radius, a position vector for a point on a line, and a direction vector
     * for that line, find the intersections of the line and the sphere. */
    std::vector<Vec3d> lineSphereIntersections(Vec3d sphereCentre, float radius, Vec3d linePoint, Vec3d lineDirection);

    class Tracker {
        protected:
            cv::KalmanFilter KF;
            const EyeProperties eye;
            const CameraProperties camera;
            const Positions positions;
        public:
            cv::KalmanFilter makePixelKalmanFilter() const;
            cv::KalmanFilter make3DKalmanFilter() const;
            // Conversions between coordinate systems
            Vec3d pixelToCCS(Point2d point) const;
            Vec3d CCStoWCS(Vec3d point) const;
            Vec3d WCStoCCS(Vec3d point) const;
            Point2d CCStoPixel(Vec3d point) const;
            inline Vec3d pixelToWCS(Point2d point) const { return CCStoWCS(pixelToCCS(point)); }
            inline Point2d WCStoPixel(Vec3d point) const { return CCStoPixel(WCStoCCS(point)); }
            Vec3d project(Vec3d point) const;
            inline Vec3d project(Point2f point) const { return project(pixelToWCS(point)); }
            Point2d unproject(Vec3d point) const;
            inline Tracker(EyeProperties eye, CameraProperties camera, Positions positions):
                eye(eye), camera(camera), positions(positions), KF(make3DKalmanFilter()) {};
            EyePosition correct(Point2f reflectionPixel, Point2f pupilPixel);
            EyePosition predict();
    };
}
