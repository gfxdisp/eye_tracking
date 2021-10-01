#pragma once
#include <opencv2/core.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <opencv2/video/tracking.hpp> // cv::KalmanFilter
#include <vector>
namespace EyeTracking {
    template<unsigned long... I> using Matrix = xt::xtensor_fixed<float, xt::xshape<I...>>;
    using Vector = Matrix<3>;

    struct CircleConstraints { // Type representing a circle to be detected
        uint8_t threshold; // Maximum brightness inside the circle
        float minRadius, maxRadius; // in pixels
        float maxRating = 0.05;
    };

    struct EyePosition {
        std::optional<Vector> corneaCurvatureCentre, pupilCentre, eyeCentre; // c, p, d
        inline operator bool() const {
            return eyeCentre and pupilCentre and corneaCurvatureCentre;
        }
    };

    struct PointWithRating {
        cv::Point2f point = {-1, -1};
        float rating = std::numeric_limits<float>::infinity();
        inline bool operator<(const PointWithRating& other) const {
            return rating < other.rating;
        }
    };

    const static cv::Point2f None = {-1, -1};
    using KFMat = cv::Mat_<float>;

    inline cv::Point2i toPoint(cv::Mat m) {
        return {static_cast<int>(m.at<float>(0, 0)), static_cast<int>(m.at<float>(0, 1))};
    }

    inline Vector toVector(cv::Mat m) {
        return {m.at<float>(0, 0), m.at<float>(0, 1), m.at<float>(0, 2)};
    }

    inline cv::Mat toMat(cv::Point p) {
        return (cv::Mat_<float>(2, 1) << p.x, p.y);
    }

    inline cv::Mat toMat(Vector v) {
        return (cv::Mat_<float>(3, 1) << v(0), v(1), v(2));
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
        float FPS; // Hz
        inline float dt() const { return 1/FPS; } // s
        int resolutionX; // px
        int resolutionY; // px
        float pixelPitch; // mm
        double exposureTime; // unknown units (ms?)
        double gain;
    };

    struct ImageProperties {
        cv::Rect ROI;
        CircleConstraints pupil, iris;
        float maxPupilIrisSeparation; // px
        float templateMatchingThreshold = 0.9;
    };

    struct Positions {
        float lambda;
        Vector nodalPoint; // o; nodal point of the camera
        Vector light; // l
        Matrix<3, 3> rotation; // dimensionless; rotation matrix from CCS to WCS
        float cameraEyeDistance;
        float cameraEyeProjectionFactor;
        inline Positions(float lambda,
                         Vector nodalPoint,
                         Vector light,
                         Matrix<3, 3> rotation = Matrix<3, 3>({{1, 0, 0}, {0, 1, 0}, {0, 0, 1}})):
            lambda(lambda),
            nodalPoint(nodalPoint),
            light(light),
            rotation(rotation),
            cameraEyeDistance(-nodalPoint(2)),
            cameraEyeProjectionFactor(cameraEyeDistance/lambda) {};
    };

    void correct(const cv::cuda::GpuMat& image, cv::cuda::GpuMat& result, float alpha=1, float beta=0, float gamma=0.5);
    std::vector<PointWithRating> findCircles(const cv::cuda::GpuMat& frame, uint8_t thresh, float min_radius, float max_radius, float max_rating);
    inline std::vector<PointWithRating> findCircles(const cv::cuda::GpuMat& frame, CircleConstraints constraints) {
        return findCircles(frame, constraints.threshold, constraints.minRadius, constraints.maxRadius, constraints.maxRating);
    }

    /* Given the centre of a sphere, its radius, a position vector for a point on a line, and a direction vector
     * for that line, find the intersections of the line and the sphere. */
    std::vector<Vector> lineSphereIntersections(Vector sphereCentre, float radius, Vector linePoint, Vector lineDirection);

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
            Vector pixelToCCS(cv::Point2f point) const;
            Vector CCStoWCS(Vector point) const;
            Vector WCStoCCS(Vector point) const;
            cv::Point2f CCStoPixel(Vector point) const;
            inline Vector pixelToWCS(cv::Point2f point) const { return CCStoWCS(pixelToCCS(point)); }
            inline cv::Point2f WCStoPixel(Vector point) const { return CCStoPixel(WCStoCCS(point)); }
            Vector project(Vector point) const;
            inline Vector project(cv::Point2f point) const { return project(pixelToWCS(point)); }
            cv::Point2f unproject(Vector point) const;
            inline Tracker(EyeProperties eye, CameraProperties camera, Positions positions):
                eye(eye), camera(camera), positions(positions), KF(make3DKalmanFilter()) {};
            EyePosition correct(cv::Point2f reflectionPixel, cv::Point2f pupilPixel);
            EyePosition predict();
    };
}
