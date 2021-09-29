#pragma once
#include <opencv2/opencv.hpp> // cv::Point2i
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xfixed.hpp>
#include <optional>
#include <vector>
#include "camera.hpp"
#include "eye.hpp"
namespace EyeTracker::Geometry {
    using namespace Camera;
    using namespace Eye;
    template<unsigned long... I> using Matrix = xt::xtensor_fixed<float, xt::xshape<I...>>;
    using Vector = Matrix<3>;
    struct EyePosition {
        std::optional<Vector> corneaCurvatureCentre, pupilCentre, eyeCentre; // c, p, d
        inline operator bool() const {
            return eyeCentre and pupilCentre and corneaCurvatureCentre;
        }
    };
    /* One-letter variable names are as defined by Guestrin & Eizenman, unless defined in comments.
     * All constants are in millimetres unless otherwise indicated. */
    constexpr float LAMBDA = 27.119;
    const Vector o({0, 0, -320});
    const Matrix<3, 3> rotation({{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}); // dimensionless
    const Vector light = o + Vector({0, -50, 0});

    // Conversions between coordinate systems
    inline Vector pixelToCCS(cv::Point2i point) {
        const float x = PIXEL_PITCH * (point.x - RESOLUTION_X/2.0);
        const float y = PIXEL_PITCH * (point.y - RESOLUTION_Y/2.0);
        return {x, y, -LAMBDA};
    }

    inline Vector CCStoWCS(Vector point) {
        return xt::linalg::dot(rotation, point) + o;
    }

    inline Vector WCStoCCS(Vector point) {
        return xt::linalg::solve(rotation, point - o);
    }

    inline cv::Point2i CCStoPixel(Vector point) {
        return {static_cast<int>(RESOLUTION_X/2.0 + point(0)/PIXEL_PITCH),
                static_cast<int>(RESOLUTION_Y/2.0 + point(1)/PIXEL_PITCH)};
    }

    inline Vector pixelToWCS(cv::Point2i point) {
        return CCStoWCS(pixelToCCS(point));
    }

    inline cv::Point2i WCStoPixel(Vector point) {
        return CCStoPixel(WCStoCCS(point));
    }

    /* Given the centre of a sphere, its radius, a position vector for a point on a line, and a direction vector
     * for that line, find the intersections of the line and the sphere. */
    std::vector<Vector> lineSphereIntersections(Vector sphereCentre, float radius, Vector linePoint, Vector lineDirection);

    EyePosition eyePosition(cv::Point2i reflectionPixel, cv::Point2i pupilPixel);
}
