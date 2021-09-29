#pragma once
#include <opencv2/core/types.hpp> // cv::Point2i
#include <xtensor-blas/xlinalg.hpp>
#include <vector>
#include "params.hpp"
#include "types.hpp"
namespace EyeTracker::Geometry {
    using namespace Params;
    using namespace Camera;
    using namespace Eye;
    using namespace Positions;

    // Conversions between coordinate systems
    inline Vector pixelToCCS(cv::Point2i point) {
        const float x = PIXEL_PITCH * (point.x - RESOLUTION_X/2.0);
        const float y = PIXEL_PITCH * (point.y - RESOLUTION_Y/2.0);
        return {x, y, -LAMBDA};
    }

    inline Vector CCStoWCS(Vector point) {
        return xt::linalg::dot(rotation, point) + nodalPoint;
    }

    inline Vector WCStoCCS(Vector point) {
        return xt::linalg::solve(rotation, point - nodalPoint);
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

    const static float CAMERA_EYE_PROJECTION_FACTOR = CAMERA_EYE_DISTANCE / LAMBDA;

    inline Vector project(Vector point) {
        return nodalPoint + CAMERA_EYE_PROJECTION_FACTOR * (nodalPoint - point);
    }

    inline Vector project(cv::Point2i point) {
        return project(pixelToWCS(point));
    }

    inline cv::Point2i unproject(Vector point) {
        return WCStoPixel((point - (1 + CAMERA_EYE_PROJECTION_FACTOR) * nodalPoint)/-CAMERA_EYE_PROJECTION_FACTOR);
    }

    /* Given the centre of a sphere, its radius, a position vector for a point on a line, and a direction vector
     * for that line, find the intersections of the line and the sphere. */
    std::vector<Vector> lineSphereIntersections(Vector sphereCentre, float radius, Vector linePoint, Vector lineDirection);

    EyePosition eyePosition(cv::Point2i reflectionPixel, cv::Point2i pupilPixel);
}
