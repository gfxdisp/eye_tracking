#include "eye_tracking.hpp"
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath> // std::pow, std::abs
namespace EyeTracking {
    using cv::Vec2d;

    std::vector<PointWithRating> findCircles(const cv::cuda::GpuMat& frame, uint8_t thresh, float min_radius, float max_radius, float max_rating) {
        const static cv::Mat morphologyElement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(13, 13));
        const static cv::Ptr<cv::cuda::Filter> morphOpen  = cv::cuda::createMorphologyFilter(cv::MORPH_OPEN,  CV_8UC1, morphologyElement);
        const static cv::Ptr<cv::cuda::Filter> morphClose = cv::cuda::createMorphologyFilter(cv::MORPH_CLOSE, CV_8UC1, morphologyElement);
        cv::cuda::GpuMat thresholded;
        cv::cuda::threshold(frame, thresholded, thresh, 255, cv::THRESH_BINARY_INV);
        morphOpen->apply(thresholded, thresholded);
        morphClose->apply(thresholded, thresholded);
        const cv::Mat thresh_cpu(thresholded);
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        std::vector<PointWithRating> result;
        cv::findContours(thresh_cpu, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        for (const std::vector<cv::Point>& contour : contours) {
            cv::Point2f centre;
            float radius;
            cv::minEnclosingCircle(contour, centre, radius);
            if (radius < min_radius or radius > max_radius) continue;
            const float contour_area = cv::contourArea(contour);
            if (contour_area <= 0) continue;
            const float circle_area = 3.14159265358979323846 * std::pow(radius, 2);
            const float rating = std::pow((circle_area-contour_area)/circle_area, 2);
            if (rating <= max_rating) result.push_back({centre, rating});
        }
        return result;
    }

    std::vector<Vec3d> lineSphereIntersections(Vec3d sphereCentre, float radius, Vec3d linePoint, Vec3d lineDirection) {
        const double a = cv::norm(lineDirection, cv::NORM_L2SQR);
        const double b = 2 * lineDirection.dot(linePoint - sphereCentre);
        const double c = cv::norm(linePoint, cv::NORM_L2SQR) + cv::norm(sphereCentre, cv::NORM_L2SQR) - 2 * linePoint.dot(sphereCentre);
        const double DISCRIMINANT = std::pow(b, 2) - 4 * a * (c - std::pow(radius, 2));
        if (std::abs(DISCRIMINANT) < 1e-6) return {linePoint - lineDirection*b/(2*a)}; // One solution
        else if (DISCRIMINANT < 0) return {}; // No solutions
        else { // Two solutions
            const double sqrtDISCRIMINANT = std::sqrt(DISCRIMINANT);
            return {linePoint + lineDirection*(-b+sqrtDISCRIMINANT)/(2*a),
                    linePoint + lineDirection*(-b-sqrtDISCRIMINANT)/(2*a)};
        }
    }

    cv::KalmanFilter Tracker::makePixelKalmanFilter() const {
        constexpr static double VELOCITY_DECAY = 0.9;
        const static cv::Mat TRANSITION_MATRIX  = (KFMat(4, 4) << 1, 0, camera.dt(), 0,
                                                                  0, 1, 0, camera.dt(),
                                                                  0, 0, VELOCITY_DECAY, 0,
                                                                  0, 0, 0, VELOCITY_DECAY);
        const static cv::Mat MEASUREMENT_MATRIX = (KFMat(2, 4) << 1, 0, 0, 0,
                                                                  0, 1, 0, 0);
        const static cv::Mat PROCESS_NOISE_COV = cv::Mat::eye(4, 4, CV_64F) * 100;
        const static cv::Mat MEASUREMENT_NOISE_COV = cv::Mat::eye(2, 2, CV_64F) * 50;
        const static cv::Mat ERROR_COV_POST = cv::Mat::eye(4, 4, CV_64F) * 0.1;
        const static cv::Mat STATE_POST = (KFMat(4, 1) << camera.resolution.width/2.0, camera.resolution.height/2.0, 0, 0);

        cv::KalmanFilter KF(4, 2);
        // clone() is needed as, otherwise, the matrices will be used by reference, and all the filters will be the same
        KF.transitionMatrix = TRANSITION_MATRIX.clone();
        KF.measurementMatrix = MEASUREMENT_MATRIX.clone();
        KF.processNoiseCov = PROCESS_NOISE_COV.clone();
        KF.measurementNoiseCov = MEASUREMENT_NOISE_COV.clone();
        KF.errorCovPost = ERROR_COV_POST.clone();
        KF.statePost = STATE_POST.clone();
        KF.predict(); // Without this line, OpenCV complains about incorrect matrix dimensions
        return KF;
    }

    cv::KalmanFilter Tracker::make3DKalmanFilter() const {
        constexpr static double VELOCITY_DECAY = 0.9;
        const static cv::Mat TRANSITION_MATRIX  = (KFMat(6, 6) << 1, 0, 0, camera.dt(), 0, 0,
                                                                  0, 1, 0, 0, camera.dt(), 0,
                                                                  0, 0, 1, 0, 0, camera.dt(),
                                                                  0, 0, 0, VELOCITY_DECAY, 0, 0,
                                                                  0, 0, 0, 0, VELOCITY_DECAY, 0,
                                                                  0, 0, 0, 0, 0, VELOCITY_DECAY);
        const static cv::Mat MEASUREMENT_MATRIX = cv::Mat::eye(3, 6, CV_64F);
        const static cv::Mat PROCESS_NOISE_COV = cv::Mat::eye(6, 6, CV_64F) * 100;
        const static cv::Mat MEASUREMENT_NOISE_COV = cv::Mat::eye(3, 3, CV_64F) * 50;
        const static cv::Mat ERROR_COV_POST = cv::Mat::eye(6, 6, CV_64F) * 0.1;
        const static cv::Mat STATE_POST = cv::Mat::zeros(6, 1, CV_64F);

        cv::KalmanFilter KF(6, 3);
        // clone() is needed as, otherwise, the matrices will be used by reference, and all the filters will be the same
        KF.transitionMatrix = TRANSITION_MATRIX.clone();
        KF.measurementMatrix = MEASUREMENT_MATRIX.clone();
        KF.processNoiseCov = PROCESS_NOISE_COV.clone();
        KF.measurementNoiseCov = MEASUREMENT_NOISE_COV.clone();
        KF.errorCovPost = ERROR_COV_POST.clone();
        KF.statePost = STATE_POST.clone();
        KF.predict(); // Without this line, OpenCV complains about incorrect matrix dimensions
        return KF;
    }

    Vec3d Tracker::pixelToCCS(Point2d point) const {
        const double x = camera.pixelPitch * (point.x - camera.resolution.width/2.0);
        const double y = camera.pixelPitch * (point.y - camera.resolution.height/2.0);
        return {x, y, -positions.lambda};
    }

    Vec3d Tracker::CCStoWCS(Vec3d point) const {
        return positions.rotation * point + positions.nodalPoint;
    }

    Vec3d Tracker::WCStoCCS(Vec3d point) const {
        Vec3d ret;
        // Warning: return value of cv::solve is not checked; if there is no solution, ret won't be set by the line below!
        cv::solve(positions.rotation, point - positions.nodalPoint, ret);
        return ret;
    }

    Point2d Tracker::CCStoPixel(Vec3d point) const {
        return static_cast<Point2d>(camera.resolution)/2 + Point2d(point(0), point(1))/camera.pixelPitch;
    }

    Vec3d Tracker::project(Vec3d point) const {
        return positions.nodalPoint + positions.cameraEyeProjectionFactor * (positions.nodalPoint - point);
    }

    Point2d Tracker::unproject(Vec3d point) const {
        return WCStoPixel((point - (1 + positions.cameraEyeProjectionFactor) * positions.nodalPoint)/-positions.cameraEyeProjectionFactor);
    }

    EyePosition Tracker::correct(Point2f reflectionPixel, Point2f pupilPixel) {
        // This code should be read in conjunction with Guestrin & Eizenman, pp1125-1126.
        Vec3d reflection = project(reflectionPixel); // u
        /* Equation numbering is as in G&E.
         * (3): l, q, o, c are coplanar.
         * (4): angle of incidence = angle of reflection.
         * (2): The corneal reflection lies on the cornea (i.e. at a distance R from its centre of curvature).
         * (intentionally out of order)
         * We use the above to obtain three scalar equations in three scalar unknowns, and thus find c.
         * (6): The point of refraction of the pupil centre lies on the cornea.
         * We use c and (6) to determine r, the point of refraction of the ray from the pupil centre.
         * (7): p, r, o, c are coplanar.
         * (8): Snell's law.
         * (9): p and c lie a distance K apart.
         * p and c are the unknowns in (7-9). Having found c using (2-4), we can now find p. */
        // (3):
        Vec3d loqo = (positions.light - positions.nodalPoint).cross(reflection - positions.nodalPoint);
        // Now dot(loqo, c) = dot(loqo, o) - a plane on which c must lie.
        // (4):
        Vec3d lqoq = (positions.light - reflection) * cv::norm(positions.nodalPoint - reflection);
        Vec3d oqlq = (positions.nodalPoint - reflection) * cv::norm(positions.light - reflection);
        Vec3d oqlqlqoq = oqlq - lqoq;
        // Now dot(oqlqlqoq, c) = dot(oqlqlqoq, q) - another plane containing c.
        // The intersection of these two planes is a line.
        cv::Matx22d squarePlaneMatrix(loqo(0), loqo(1), oqlqlqoq(0), oqlqlqoq(1));
        // Calculate rank
        cv::Mat1d singularValues;
        cv::Mat leftSingularVectors, rightSingularVectorsT; // Unused outputs
        cv::SVDecomp(squarePlaneMatrix, singularValues, leftSingularVectors, rightSingularVectorsT, cv::SVD::NO_UV);
        if (cv::countNonZero(singularValues > 1e-4) < 2) {
            /* The line lies in the plane z = 0.
             * Very unexpected, as the eye and the camera are facing each other on the z axis.
             * Should not occur in normal operation.
             * Can still be solved, just requires writing a lot of extra code for a situation that should never occur.
             * It will also break the logic used to distinguish between duplicate solutions of quadratics
             * (which assumes that the eye is facing roughly in the negative z direction). */
            return {};
        }
        else { // Far more likely
            // We now consider z = 0 and z = 1, and find two points (x, y, 0) and (x', y', 1), which define the line.
            Vec2d b(loqo.dot(positions.nodalPoint), oqlqlqoq.dot(reflection));
            Vec2d lastRow(loqo(2), oqlqlqoq(2));
            Vec2d pointA_xy, pointB_xy;
            if (!cv::solve(squarePlaneMatrix, b, pointA_xy)) return {}; // z = 0
            if (!cv::solve(squarePlaneMatrix, b - lastRow, pointB_xy)) return {}; // z = 1
            Vec2d direction_xy = pointB_xy - pointA_xy;
            Vec3d pointA(pointA_xy(0), pointA_xy(1), 0);
            Vec3d direction(direction_xy(0), direction_xy(1), 1);
            /* Now we have q, the centre of a sphere of radius R on which c lies (2), and two points, pointA and pointB,
             * defining a line on which c also lies. */
            std::vector<Vec3d> intersections = lineSphereIntersections(reflection, eye.R, pointA, direction);
            std::optional<Vec3d> corneaCurvatureCentre; // c
            switch (intersections.size()) {
                case 1:
                    corneaCurvatureCentre = intersections[0];
                    break;
                case 2:
                    // Take the one with the highest Z. The eye can't be pointed backwards...
                    corneaCurvatureCentre = intersections[intersections[0](2) > intersections[1](2) ? 0 : 1];
                    break;
            }
            if (!corneaCurvatureCentre) return {};

            // (6): We now project the pupil from the image sensor (flat) onto the cornea (spherical).
            Vec3d pupilImage = pixelToWCS(pupilPixel);
            intersections = lineSphereIntersections(*corneaCurvatureCentre, eye.R, pupilImage, positions.nodalPoint - pupilImage);
            std::optional<Vec3d> pupil;
            switch (intersections.size()) {
                case 1:
                    pupil = intersections[0];
                    break;
                case 2:
                    // Take the one with the lowest Z.
                    pupil = intersections[intersections[0](2) < intersections[1](2) ? 0 : 1];
                    break;
            }
            if (!pupil) return {corneaCurvatureCentre}; // No solution, but at least we have c

            // Now we find p in a somewhat similar way.
            // (7):
            Vec3d roco = (*pupil - positions.nodalPoint).cross(*corneaCurvatureCentre - positions.nodalPoint);
            // Now dot(roco, p) = dot(roco, o) - a plane containing p.
            // (8): n_1 · ‖o - r‖ / ‖(r - c) × (o - r)‖ = ‖p - r‖ / ‖(r - c) × (p - r)‖
            double n1orrcor = eye.n1 * cv::norm(positions.nodalPoint - *pupil)
                              / cv::norm((*pupil - *corneaCurvatureCentre).cross(positions.nodalPoint - *pupil));
            /* This is easier to solve if we extract the angle from the remaining × product:
             * ‖p - r‖ / ‖(r - c) × (p - r)‖ = ‖p - r‖ / (‖r - c‖ · ‖p - r‖ · sin(π+θ))
             * where θ = ∠PRC, the angle between the optic axis of the eye and the
             * normal at the point of refraction of the pupil centre.
             * The ‖p - r‖ term cancels, and we are left with
             * n1orrcor * ‖r - c‖ = 1 / sin(π+θ). */
            double angle = std::asin(-1/(n1orrcor * cv::norm(*pupil - *corneaCurvatureCentre))); // θ
            /* We now have three constraints on p: a plane, the angle ∠PRC, and the sphere of radius K centred on c.
             * It is easy to combine the first two contraints: (7) states that p, r, o and c are coplanar.
             * Furthermore, ∠PRC is known. This allows us to construct a ray from r in the direction of p, which lies
             * in the plane of p, r, o and c.
             * Then, p lies at the intersection of this ray and the sphere.
             * p = r + μw, where w ∝ (p-r)
             * We construct w first.
             * roco is the normal of our plane, it is at 90° to w.
             * c - r is at θ to w. */
            Vec3d perpendicular = (*corneaCurvatureCentre - *pupil).cross(roco);
            perpendicular /= cv::norm(perpendicular);
            /* w = (c-r)*cos(θ) ± perpendicular*sin(θ)
             * https://math.stackexchange.com/a/2320448
             * This in itself is ambiguous: the w given by this formula can be on either side of r - c, the normal at
             * the point of refraction. However, because of how the cross products are oriented, the positive direction
             * seems to be the right one. */
            direction = (*corneaCurvatureCentre - *pupil) * std::cos(angle)
                        / cv::norm(*corneaCurvatureCentre - *pupil)
                        + perpendicular * std::abs(std::sin(angle)); // w
            intersections = lineSphereIntersections(*corneaCurvatureCentre, eye.K, *pupil, direction);
            std::optional<Vec3d> pupilCentre;
            switch (intersections.size()) {
                case 1:
                    pupilCentre = intersections[0];
                    break;
                case 2:
                    // Take the one with the lowest Z; the pupil is in front of the lens.
                    pupilCentre = intersections[intersections[0](2) < intersections[1](2) ? 0 : 1];
                    break;
            }
            if (!pupilCentre) return {corneaCurvatureCentre}; // No solution for p, but at least we have c.
            /* We have p and c. Together, they give the position and orientation of the eye. We now need to trace the
             * line p - c to the point d, the centre of rotation of the eye, using D, a further eye parameter not used
             * by G&E. d will be our head position.
             * NB: The eye is not actually spherical, so this may move around in unexpected ways. */
            Vec3d eyeCentre = *pupilCentre + eye.D * (*corneaCurvatureCentre - *pupilCentre)
                                                    / cv::norm(*corneaCurvatureCentre - *pupilCentre);
            KF.correct((KFMat(3, 1) << (*corneaCurvatureCentre)(0), (*corneaCurvatureCentre)(1), (*corneaCurvatureCentre)(2)));
            return {corneaCurvatureCentre, pupilCentre, eyeCentre};
        }
    }

    EyePosition Tracker::predict() {
        return {{}, {}, KF.predict()};
    }
}
