#include "eye_tracking.hpp"
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath> // std::pow, std::abs
#include <iostream>

namespace EyeTracking {
    using cv::Vec2d;

    std::vector<RatedCircleCentre>
    findCircles(const cv::cuda::GpuMat &frame, CircleConstraints constraints, cv::cuda::GpuMat &thresholded) {
        // these are static as they are reused between invocations
        const static cv::Mat morphologyElement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(13, 13));
        const static cv::Ptr<cv::cuda::Filter> morphOpen = cv::cuda::createMorphologyFilter(cv::MORPH_OPEN, CV_8UC1,
                                                                                            morphologyElement);
        const static cv::Ptr<cv::cuda::Filter> morphClose = cv::cuda::createMorphologyFilter(cv::MORPH_CLOSE, CV_8UC1,
                                                                                             morphologyElement);

        const static cv::Mat dilateElement = cv::getStructuringElement(cv::MORPH_RECT,
                                                                       cv::Size(constraints.dilationSize,
                                                                                constraints.dilationSize));
        const static cv::Mat erodeElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(constraints.erosionSize,
                                                                                               constraints.erosionSize));
        const static cv::Ptr<cv::cuda::Filter> dilateFilter = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE,
                                                                                               CV_8UC1, dilateElement);
        const static cv::Ptr<cv::cuda::Filter> erodeFilter = cv::cuda::createMorphologyFilter(cv::MORPH_ERODE, CV_8UC1,
                                                                                              erodeElement);


        cv::cuda::threshold(frame, thresholded, constraints.threshold, 255, cv::THRESH_BINARY_INV);
        dilateFilter->apply(thresholded, thresholded);
        erodeFilter->apply(thresholded, thresholded);
//        morphOpen->apply(thresholded, thresholded);
//        morphClose->apply(thresholded, thresholded);

        // Have to download the frame from the GPU to work with contours
        const cv::Mat thresh_cpu(thresholded);
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy; // Unused output
        std::vector<RatedCircleCentre> result;
        cv::findContours(thresh_cpu, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        for (const std::vector<cv::Point> &contour: contours) {
            /* For each contour:
             * - calculate the minimum enclosing circle
             * - if its radius is outside the range given by constraints, skip it
             * - otherwise, calculate its area and that of the contour
             * - calculate the fraction of the circle's area filled by the contour
             * - if it exceeds constraints.minRating, add it to the results */
            cv::Point2f centre;
            float radius;
            cv::minEnclosingCircle(contour, centre, radius);
            if (radius < constraints.minRadius or radius > constraints.maxRadius) continue;
            const float contour_area = cv::contourArea(contour);
            if (contour_area <= 0) continue;
            const float circle_area = 3.14159265358979323846 * std::pow(radius, 2);
            float rating = contour_area / circle_area;
            if (rating >= constraints.minRating) result.push_back({centre, rating, radius});
        }
        return result;
    }

    std::vector<Vec3d> lineSphereIntersections(Vec3d sphereCentre, float radius, Vec3d linePoint, Vec3d lineDirection) {
        /* We are looking for points of the form linePoint + k*lineDirection, which are also radius away
         * from sphereCentre. This can be expressed as a quadratic in k: ak² + bk + c = radius². */
        const double a = cv::norm(lineDirection, cv::NORM_L2SQR);
        const double b = 2 * lineDirection.dot(linePoint - sphereCentre);
        const double c = cv::norm(linePoint, cv::NORM_L2SQR) + cv::norm(sphereCentre, cv::NORM_L2SQR) -
                         2 * linePoint.dot(sphereCentre);
        const double DISCRIMINANT = std::pow(b, 2) - 4 * a * (c - std::pow(radius, 2));
        if (std::abs(DISCRIMINANT) < 1e-6) return {linePoint - lineDirection * b / (2 * a)}; // One solution
        else if (DISCRIMINANT < 0) return {}; // No solutions
        else { // Two solutions
            const double sqrtDISCRIMINANT = std::sqrt(DISCRIMINANT);
            return {linePoint + lineDirection * (-b + sqrtDISCRIMINANT) / (2 * a),
                    linePoint + lineDirection * (-b - sqrtDISCRIMINANT) / (2 * a)};
        }
    }

    cv::KalmanFilter Tracker::makeICSKalmanFilter() const {
        constexpr static double VELOCITY_DECAY = 0.9;
        const static cv::Mat TRANSITION_MATRIX = (KFMat(4, 4) << 1, 0, camera.dt(), 0,
                0, 1, 0, camera.dt(),
                0, 0, VELOCITY_DECAY, 0,
                0, 0, 0, VELOCITY_DECAY);
        const static cv::Mat MEASUREMENT_MATRIX = (KFMat(2, 4) << 1, 0, 0, 0,
                0, 1, 0, 0);
        const static cv::Mat PROCESS_NOISE_COV = cv::Mat::eye(4, 4, CV_64F) * 100;
        const static cv::Mat MEASUREMENT_NOISE_COV = cv::Mat::eye(2, 2, CV_64F) * 50;
        const static cv::Mat ERROR_COV_POST = cv::Mat::eye(4, 4, CV_64F) * 0.1;
        const static cv::Mat STATE_POST = (KFMat(4, 1) << camera.resolution.width / 2.0, camera.resolution.height /
                                                                                         2.0, 0, 0);

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

    cv::KalmanFilter Tracker::makeWCSKalmanFilter() const {
        constexpr static double VELOCITY_DECAY = 0.9;
        const static cv::Mat TRANSITION_MATRIX = (KFMat(6, 6) << 1, 0, 0, camera.dt(), 0, 0,
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

    Vec3d Tracker::ICStoCCS(Point2d point) const {
        const double x = camera.pixelPitch * (point.x - camera.resolution.width / 2.0);
        const double y = camera.pixelPitch * (point.y - camera.resolution.height / 2.0);
        return {x, y, -positions.lambda};
    }

    Vec3d Tracker::CCStoWCS(Vec3d point) const {
        return positions.rotation * point + positions.nodalPoint;
    }

    Vec3d Tracker::WCStoCCS(Vec3d point) const {
        Vec3d ret;
        /* Warning: return value of cv::solve is not checked;
         * if there is no solution, ret won't be set by the line below! */
        cv::solve(positions.rotation, point - positions.nodalPoint, ret);
        return ret;
    }

    Point2d Tracker::CCStoICS(Vec3d point) const {
        return static_cast<Point2d>(camera.resolution) / 2 + Point2d(point(0), point(1)) / camera.pixelPitch;
    }

    Vec3d Tracker::project(Vec3d point) const {
        return positions.nodalPoint + positions.cameraEyeProjectionFactor * (positions.nodalPoint - point);
    }

    Point2d Tracker::unproject(Vec3d point) const {
        return WCStoICS((point - (1 + positions.cameraEyeProjectionFactor) * positions.nodalPoint) /
                        -positions.cameraEyeProjectionFactor);
    }

    EyePosition Tracker::correct(Point2f reflectionPixel, Point2f pupilPixel, Vec3d light) {
        if (temporalPositionsReady) {
           positions.cameraEyeDistance = temporalPositions.cameraEyeDistance;
           positions.lambda = temporalPositions.lambda;
           positions.cameraEyeProjectionFactor = temporalPositions.cameraEyeProjectionFactor;
           positions.light1 = temporalPositions.light1;
           positions.light2 = temporalPositions.light2;
           temporalPositionsReady = false;
        }

        /* This code is based on Guestrin & Eizenman, pp1125-1126.
         * Algorithm:
         * - convert reflectionPixel to the WCS
         * - project it onto the presumed position of the eye (using positions.cameraEyeDistance)
         * - use (3), the fact that the light (l), the Purkyně reflection (q), the camera nodal point (o), and the
         *   centre of curvature of the cornea (c) are coplanar; and (4), the law of reflection, to find a line
         *   containing c.
         * - use (2), the fact that the cornea is spherical and its radius is known, to find c (find the intersection
         *   of the line and the sphere, and use the Z coordinate to discriminate between multiple intersections.
         * - use (6), the fact that the point of refraction of the pupil centre also lies on the cornea, to project
         *   pupilPixel onto the cornea and find this point.
         * - use (7), the coplanarity of the pupil centre (p), pupil centre's point of refraction (r), o and c, and
         *   Snell's law to find a line containing p.
         * - use (9) to find p (again, finding the intersections of a sphere and the line).
         * - trace the vector p-c to find d, the centre of rotation of the eye. */

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
        Vec3d loqo = (light - positions.nodalPoint).cross(reflection - positions.nodalPoint);

        // Now dot(loqo, c) = dot(loqo, o) - a plane on which c must lie.
        // (4):
        Vec3d lqoq = (light - reflection) * cv::norm(positions.nodalPoint - reflection);
        Vec3d oqlq = (positions.nodalPoint - reflection) * cv::norm(light - reflection);
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
        } else { // Far more likely
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
            Vec3d pupilImage = ICStoWCS(pupilPixel);
            intersections = lineSphereIntersections(*corneaCurvatureCentre, eye.R, pupilImage,
                                                    positions.nodalPoint - pupilImage);
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
            double angle = std::asin(-1 / (n1orrcor * cv::norm(*pupil - *corneaCurvatureCentre))); // θ
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

            KF.correct(
                    (KFMat(3, 1) << (*corneaCurvatureCentre)(0), (*corneaCurvatureCentre)(1), (*corneaCurvatureCentre)(
                            2)));

            return {corneaCurvatureCentre, pupilCentre, eyeCentre};
        }
    }

    EyePosition Tracker::predict() {
        return {{}, {}, KF.predict()};
    }

    EyePosition Tracker::correct(Point2f reflectionPixel1, Point2f reflectionPixel2, Point2f pupilPixel) {

        mtx_image.lock();
        imagePositions.reflectionPixel1 = reflectionPixel1;
        imagePositions.reflectionPixel2 = reflectionPixel2;
        imagePositions.pupilPixel = pupilPixel;
        mtx_image.unlock();

        /* This code is based on Guestrin & Eizenman, pp1125-1126.
         * Algorithm:
         * - convert reflectionPixel to the WCS
         * - project it onto the presumed position of the eye (using positions.cameraEyeDistance)
         * - use (3), the fact that the light (l), the Purkyně reflection (q), the camera nodal point (o), and the
         *   centre of curvature of the cornea (c) are coplanar; and (4), the law of reflection, to find a line
         *   containing c.
         * - use (2), the fact that the cornea is spherical and its radius is known, to find c (find the intersection
         *   of the line and the sphere, and use the Z coordinate to discriminate between multiple intersections.
         * - use (6), the fact that the point of refraction of the pupil centre also lies on the cornea, to project
         *   pupilPixel onto the cornea and find this point.
         * - use (7), the coplanarity of the pupil centre (p), pupil centre's point of refraction (r), o and c, and
         *   Snell's law to find a line containing p.
         * - use (9) to find p (again, finding the intersections of a sphere and the line).
         * - trace the vector p-c to find d, the centre of rotation of the eye. */

        Vec3d reflection1 = project(reflectionPixel1); // u
        Vec3d reflection2 = project(reflectionPixel2); // u
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
        Vec3d loqo1 = (positions.light1 - positions.nodalPoint).cross(reflection1 - positions.nodalPoint);
        Vec3d loqo2 = (positions.light2 - positions.nodalPoint).cross(reflection2 - positions.nodalPoint);

        // Now dot(loqo, c) = dot(loqo, o) - a plane on which c must lie.
        // (4):
        Vec3d lqoq1 = (positions.light1 - reflection1) * cv::norm(positions.nodalPoint - reflection1);
        Vec3d lqoq2 = (positions.light2 - reflection2) * cv::norm(positions.nodalPoint - reflection2);
        Vec3d oqlq1 = (positions.nodalPoint - reflection1) * cv::norm(positions.light1 - reflection1);
        Vec3d oqlq2 = (positions.nodalPoint - reflection2) * cv::norm(positions.light2 - reflection2);
        Vec3d oqlqlqoq1 = oqlq1 - lqoq1;
        Vec3d oqlqlqoq2 = oqlq2 - lqoq2;
        // Now dot(oqlqlqoq, c) = dot(oqlqlqoq, q) - another plane containing c.
        // The intersection of these two planes is a line.
        cv::Matx22d squarePlaneMatrix1(loqo1(0), loqo1(1), oqlqlqoq1(0), oqlqlqoq1(1));
        cv::Matx22d squarePlaneMatrix2(loqo2(0), loqo2(1), oqlqlqoq2(0), oqlqlqoq2(1));
        // Calculate rank
        cv::Mat1d singularValues1;
        cv::Mat1d singularValues2;
        cv::Mat leftSingularVectors, rightSingularVectorsT; // Unused outputs
        cv::SVDecomp(squarePlaneMatrix1, singularValues1, leftSingularVectors, rightSingularVectorsT, cv::SVD::NO_UV);
        cv::SVDecomp(squarePlaneMatrix2, singularValues2, leftSingularVectors, rightSingularVectorsT, cv::SVD::NO_UV);
        if (cv::countNonZero(singularValues1 > 1e-4) < 2 or cv::countNonZero(singularValues2 > 1e-4) < 2) {
            /* The line lies in the plane z = 0.
             * Very unexpected, as the eye and the camera are facing each other on the z axis.
             * Should not occur in normal operation.
             * Can still be solved, just requires writing a lot of extra code for a situation that should never occur.
             * It will also break the logic used to distinguish between duplicate solutions of quadratics
             * (which assumes that the eye is facing roughly in the negative z direction). */
            return {};
        } else { // Far more likely
            // We now consider z = 0 and z = 1, and find two points (x, y, 0) and (x', y', 1), which define the line.
            Vec2d b1(loqo1.dot(positions.nodalPoint), oqlqlqoq1.dot(reflection1));
            Vec2d b2(loqo2.dot(positions.nodalPoint), oqlqlqoq2.dot(reflection2));
            Vec2d lastRow1(loqo1(2), oqlqlqoq1(2));
            Vec2d lastRow2(loqo2(2), oqlqlqoq2(2));
            Vec2d pointA_xy1, pointB_xy1;
            Vec2d pointA_xy2, pointB_xy2;
            if (!cv::solve(squarePlaneMatrix1, b1, pointA_xy1)) return {}; // z = 0
            if (!cv::solve(squarePlaneMatrix2, b2, pointA_xy2)) return {}; // z = 0
            if (!cv::solve(squarePlaneMatrix1, b1 - lastRow1, pointB_xy1)) return {}; // z = 1
            if (!cv::solve(squarePlaneMatrix2, b2 - lastRow2, pointB_xy2)) return {}; // z = 1
            Vec2d direction_xy1 = pointB_xy1 - pointA_xy1;
            Vec2d direction_xy2 = pointB_xy2 - pointA_xy2;
            Vec3d pointA1(pointA_xy1(0), pointA_xy1(1), 0);
            Vec3d pointA2(pointA_xy2(0), pointA_xy2(1), 0);
            Vec3d direction1(direction_xy1(0), direction_xy1(1), 1);
            Vec3d direction2(direction_xy2(0), direction_xy2(1), 1);

            /* Now we have q, the centre of a sphere of radius R on which c lies (2), and two points, pointA and pointB,
             * defining a line on which c also lies. */
            std::vector<Vec3d> intersections = lineSphereIntersections(reflection1, eye.R, pointA1, direction1);
            std::optional<Vec3d> corneaCurvatureCentre1, corneaCurvatureCentre2; // c
            switch (intersections.size()) {
                case 1:
                    corneaCurvatureCentre1 = intersections[0];
                    break;
                case 2:
                    // Take the one with the highest Z. The eye can't be pointed backwards...
                    corneaCurvatureCentre1 = intersections[intersections[0](2) > intersections[1](2) ? 0 : 1];
                    break;
            }
            if (!corneaCurvatureCentre1) return {};

            intersections = lineSphereIntersections(reflection2, eye.R, pointA2, direction2);
            switch (intersections.size()) {
                case 1:
                    corneaCurvatureCentre2 = intersections[0];
                    break;
                case 2:
                    // Take the one with the highest Z. The eye can't be pointed backwards...
                    corneaCurvatureCentre2 = intersections[intersections[0](2) > intersections[1](2) ? 0 : 1];
                    break;
            }
            if (!corneaCurvatureCentre2) return {};

            std::optional<Vec3d> corneaCurvatureCentre = (*corneaCurvatureCentre1 + *corneaCurvatureCentre2) / 2;

            // (6): We now project the pupil from the image sensor (flat) onto the cornea (spherical).
            Vec3d pupilImage = ICStoWCS(pupilPixel);
            intersections = lineSphereIntersections(*corneaCurvatureCentre, eye.R, pupilImage,
                                                    positions.nodalPoint - pupilImage);
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
            if (!pupil) {
                return {corneaCurvatureCentre};
            }; // No solution, but at least we have c

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
            double angle = std::asin(-1 / (n1orrcor * cv::norm(*pupil - *corneaCurvatureCentre))); // θ
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
            Vec3d direction = (*corneaCurvatureCentre - *pupil) * std::cos(angle)
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
            if (!pupilCentre) {
                // No solution for p, but at least we have c.
                return {corneaCurvatureCentre};
            }
            /* We have p and c. Together, they give the position and orientation of the eye. We now need to trace the
             * line p - c to the point d, the centre of rotation of the eye, using D, a further eye parameter not used
             * by G&E. d will be our head position.
             * NB: The eye is not actually spherical, so this may move around in unexpected ways. */

            Vec3d eyeCentre = *pupilCentre + eye.D * (*corneaCurvatureCentre - *pupilCentre)
                                             / cv::norm(*corneaCurvatureCentre - *pupilCentre);

            KF.correct(
                    (KFMat(3, 1) << (*corneaCurvatureCentre)(0), (*corneaCurvatureCentre)(1), (*corneaCurvatureCentre)(
                            2)));

            mtx_eye.lock();
            eyePosition = {corneaCurvatureCentre, pupilCentre, eyeCentre};
            mtx_eye.unlock();
            return {corneaCurvatureCentre, pupilCentre, eyeCentre};
        }
    }

    void Tracker::getEyePosition(EyePosition &eyePosition) {
        mtx_eye.lock();
        eyePosition = this->eyePosition;
        mtx_eye.unlock();
    }

    void Tracker::getImagePositions(ImagePositions &imagePositions) {
        mtx_image.lock();
        imagePositions = this->imagePositions;
        mtx_image.unlock();
    }

    void Tracker::setNewParameters(float lambda, Vec3d nodalPoint, Vec3d light1, Vec3d light2) {
        temporalPositions.cameraEyeDistance = -nodalPoint(2);
        temporalPositions.lambda = lambda;
        temporalPositions.cameraEyeProjectionFactor = positions.cameraEyeDistance / lambda;
        temporalPositions.light1 = light1;
        temporalPositions.light2 = light2;
        temporalPositionsReady = true;
    }
}
