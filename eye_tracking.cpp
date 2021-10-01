#include "eye_tracking.hpp"
#include <xtensor/xfixed.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xoperation.hpp>
#include <xtensor/xnorm.hpp> // xt::norm_sq
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath> // std::pow, std::abs
namespace EyeTracking {
    void correct(const cv::cuda::GpuMat& image, cv::cuda::GpuMat& result, float alpha, float beta, float gamma) {
        const static xt::xtensor_fixed<float, xt::xshape<256>> RANGE = xt::arange<float>(0, 255, 1)/255;
        xt::xtensor_fixed<uint8_t, xt::xshape<256>> xLUT = xt::cast<uint8_t>(xt::clip(xt::pow(RANGE, gamma) * alpha * 255.0 + beta, 0, 255));
        cv::Mat mLUT(1, 256, CV_8UC1, xLUT.data());
        cv::Ptr<cv::cuda::LookUpTable> cudaLUT = cv::cuda::createLookUpTable(mLUT);
        cudaLUT->transform(image, result);
    }

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
            const float circle_area = xt::numeric_constants<float>::PI * std::pow(radius, 2);
            const float rating = xt::square((circle_area-contour_area)/circle_area);
            if (rating <= max_rating) result.push_back({centre, rating});
        }
        return result;
    }

    std::vector<Vector> lineSphereIntersections(Vector sphereCentre, float radius, Vector linePoint, Vector lineDirection) {
        const float a = xt::norm_sq(lineDirection)();
        const float b = 2 * xt::linalg::dot(lineDirection, linePoint - sphereCentre)();
        const float c = (xt::norm_sq(linePoint) + xt::norm_sq(sphereCentre) - 2 * xt::linalg::dot(linePoint, sphereCentre))();
        const float DISCRIMINANT = std::pow(b, 2) - 4 * a * (c - std::pow(radius, 2));
        if (std::abs(DISCRIMINANT) < 1e-6) return {linePoint - lineDirection*b/(2*a)}; // One solution
        else if (DISCRIMINANT < 0) return {}; // No solutions
        else { // Two solutions
            const float sqrtDISCRIMINANT = std::sqrt(DISCRIMINANT);
            return {linePoint + lineDirection*(-b+sqrtDISCRIMINANT)/(2*a),
                    linePoint + lineDirection*(-b-sqrtDISCRIMINANT)/(2*a)};
        }
    }

    cv::KalmanFilter Tracker::makePixelKalmanFilter() const {
        constexpr static float VELOCITY_DECAY = 0.9;
        const static cv::Mat TRANSITION_MATRIX  = (KFMat(4, 4) << 1, 0, camera.dt(), 0,
                                                                  0, 1, 0, camera.dt(),
                                                                  0, 0, VELOCITY_DECAY, 0,
                                                                  0, 0, 0, VELOCITY_DECAY);
        const static cv::Mat MEASUREMENT_MATRIX = (KFMat(2, 4) << 1, 0, 0, 0,
                                                                  0, 1, 0, 0);
        const static cv::Mat PROCESS_NOISE_COV = cv::Mat::eye(4, 4, CV_32F) * 100;
        const static cv::Mat MEASUREMENT_NOISE_COV = cv::Mat::eye(2, 2, CV_32F) * 50;
        const static cv::Mat ERROR_COV_POST = cv::Mat::eye(4, 4, CV_32F) * 0.1;
        const static cv::Mat STATE_POST = (KFMat(4, 1) << camera.resolutionX/2.0, camera.resolutionY/2.0, 0, 0);

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
        constexpr static float VELOCITY_DECAY = 0.9;
        const static cv::Mat TRANSITION_MATRIX  = (KFMat(6, 6) << 1, 0, 0, camera.dt(), 0, 0,
                                                                  0, 1, 0, 0, camera.dt(), 0,
                                                                  0, 0, 1, 0, 0, camera.dt(),
                                                                  0, 0, 0, VELOCITY_DECAY, 0, 0,
                                                                  0, 0, 0, 0, VELOCITY_DECAY, 0,
                                                                  0, 0, 0, 0, 0, VELOCITY_DECAY);
        const static cv::Mat MEASUREMENT_MATRIX = cv::Mat::eye(3, 6, CV_32F);
        const static cv::Mat PROCESS_NOISE_COV = cv::Mat::eye(6, 6, CV_32F) * 100;
        const static cv::Mat MEASUREMENT_NOISE_COV = cv::Mat::eye(3, 3, CV_32F) * 50;
        const static cv::Mat ERROR_COV_POST = cv::Mat::eye(6, 6, CV_32F) * 0.1;
        const static cv::Mat STATE_POST = cv::Mat::zeros(6, 1, CV_32F);

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

    Vector Tracker::pixelToCCS(cv::Point2f point) const {
        const float x = camera.pixelPitch * (point.x - camera.resolutionX/2.0);
        const float y = camera.pixelPitch * (point.y - camera.resolutionY/2.0);
        return {x, y, -positions.lambda};
    }

    Vector Tracker::CCStoWCS(Vector point) const {
        return xt::linalg::dot(positions.rotation, point) + positions.nodalPoint;
    }

    Vector Tracker::WCStoCCS(Vector point) const {
        return xt::linalg::solve(positions.rotation, point - positions.nodalPoint);
    }

    cv::Point2f Tracker::CCStoPixel(Vector point) const {
        return {camera.resolutionX/2.0 + point(0)/camera.pixelPitch,
                camera.resolutionY/2.0 + point(1)/camera.pixelPitch};
    }

    Vector Tracker::project(Vector point) const {
        return positions.nodalPoint + positions.cameraEyeProjectionFactor * (positions.nodalPoint - point);
    }

    cv::Point2f Tracker::unproject(Vector point) const {
        return WCStoPixel((point - (1 + positions.cameraEyeProjectionFactor) * positions.nodalPoint)/-positions.cameraEyeProjectionFactor);
    }

    template<typename T> static inline Matrix<1> Scalar(T x) {
        // Convert a scalar to a 0-dimensional tensor
        return {static_cast<Matrix<1>::value_type>(x)};
    }

    EyePosition Tracker::correct(cv::Point2f reflectionPixel, cv::Point2f pupilPixel) {
        // This code should be read in conjunction with Guestrin & Eizenman, pp1125-1126.
        Vector reflectionImage = pixelToWCS(reflectionPixel); // u
        Vector pupilImage = pixelToWCS(pupilPixel); // v
        /* We now need to convert reflectionImage and pupilImage (called u and v by G&E), located on the image sensor,
         * to their counterparts on the cornea (q and r).
         * q lies at an unknown location on the line o-u and r lies on o-v, so there are two scalar unknowns.
         * Temporary solution: assume q and r lie in the same plane and that we know the distance to the user (e.g.
         * from the iris diameter).
         * TODO: Calculate the scale factor below from the iris diameter (either ahead of time or at runtime).
         * Iris diameter, like other eye parameters, varies little between adults - so it should be possible to use it
         * to determine the scale of the image. */
        Vector reflection = project(reflectionImage); // q
        Vector pupil = project(pupilImage); // r
        /* Equation numbering is as in G&E.
         * (3): l, q, o, c are coplanar.
         * (4): angle of incidence = angle of reflection.
         * (2): The corneal reflection lies on the cornea (i.e. at a distance R from its centre of curvature).
         * (intentionally out of order)
         * We use the above to obtain three scalar equations in three scalar unknowns, and thus find c.
         * (7): p, r, o, c are coplanar.
         * (8): Snell's law.
         * (9): p and c lie a distance K apart.
         * p and c are the unknowns in (7-9). Having found c using (2-4), we can now find p. */
        // (3):
        Vector loqo = xt::linalg::cross(positions.light - positions.nodalPoint, reflection - positions.nodalPoint);
        // Now dot(loqo, c) = dot(loqo, o) - a plane on which c must lie.
        // (4):
        Vector lqoq = (positions.light - reflection) * xt::linalg::norm(positions.nodalPoint - reflection);
        Vector oqlq = (positions.nodalPoint - reflection) * xt::linalg::norm(positions.light - reflection);
        Vector oqlqlqoq = oqlq - lqoq;
        // Now dot(oqlqlqoq, c) = dot(oqlqlqoq, q) - another plane containing c.
        // The intersection of these two planes is a line.
        // xt::vstack/xt::hstack don't work for some reason (maybe https://github.com/xtensor-stack/xtensor/issues/2372)
        Matrix<2, 2> squarePlaneMatrix({{loqo(0), loqo(1)}, {oqlqlqoq(0), oqlqlqoq(1)}});
        if (xt::linalg::matrix_rank(squarePlaneMatrix) < 2) {
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
            Matrix<2> b({xt::linalg::dot(loqo, positions.nodalPoint)(), xt::linalg::dot(oqlqlqoq, reflection)()});
            Matrix<2> lastRow({loqo(2), oqlqlqoq(2)});
            Matrix<2> pointA_xy = xt::linalg::solve(squarePlaneMatrix, b); // z = 0
            Matrix<2> pointB_xy = xt::linalg::solve(squarePlaneMatrix, b - lastRow); // z = 1
            Matrix<2> direction_xy = pointB_xy - pointA_xy;
            Vector pointA({pointA_xy(0), pointA_xy(1), 0});
            Vector direction({direction_xy(0), direction_xy(1), 1});
            /* Now we have q, the centre of a sphere of radius R on which c lies (2), and two points, pointA and pointB,
             * defining a line on which c also lies. */
            std::vector<Vector> intersections = lineSphereIntersections(reflection, eye.R, pointA, direction);
            std::optional<Vector> corneaCurvatureCentre; // c
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

            // Now we find p in a somewhat similar way.
            // (7):
            Vector roco = xt::linalg::cross(pupil - positions.nodalPoint, *corneaCurvatureCentre - positions.nodalPoint);
            // Now dot(roco, p) = dot(roco, o) - a plane containing p.
            // (8): n_1 · ‖o - r‖ / ‖(r - c) × (o - r)‖ = ‖p - r‖ / ‖(r - c) × (p - r)‖
            float n1orrcor = eye.n1 * xt::linalg::norm(positions.nodalPoint - pupil)
                             / xt::linalg::norm(xt::linalg::cross(pupil - *corneaCurvatureCentre,
                                                                  positions.nodalPoint - pupil));
            /* This is easier to solve if we extract the angle from the remaining × product:
             * ‖p - r‖ / ‖(r - c) × (p - r)‖ = ‖p - r‖ / (‖r - c‖ · ‖p - r‖ · sin(π+θ))
             * where θ = ∠PRC, the angle between the optic axis of the eye and the
             * normal at the point of refraction of the pupil centre.
             * The ‖p - r‖ term cancels, and we are left with
             * n1orrcor * ‖r - c‖ = 1 / sin(π+θ). */
            float angle = std::asin(-1/(n1orrcor * xt::linalg::norm(pupil - *corneaCurvatureCentre))); // θ
            /* We now have three constraints on p: a plane, the angle ∠PRC, and the sphere of radius K centred on c.
             * It is easy to combine the first two contraints: (7) states that p, r, o and c are coplanar.
             * Furthermore, ∠PRC is known. This allows us to construct a ray from r in the direction of p, which lies
             * in the plane of p, r, o and c.
             * Then, p lies at the intersection of this ray and the sphere.
             * p = r + μw, where w ∝ (p-r)
             * We construct w first.
             * roco is the normal of our plane, it is at 90° to w.
             * c - r is at θ to w. */
            Vector perpendicular = xt::linalg::cross(*corneaCurvatureCentre - pupil, roco);
            perpendicular /= xt::linalg::norm(perpendicular);
            /* w = (c-r)*cos(θ) ± perpendicular*sin(θ)
             * https://math.stackexchange.com/a/2320448
             * This in itself is ambiguous: the w given by this formula can be on either side of r - c, the normal at
             * the point of refraction. However, because of how the cross products are oriented, the positive direction
             * seems to be the right one. */
            direction = (*corneaCurvatureCentre - pupil) * std::cos(angle)
                        / xt::linalg::norm(*corneaCurvatureCentre - pupil)
                        + perpendicular * std::abs(std::sin(angle)); // w
            intersections = lineSphereIntersections(*corneaCurvatureCentre, eye.K, pupil, direction);
            std::optional<Vector> pupilCentre;
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
            Vector eyeCentre = *pupilCentre + eye.D * (*corneaCurvatureCentre - *pupilCentre)
                                              / xt::linalg::norm(*corneaCurvatureCentre - *pupilCentre);
            KF.correct(toMat(*corneaCurvatureCentre));
            return {corneaCurvatureCentre, pupilCentre, eyeCentre};
        }
    }

    EyePosition Tracker::predict() {
        return {{}, {}, toVector(KF.predict())};
    }
}
