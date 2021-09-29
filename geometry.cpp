#include "geometry.hpp"
#include <cmath>
#include <xtensor/xfixed.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xnorm.hpp>
namespace EyeTracker::Geometry {
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

    template<typename T> static inline Matrix<1> Scalar(T x) {
        // Convert a scalar to a 0-dimensional tensor
        return {static_cast<Matrix<1>::value_type>(x)};
    }

    EyePosition eyePosition(cv::Point2i reflectionPixel, cv::Point2i pupilPixel) {
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
        Vector reflection = reflectionImage + 12.79*(o - reflectionImage); // q
        Vector pupil = pupilImage + 12.79*(o - pupilImage); // r
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
        Vector loqo = xt::linalg::cross(light - o, reflection - o);
        float loqoo = xt::linalg::dot(loqo, o)();
        // Now dot(loqo, c) = dot(loqo, o) - a plane on which c must lie.
        // (4):
        Vector lqoq = (light - reflection) * xt::linalg::norm(o - reflection);
        Vector oqlq = (o - reflection) * xt::linalg::norm(light - reflection);
        Vector oqlqlqoq = oqlq - lqoq;
        float oqlqlqoqq = xt::linalg::dot(oqlqlqoq, reflection)();
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
            Matrix<2> b({loqoo, oqlqlqoqq});
            Matrix<2> lastRow({loqo(2), oqlqlqoq(2)});
            Matrix<2> pointA_xy = xt::linalg::solve(squarePlaneMatrix, b); // z = 0
            Matrix<2> pointB_xy = xt::linalg::solve(squarePlaneMatrix, b - lastRow); // z = 1
            Matrix<2> direction_xy = pointB_xy - pointA_xy;
            Vector pointA({pointA_xy(0), pointA_xy(1), 0});
            Vector direction({direction_xy(0), direction_xy(1), 1});
            /* Now we have q, the centre of a sphere of radius R on which c lies (2), and two points, pointA and pointB,
             * defining a line on which c also lies. */
            std::vector<Vector> intersections = lineSphereIntersections(reflection, R, pointA, direction);
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
            Vector roco = xt::linalg::cross(pupil - o, *corneaCurvatureCentre - o);
            float rocoo = xt::linalg::dot(roco, o)();
            // Now dot(roco, p) = dot(roco, o) - a plane containing p.
            // (8): n_1 · ‖o - r‖ / ‖(r - c) × (o - r)‖ = ‖p - r‖ / ‖(r - c) × (p - r)‖
            float n1orrcor = n1 * xt::linalg::norm(o - pupil)
                             / xt::linalg::norm(xt::linalg::cross(pupil - *corneaCurvatureCentre, o - pupil));
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
            intersections = lineSphereIntersections(*corneaCurvatureCentre, K, pupil, direction);
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
            Vector eyeCentre = *pupilCentre + D * (*corneaCurvatureCentre - *pupilCentre)
                                              / xt::linalg::norm(*corneaCurvatureCentre - *pupilCentre);
            return {corneaCurvatureCentre, pupilCentre, eyeCentre};
        }
    }
}
