#pragma once
namespace EyeTracker::Eye { // Eye parameters
    /* From Guestrin & Eizenman
     * They provide a calibration procedure, but in fact there is not much variation in these parameters. */
    constexpr float R = 7.8; // mm, radius of corneal curvature
    constexpr float K = 4.2; // mm, distance between pupil centre and centre of corneal curvature
    constexpr float n1 = 1.3375; // Standard Keratometric Index, refractive index of cornea and aqueous humour
    // From Bekerman, Gottlieb & Vaiman
    constexpr float D = 10; // mm, distance between pupil centre and centre of eye rotation
}
