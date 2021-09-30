#pragma once
#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp> // cv::KalmanFilter
#include <cstdint> // uint8_t
#include "types.hpp"
namespace EyeTracker {
    namespace Params {
        namespace Eye {
            /* From Guestrin & Eizenman
             * They provide a calibration procedure, but in fact there is not much variation in these parameters. */
            constexpr float R = 7.8; // mm, radius of corneal curvature
            constexpr float K = 4.2; // mm, distance between pupil centre and centre of corneal curvature
            constexpr float n1 = 1.3375; // Standard Keratometric Index, refractive index of cornea and aqueous humour
            // From Bekerman, Gottlieb & Vaiman
            constexpr float D = 10; // mm, distance between pupil centre and centre of eye rotation
            namespace Pupil {
                constexpr uint8_t THRESHOLD = 2;
                constexpr float MIN_RADIUS = 66; // px
                constexpr float MAX_RADIUS = 90; // px
            }
            namespace Iris {
                constexpr uint8_t THRESHOLD = 5;
                constexpr float MIN_RADIUS = 110; // px
                constexpr float MAX_RADIUS = 150; // px
            }
            constexpr float MAX_PUPIL_IRIS_SEPARATION = 12.24; // mm
            constexpr float TEMPLATE_MATCHING_THRESHOLD = 0.9;
        }

        namespace Camera {
            constexpr float FPS = 60; // Hz
            constexpr float DT = 1/FPS; // s
            constexpr int FRAMES_FOR_FPS_MEASUREMENT = 8;
            constexpr int RESOLUTION_X = 1280; // px
            constexpr int RESOLUTION_Y = 1024; // px
            constexpr float SENSOR_X = 6.144; // mm
            constexpr float SENSOR_Y = 4.915; // mm
            constexpr float PIXEL_PITCH = 0.0048; // mm
            const cv::Rect ROI(200, 150, 850, 650);
            // Camera model: "UI-3140CP-M-GL Rev.2 (AB00613)"
            constexpr double EXPOSURE_TIME = 5; // unknown units (ms?)
            /* The maximum value of the gain is 400. The maximum introduces a lot of noise; it can be mitigated
             * by subtracting the regular banding pattern that appears and applying a median filter. */
            constexpr double GAIN = 200;
        }

        inline cv::KalmanFilter makePixelKalmanFilter() {
            constexpr static float VELOCITY_DECAY = 0.9;
            const static cv::Mat TRANSITION_MATRIX  = (KFMat(4, 4) << 1, 0, Camera::DT, 0,
                                                                      0, 1, 0, Camera::DT,
                                                                      0, 0, VELOCITY_DECAY, 0,
                                                                      0, 0, 0, VELOCITY_DECAY);
            const static cv::Mat MEASUREMENT_MATRIX = (KFMat(2, 4) << 1, 0, 0, 0,
                                                                      0, 1, 0, 0);
            const static cv::Mat PROCESS_NOISE_COV = cv::Mat::eye(4, 4, CV_32F) * 100;
            const static cv::Mat MEASUREMENT_NOISE_COV = cv::Mat::eye(2, 2, CV_32F) * 50;
            const static cv::Mat ERROR_COV_POST = cv::Mat::eye(4, 4, CV_32F) * 0.1;
            const static cv::Mat STATE_POST = (KFMat(4, 1) << Camera::RESOLUTION_X/2.0, Camera::RESOLUTION_Y/2.0, 0, 0);

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

        inline cv::KalmanFilter make3DKalmanFilter() {
            constexpr static float ACCELERATION_DECAY = 0.9;
            // TODO: Use a better numerical integrator?
            const static cv::Mat TRANSITION_MATRIX  = (KFMat(9, 9) << 1, 0, 0, Camera::DT, 0, 0, 0, 0, 0,
                                                                      0, 1, 0, 0, Camera::DT, 0, 0, 0, 0,
                                                                      0, 0, 1, 0, 0, Camera::DT, 0, 0, 0,
                                                                      0, 0, 0, 1, 0, 0, Camera::DT, 0, 0,
                                                                      0, 0, 0, 0, 1, 0, 0, Camera::DT, 0,
                                                                      0, 0, 0, 0, 0, 1, 0, 0, Camera::DT,
                                                                      0, 0, 0, 0, 0, 0, ACCELERATION_DECAY, 0, 0,
                                                                      0, 0, 0, 0, 0, 0, 0, ACCELERATION_DECAY, 0,
                                                                      0, 0, 0, 0, 0, 0, 0, 0, ACCELERATION_DECAY);
            const static cv::Mat MEASUREMENT_MATRIX = cv::Mat::eye(3, 9, CV_32F);
            const static cv::Mat PROCESS_NOISE_COV = cv::Mat::eye(9, 9, CV_32F) * 100;
            const static cv::Mat MEASUREMENT_NOISE_COV = cv::Mat::eye(3, 3, CV_32F) * 50;
            const static cv::Mat ERROR_COV_POST = cv::Mat::eye(9, 9, CV_32F) * 0.1;
            const static cv::Mat STATE_POST = cv::Mat::zeros(9, 1, CV_32F);

            cv::KalmanFilter KF(9, 3);
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

        namespace Positions {
            /* One-letter variable names are as defined by Guestrin & Eizenman, unless defined in comments.
             * All constants are in millimetres unless otherwise indicated. */
            constexpr float LAMBDA = 27.119;
            const Vector nodalPoint({0, 0, -320}); // o; nodal point of the camera
            const Matrix<3, 3> rotation({{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}); // dimensionless
            const Vector light = nodalPoint + Vector({0, -50, 0}); // l
            const float CAMERA_EYE_DISTANCE = -nodalPoint(2);
        }
    }
}
