#ifndef HEADLESS

#include <opencv2/highgui.hpp>
#include <sstream> // std::ostringstream

#endif

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp> // cv::KalmanFilter
#include <opencv2/videoio.hpp> // cv::VideoCapture
#include <opencv2/videoio/registry.hpp> // cv::videoio_registry
#include <opencv2/videoio/registry.hpp> // cv::videoio_registry
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <algorithm> // std::max_element, std::transform
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <cctype> // std::tolower
#include <cstdlib> // std::strtoul
#include <opencv2/cudafilters.hpp>
#include <thread>
#include "eye_tracking.hpp"
#include "FeatureDetector.h"
#include "EyeTrackerServer.h"

using namespace std::chrono_literals;
using namespace EyeTracking;

const char *windowName = "frame";

enum class Error : int {
    SUCCESS = 0,
    ARGUMENTS,
    FILE_NOT_FOUND,
    UEYE_NOT_FOUND,
    NO_CUDA,
    WRONG_CUDA_INDEX,
    BUILT_WITHOUT_UEYE,
};

int fail(Error e) {
    switch (e) {
        case Error::ARGUMENTS:
            std::cerr << "Error: invoke this program as "
                      << "./headtrack {file|IDS} {video filename|IDS camera index} {CUDA device index}";
            break;
        case Error::FILE_NOT_FOUND:
            std::cerr << "Error: failed to open video file.";
            break;
        case Error::UEYE_NOT_FOUND:
            std::cerr << "Error: failed to open IDS camera.";
            break;
        case Error::NO_CUDA:
            std::cerr << "Error: no CUDA device found.";
            break;
        case Error::WRONG_CUDA_INDEX:
            std::cerr << "Error: no CUDA device with the given index was found.";
            break;
        case Error::BUILT_WITHOUT_UEYE:
            std::cerr << "Error: use of IDS camera requested, but OpenCV was compiled without uEye support.";
            break;
        default:
            std::cerr << "Unknown error.";
            break;
    }
    std::cerr << std::endl;
    return static_cast<int>(e);
}

void toLowercase(std::string &text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return std::tolower(c); });
}

int main(int argc, char *argv[]) {
    // Use a pointer to abstract away the type of VideoCapture (either a video file or an IDS camera)
    cv::VideoCapture video;
    cv::VideoWriter vwInput, vwOutput;
    const int FOURCC = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    bool isRealtime = false;
    bool headless = false;
    bool thresholdEnabled = false;
    bool pupilFound, reflection1Found, reflection2Found;

    const int FRAMES_FOR_FPS_MEASUREMENT = 8;

    const EyeProperties EYE; // Use default (average) values
    /* Camera model: "UI-3140CP-M-GL Rev.2 (AB00613)"
     * The maximum value of the gain is 400. The maximum introduces a lot of noise; it can be mitigated somewhat
     * by subtracting the regular banding pattern that appears and applying a median filter. */
    const CameraProperties CAMERA = {.FPS = 30,
            .resolution = {1280, 1024},
            .pixelPitch = 0.0048,
            .exposureTime = 10, .gamma = 150};

    Vec3d light1 = {5, 0, -50};
    Vec3d light2 = {-5, 0, -50};

    const Positions POSITIONS(27.119, {0, 0, -370}, light1, light2);
    const ImageProperties IMAGE_PROPS = {.ROI = {450, 400, 400, 200},
            .pupil = {10, 20, 90, 11, 11}};

    Tracker tracker(EYE, CAMERA, POSITIONS);

    cv::KalmanFilter KF_reflection1 = tracker.makeICSKalmanFilter();
    cv::KalmanFilter KF_reflection2 = tracker.makeICSKalmanFilter();
    cv::KalmanFilter KF_pupil = tracker.makeICSKalmanFilter();

    // Process arguments
    if (argc < 2) return fail(Error::ARGUMENTS);
    else {
        std::string mode(argv[1]);
        toLowercase(mode);

        if (mode == "file") {
            if (argc < 3) return fail(Error::ARGUMENTS); // No file path specified
            else {
                video.open(argv[2]);
                if (!video.isOpened()) return fail(Error::FILE_NOT_FOUND);
            }
        } else if (mode == "ids" || mode == "ueye") {
            if (cv::videoio_registry::hasBackend(cv::CAP_UEYE)) {
                unsigned long cameraIndex = argc >= 3 ? std::strtoul(argv[2], nullptr, 10) : 0;
                video.open(cameraIndex, cv::CAP_UEYE);
                if (!video.isOpened()) return fail(Error::UEYE_NOT_FOUND);
                video.set(cv::CAP_PROP_EXPOSURE, CAMERA.exposureTime);
                video.set(cv::CAP_PROP_GAIN, 400);
                /* NB: According to the uEye API documentation, setting the FPS may change the exposure time too
                 * (presumably, if it is too long, it is decreased to the maximum achievable with the given framerate). */
                //video.set(cv::CAP_PROP_FPS, CAMERA.FPS);
                isRealtime = true;
            } else return fail(Error::BUILT_WITHOUT_UEYE);
        } else return fail(Error::ARGUMENTS);
    }

    if (argc > 3) {
        std::string mode(argv[3]);
        std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c) { return std::tolower(c); });
        headless = (mode == "headless");
    } else {
        headless = false;
    }

    cv::cuda::GpuMat fullFrame, croppedFrame, thresholded, correlation;
    cv::Mat fullFrameCPU;

    int nCUDADevices = cv::cuda::getCudaEnabledDeviceCount();
    if (nCUDADevices <= 0) fail(Error::NO_CUDA);
    else if (argc > 4) { // Use a non-default CUDA device
        unsigned long CUDAIndex = argc > 4 ? std::strtoul(argv[4], nullptr, 10) : 0;
        if (CUDAIndex < nCUDADevices) cv::cuda::setDevice(CUDAIndex);
        else return fail(Error::WRONG_CUDA_INDEX);
    }

    
    FeatureDetector featureDetector(IMAGE_PROPS, "template.png");

    cv::Point2f reflection1 = None, reflection2 = None, pupil = None, head;

    std::chrono::time_point<std::chrono::steady_clock> last_frame_time = std::chrono::steady_clock::now();
    int frameIndex = 0;
    cv::cuda::Stream streamDisplay;
    std::ostringstream fpsText;

    if (!headless) {
        fpsText << std::fixed << std::setprecision(2);
    }

    cv::theRNG().state = time(nullptr);
                        
    EyeTrackerServer eyeTrackerServer(&tracker);
    eyeTrackerServer.startServer();

    std::thread t(&EyeTrackerServer::openSocket, &eyeTrackerServer);

    while (!eyeTrackerServer.finished) {
        if (!video.read(fullFrameCPU) || fullFrameCPU.empty()) {
            if (!isRealtime) {
                video.set(cv::CAP_PROP_POS_FRAMES, 0);
                continue;
            } else {
                break;
            }
        }

        pupilFound = reflection1Found = reflection2Found = false;

        /* Complicated logic here depending on:
         * - whether we are running in headless mode
         * - whether the input is BGR or monochrome
         * - whether we are recording the input to a file. */
        if (headless) {
            if (vwInput.isOpened()) {
                featureDetector.extractFrameAndWrite(fullFrameCPU, vwInput, croppedFrame, fullFrame);
            } else {
                featureDetector.extractFrame(fullFrameCPU, croppedFrame, fullFrame);
            }
        } else {
            featureDetector.createBorder(fullFrameCPU, CAMERA.resolution);
            fullFrame.upload(fullFrameCPU);
            if (vwInput.isOpened()) {
                featureDetector.extractFrameAndWrite(fullFrameCPU, vwInput, croppedFrame, fullFrame);
            } else {
                featureDetector.extractFrame(fullFrameCPU, croppedFrame, fullFrame);
            }
        }


        std::vector<RatedCircleCentre> pupils = findCircles(croppedFrame, IMAGE_PROPS.pupil, thresholded);
        int pupilRadius = 32;
        int glintSize = 16;
        cv::Point maxLoc1, maxLoc2;

        pupilFound = pupils.size() > 0;
        double maxVal;
        cv::Point templateLoc[2];

        reflection1Found = featureDetector.findReflection(croppedFrame, templateLoc[0]);
        featureDetector.setMapToVertical(templateLoc[0]);
        reflection2Found = featureDetector.findReflection(croppedFrame, templateLoc[1]);
        featureDetector.setMapToDistance(templateLoc[0], templateLoc[1]);

        maxLoc1 = templateLoc[0];
        maxLoc2 = templateLoc[1];

        if (maxLoc1.y > maxLoc2.y) {
            swap(maxLoc1, maxLoc2);
        }

        if (pupilFound and reflection1Found and reflection2Found) {
            KF_reflection1.correct((KFMat(2, 1) << maxLoc1.x, maxLoc1.y));
            reflection1 = toPoint(KF_reflection1.predict());
            KF_reflection2.correct((KFMat(2, 1) << maxLoc2.x, maxLoc2.y));
            reflection2 = toPoint(KF_reflection2.predict());
            std::vector<RatedCircleCentre>::const_iterator bestPupil = std::max_element(pupils.cbegin(), pupils.cend());
            KF_pupil.correct(toMat(static_cast<cv::Point2f>(IMAGE_PROPS.ROI.tl()) + (bestPupil->point)));
            pupil = toPoint(KF_pupil.predict());
            pupilRadius = (int) bestPupil->radius;
        } 

        EyePosition eyePos = tracker.correct(reflection1, reflection2, pupil, pupilRadius * 2.0f);

        if (headless) {
            /*std::cout << (*eyePos.eyeCentre)(0)
                      << ", " << (*eyePos.eyeCentre)(1)
                      << ", " << (*eyePos.eyeCentre)(2) << "\n";*/
        } else {
            head = tracker.unproject(*eyePos.eyeCentre);
            if (++frameIndex == FRAMES_FOR_FPS_MEASUREMENT) {
                const std::chrono::duration<float> frame_time = std::chrono::steady_clock::now() - last_frame_time;
                fpsText.str(""); // Clear contents of fpsText
                fpsText << 1s / (frame_time / FRAMES_FOR_FPS_MEASUREMENT);
                frameIndex = 0;
                last_frame_time = std::chrono::steady_clock::now();
            }

            cv::cuda::cvtColor(thresholded, thresholded, cv::COLOR_GRAY2BGR, 0, streamDisplay);
            cv::cuda::cvtColor(fullFrame, fullFrame, cv::COLOR_GRAY2BGR);

            cv::cuda::copyMakeBorder(thresholded, thresholded,
                                     IMAGE_PROPS.ROI.y,
                                     CAMERA.resolution.height - IMAGE_PROPS.ROI.y - IMAGE_PROPS.ROI.height,
                                     IMAGE_PROPS.ROI.x,
                                     CAMERA.resolution.width - IMAGE_PROPS.ROI.x - IMAGE_PROPS.ROI.width,
                                     cv::BORDER_CONSTANT, 0, streamDisplay);

            cv::cuda::addWeighted(fullFrame, 1.0f - (float) thresholdEnabled / 2, thresholded, (float) thresholdEnabled / 2, 0,
                                  fullFrame, -1, streamDisplay);


            streamDisplay.waitForCompletion();
            fullFrame.download(fullFrameCPU);

            if (reflection1 != None)
                cv::circle(fullFrameCPU, reflection1, glintSize / 2, cv::Scalar(0x00, 0x00, 0xFF), 2);
            if (reflection2 != None)
                cv::circle(fullFrameCPU, reflection2, glintSize / 2, cv::Scalar(0x00, 0x00, 0xFF), 2);
            cv::circle(fullFrameCPU, pupil, pupilRadius, cv::Scalar(0xFF, 0x00, 0x00), 5);
            cv::putText(fullFrameCPU,
                        fpsText.str(),
                        cv::Point2i(100, 100),
                        cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(0x00, 0x00, 0xFF), 3);

            cv::imshow(windowName, fullFrameCPU);
            if (vwOutput.isOpened()) vwOutput.write(fullFrameCPU);

            bool quitting = false;
            switch (cv::waitKey(1) & 0xFF) {
                case 27: // Esc
                case 'q':
                    quitting = true;
                    break;
                case 's':
                    imwrite("fullFrame.png", fullFrameCPU);
                    break;
                case 'v': // Record input video; only if the input is a live feed
                    if (isRealtime)
                        vwInput.open("recorded_input.mp4", FOURCC, CAMERA.FPS,
                                     {CAMERA.resolution.width, CAMERA.resolution.height}, false);
                    break;
                case 'w': // Record output video
                    vwOutput.open("recorded_output.mp4", FOURCC, CAMERA.FPS,
                                  {CAMERA.resolution.width, CAMERA.resolution.height}, true);
                    break;
                case 't':
                    thresholdEnabled = !thresholdEnabled;
                    break;
            }

            if (quitting || cv::getWindowProperty(windowName, cv::WND_PROP_AUTOSIZE) == -1)
                break; // Window closed by user
        }
    }
    //t.join();
    video.release();

    if (vwOutput.isOpened()) vwOutput.release();

    eyeTrackerServer.closeSocket();


    if (!headless) {
        cv::destroyAllWindows();
    }
    return 0;
}
