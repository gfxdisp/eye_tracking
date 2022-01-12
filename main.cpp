#ifndef HEADLESS
    #include <opencv2/highgui.hpp>
    #include <sstream> // std::ostringstream
#endif
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp> // cv::KalmanFilter
#include <opencv2/videoio.hpp> // cv::VideoCapture
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
#include "eye_tracking.hpp"
using namespace std::chrono_literals;
using namespace EyeTracking;

#ifndef HEADLESS
    const char* windowName = "frame";
#endif

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

int main(int argc, char* argv[]) {
    // Use a pointer to abstract away the type of VideoCapture (either a video file or an IDS camera)
    cv::VideoCapture video;
    cv::VideoWriter vwInput, vwOutput;
    const int FOURCC = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    bool isRealtime = false;

    const int FRAMES_FOR_FPS_MEASUREMENT = 8;

    const EyeProperties EYE; // Use default (average) values
    /* Camera model: "UI-3140CP-M-GL Rev.2 (AB00613)"
     * The maximum value of the gain is 400. The maximum introduces a lot of noise; it can be mitigated somewhat
     * by subtracting the regular banding pattern that appears and applying a median filter. */
    const CameraProperties CAMERA = {.FPS = 30,
                                     .resolution = {1280, 1024},
                                     .pixelPitch = 0.0048,
                                     .exposureTime = 5, .gain = 200};

    const Positions POSITIONS(27.119, {0, 0, -327});
    const ImageProperties IMAGE_PROPS = {.ROI = {350, 500, 300, 300},
    //Dmitry
//  .pupil = {2, 66, 90}, .iris = {5, 110, 150}, .maxPupilIrisSeparation = 12.24};
    //Radek
//  .pupil = {30, 40, 80}, .iris = {90, 100, 300}, .maxPupilIrisSeparation = 200.24};
    //Blender
    .pupil = {26, 50, 90}};

    Tracker tracker(EYE, CAMERA, POSITIONS);

    Vec3d light1 = {27.98, 0, argc > 3 ? (double)std::stof(argv[3]) : -327};
    Vec3d light2 = {-27.98, 0, argc > 3 ? (double)std::stof(argv[3]) : -327};

    cv::KalmanFilter KF_reflection1 = tracker.makeICSKalmanFilter();
    cv::KalmanFilter KF_reflection2 = tracker.makeICSKalmanFilter();
    cv::KalmanFilter KF_pupil      = tracker.makeICSKalmanFilter();

    // Process arguments
    if (argc < 2) return fail(Error::ARGUMENTS);
    else {
        std::string mode(argv[1]);
        // Convert to lowercase (for user's convenience)
        std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c){ return std::tolower(c); });
        if (mode == "file") {
            if (argc < 3) return fail(Error::ARGUMENTS); // No file path specified
            else {
                video.open(argv[2]);
                if (!video.isOpened()) return fail(Error::FILE_NOT_FOUND);
            }
        }
        else if (mode == "ids" || mode == "ueye") {
            if (cv::videoio_registry::hasBackend(cv::CAP_UEYE)) {
                unsigned long cameraIndex = argc >= 3 ? std::strtoul(argv[2], nullptr, 10) : 0;
                video.open(cameraIndex, cv::CAP_UEYE);
                if (!video.isOpened()) return fail(Error::UEYE_NOT_FOUND);
                video.set(cv::CAP_PROP_EXPOSURE, CAMERA.exposureTime);
                /* NB: According to the uEye API documentation, setting the FPS may change the exposure time too
                 * (presumably, if it is too long, it is decreased to the maximum achievable with the given framerate). */
                video.set(cv::CAP_PROP_FPS, CAMERA.FPS);
                video.set(cv::CAP_PROP_GAIN, CAMERA.gain);
                isRealtime = true;
            }
            else return fail(Error::BUILT_WITHOUT_UEYE);
        }
        else return fail(Error::ARGUMENTS);
    }

    int nCUDADevices = cv::cuda::getCudaEnabledDeviceCount();
    if (nCUDADevices <= 0) fail(Error::NO_CUDA);
    else if (argc >= 4) { // Use a non-default CUDA device
        unsigned long CUDAIndex = argc >= 3 ? std::strtoul(argv[2], nullptr, 10) : 0;
        if (CUDAIndex < nCUDADevices) cv::cuda::setDevice(CUDAIndex);
        else return fail(Error::WRONG_CUDA_INDEX);
    }

    cv::Mat frameBGRCPU, frameCPU, correlationCPU;
    cv::cuda::GpuMat frameBGR, frame, correlation1, correlation2;
//    cv::Mat maskCPU;

    cv::Point2f reflection1 = None, reflection2 = None, pupil = None, head = None;

    std::chrono::time_point<std::chrono::steady_clock> last_frame_time = std::chrono::steady_clock::now();
    int frameIndex = 0;

    #ifndef HEADLESS
        cv::cuda::Stream streamDisplay;
        std::ostringstream fpsText;
        fpsText << std::fixed << std::setprecision(2);
    #endif
    int border_v = 0, border_h = 0;
    bool first = true;

    cv::theRNG().state = time(nullptr);

    float sigma = 0.0f;

    if (argc > 4) {
        sigma = (float)(std::stoi(argv[4]));
    }

    while (true) {
        if (!video.read(frameBGRCPU) || frameBGRCPU.empty()) break; // Video has ended

        /* Complicated logic here depending on:
         * - whether we are running in headless mode
         * - whether the input is BGR or monochrome
         * - whether we are recording the input to a file. */
        #ifdef HEADLESS
            first = false;
            if (vwInput.isOpened()) {
                // Need to save the frame before cropping to the ROI
                frameBGR.upload(frameBGRCPU);
                if (frameBGRCPU.type() == CV_8UC3) cv::cuda::cvtColor(frameBGR, frame, cv::COLOR_BGR2GRAY);
                else frame = frameBGR;
                frame.download(frameBGRCPU);
                vwInput.write(frameBGRCPU);
                frame = frame(IMAGE_PROPS.ROI);
            }
            else {
                // Crop the ROI immediately, we won't need the rest of the image again
                frameBGR.upload(frameBGRCPU(IMAGE_PROPS.ROI));
                // frameBGR might not actually be BGR if it's loaded from a greyscale video
                if (frameBGRCPU.type() == CV_8UC3) cv::cuda::cvtColor(frameBGR, frame, cv::COLOR_BGR2GRAY);
                else frame = frameBGR;
            }
        #else
            if (first) {
                first = false;
                float a = CAMERA.resolution.height;
                float b = CAMERA.resolution.width;
                float c = frameBGRCPU.rows;
                float d = frameBGRCPU.cols;
                if (a / b >= c / d) {
                    border_v = (int) ((((a / b) * d) - c) / 2);
                } else {
                    border_h = (int) ((((a / b) * c) - d) / 2);
                }
            }
            cv::copyMakeBorder(frameBGRCPU, frameBGRCPU, border_v, border_v, border_h, border_h, cv::BORDER_CONSTANT, 0);
            // Keep the rest of the image for display
            cv::resize(frameBGRCPU, frameBGRCPU, CAMERA.resolution);
            frameBGR.upload(frameBGRCPU);
            if (frameBGRCPU.type() == CV_8UC3) {
                /* If we are recording the input, we need to convert it all to greyscale.
                 * Otherwise, we only need to convert the ROI. */
                if (vwInput.isOpened()) {
                    cv::cuda::cvtColor(frameBGR, frame, cv::COLOR_BGR2GRAY);
                    frame.download(frameBGRCPU);
                    vwInput.write(frameBGRCPU);
                    frame = frame(IMAGE_PROPS.ROI);
                }
                else cv::cuda::cvtColor(frameBGR(IMAGE_PROPS.ROI), frame, cv::COLOR_BGR2GRAY);
            }
            else {
                if (vwInput.isOpened()) vwInput.write(frameBGRCPU);
                frame = frameBGR(IMAGE_PROPS.ROI);
            }
        #endif

        std::vector<RatedCircleCentre> pupils = findCircles(frame, IMAGE_PROPS.pupil);
        int totalPoints = 7;
        int pupilRadius = 32;
        int glintSize = 16;

        if (pupils.size() > 0) {
            std::vector<RatedCircleCentre>::const_iterator bestPupil = std::max_element(pupils.cbegin(), pupils.cend());
            cv::Point2f correctedPoint(bestPupil->point.x + cv::theRNG().gaussian(sigma), bestPupil->point.y + cv::theRNG().gaussian(sigma));
            KF_pupil.correct(toMat(static_cast<cv::Point2f>(IMAGE_PROPS.ROI.tl()) + (correctedPoint)));
            pupil = toPoint(KF_pupil.predict());
            pupilRadius = (int)bestPupil->radius;
        }

//        cv::cuda::threshold(frame, frame, 150, 255, cv::THRESH_TOZERO_INV);

        cv::Point maxLoc[totalPoints];
        cv::Point glints[totalPoints];
        float mean[2], stddev[2];

        mean[0] = 0.0f;
        mean[1] = 0.0f;
        stddev[0] = 0.0f;
        stddev[1] = 0.0f;
        double maxVal;


        for (int i = 0; i < totalPoints; i++) {
            cv::cuda::minMaxLoc(frame, nullptr, &maxVal, nullptr, &maxLoc[i]);
            int x = std::clamp(maxLoc[i].x - glintSize / 2, 0, IMAGE_PROPS.ROI.width);
            int y = std::clamp(maxLoc[i].y - glintSize / 2, 0, IMAGE_PROPS.ROI.height);
            int width = std::clamp(x + glintSize, 0, IMAGE_PROPS.ROI.width) - x;
            int height = std::clamp(y + glintSize, 0, IMAGE_PROPS.ROI.height) - y;
            cv::Rect glint = cv::Rect(x, y, width, height);
            frame(glint).setTo(0);
            maxLoc[i] += IMAGE_PROPS.ROI.tl();
            mean[0] += (float)maxLoc[i].x;
            mean[1] += (float)maxLoc[i].y;
        }
        mean[0] /= totalPoints;
        mean[1] /= totalPoints;

        for (int i = 0; i < totalPoints; i++) {
            stddev[0] += (mean[0] - (float)maxLoc[i].x) * (mean[0] - (float)maxLoc[i].x);
            stddev[1] += (mean[1] - (float)maxLoc[i].y) * (mean[1] - (float)maxLoc[i].y);
        }

        stddev[0] = sqrt(stddev[0] / totalPoints);
        stddev[1] = sqrt(stddev[1] / totalPoints);

        int k = 0;
        for (int i = 0; i < totalPoints; i++) {
            if (abs(mean[0] - (float)maxLoc[i].x) < 1.5 * stddev[0] && abs(mean[1] - (float)maxLoc[i].y) < 1.5 * stddev[1]) {
                glints[k] = maxLoc[i];
                k++;
            }
        }

        totalPoints = k;
        mean[0] = 0.0f;
        mean[1] = 0.0f;
        for (int i = 0; i < totalPoints; i++) {
            mean[0] += (float)glints[i].x;
            mean[1] += (float)glints[i].y;
        }
        mean[0] /= totalPoints;
        mean[1] /= totalPoints;

        cv::Point maxLoc1, maxLoc2;
        float best_distance = 999999;

        for (int i = 0; i < totalPoints; i++) {
            float distance = (mean[0] - (float) glints[i].x) * (mean[0] - (float) glints[i].x);
            if (distance < best_distance) {
                maxLoc1 = glints[i];
                best_distance = distance;
                k = i;
            }
        }

        best_distance = 999999;
        for (int i = 0; i < totalPoints; i++) {
            if (i == k) {
                continue;
            }
            float distance = (maxLoc1.x - (float) glints[i].x) * (maxLoc1.x - (float) glints[i].x);
            if (distance < best_distance) {
                maxLoc2 = glints[i];
                best_distance = distance;
            }
        }


        if (maxLoc1.y > maxLoc2.y) {
            swap(maxLoc1, maxLoc2);
        }

        KF_reflection1.correct((KFMat(2, 1) << maxLoc1.x, maxLoc1.y));
        reflection1 = toPoint(KF_reflection1.predict());
        KF_reflection2.correct((KFMat(2, 1) << maxLoc2.x, maxLoc2.y));
        reflection2 = toPoint(KF_reflection2.predict());

        #ifndef HEADLESS
            cv::cuda::cvtColor(frame, frame, cv::COLOR_GRAY2BGR, 0, streamDisplay);
            cv::cuda::copyMakeBorder(frame, frame,
                                     IMAGE_PROPS.ROI.y,
                                     CAMERA.resolution.height - IMAGE_PROPS.ROI.y - IMAGE_PROPS.ROI.height,
                                     IMAGE_PROPS.ROI.x,
                                     CAMERA.resolution.width - IMAGE_PROPS.ROI.x - IMAGE_PROPS.ROI.width,
                                     cv::BORDER_CONSTANT, 0, streamDisplay);
//            cv::cuda::addWeighted(frameBGR, 0.75, frame, 0.25, 0, frameBGR, -1, streamDisplay);
        #endif

        EyePosition eyePos;

        eyePos = tracker.correct2(reflection1, reflection2, pupil, light1, light2);

        head = tracker.unproject(*eyePos.eyeCentre);

        #ifdef HEADLESS
            std::cout << '{' <<  (*eyePos.eyeCentre)(0)
                      << ", " << (*eyePos.eyeCentre)(1)
                      << ", " << (*eyePos.eyeCentre)(2) << "}\n";
        #else
            if (++frameIndex == FRAMES_FOR_FPS_MEASUREMENT) {
                const std::chrono::duration<float> frame_time = std::chrono::steady_clock::now() - last_frame_time;
                fpsText.str(""); // Clear contents of fpsText
                fpsText << 1s/(frame_time/FRAMES_FOR_FPS_MEASUREMENT);
                frameIndex = 0;
                last_frame_time = std::chrono::steady_clock::now();
            }

            streamDisplay.waitForCompletion();
            /* We could avoid having to download the frame from the GPU by using OpenGL.
             * This requires OpenCV to be build with OpenGL support, and the line
             * namedWindow(windowName, WINDOW_AUTOSIZE | WINDOW_OPENGL)
             * to be run before the start of the main loop.
             * Then, cv::imshow can be called directly on cv::cuda::GpuMat.
             * However, instead of using cv::circle and cv::putText, we would have to use OpenGL functions directly.
             * This would be a lot of effort for a very marginal performance benefit. */
            frameBGR.download(frameBGRCPU);

//            for (int i = 0; i < totalPoints; i++) {
//                cv::circle(frameBGRCPU, maxLoc[i],  glintSize/2, cv::Scalar(0x00, 0xFF, 0xFF), 2);
//            }

            if (reflection1 != None) cv::circle(frameBGRCPU, reflection1,  glintSize/2, cv::Scalar(0x00, 0x00, 0xFF), 2);
            if (reflection2 != None) cv::circle(frameBGRCPU, reflection2,  glintSize/2, cv::Scalar(0x00, 0x00, 0xFF), 2);
            if (pupil != None) cv::circle(frameBGRCPU, pupil, pupilRadius, cv::Scalar(0xFF, 0x00, 0x00), 5);
//            if (head != None) cv::circle(frameBGRCPU, head, 2, cv::Scalar(0x00, 0xFF, 0x00), 5);
            cv::putText(frameBGRCPU,
                        fpsText.str(),
                        cv::Point2i(100, 100),
                        cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(0x00, 0x00, 0xFF), 3);

//            cv::cuda::threshold(frameBGRCPU, frameBGRCPU, IMAGE_PROPS.pupil.threshold, 255, cv::THRESH_BINARY_INV);
            cv::imshow(windowName, frameBGRCPU);
            if (vwOutput.isOpened()) vwOutput.write(frameBGRCPU);

            bool quitting = false;
            switch (cv::waitKey(1) & 0xFF) {
            case 27: // Esc
            case 'q':
                quitting = true;
                break;
            case 's':
                imwrite("frame.png", frameBGRCPU);
                break;
            case 'v': // Record input video; only if the input is a live feed
                if (isRealtime) vwInput.open("recorded_input.mp4", FOURCC, CAMERA.FPS, {CAMERA.resolution.width, CAMERA.resolution.height}, false);
                break;
            case 'w': // Record output video
                vwOutput.open("recorded_output.mp4", FOURCC, CAMERA.FPS, {CAMERA.resolution.width, CAMERA.resolution.height}, true);
                break;
            }

            if (quitting || cv::getWindowProperty(windowName, cv::WND_PROP_AUTOSIZE) == -1) break; // Window closed by user
        #endif
    }

    video.release();
    #ifndef HEADLESS
        cv::destroyAllWindows();
    #endif
    return 0;
}
