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
    const CameraProperties CAMERA = {.FPS = 60,
                                     .resolution = {1280, 1024},
                                     .pixelPitch = 0.0048,
                                     .exposureTime = 5, .gain = 200};
    const Positions POSITIONS(27.119, {0, 0, -320}, {0, -50, -320});
    const ImageProperties IMAGE_PROPS = {.ROI = {200, 150, 850, 650},
                                         .pupil = {2, 66, 90}, .iris = {5, 110, 150}, .maxPupilIrisSeparation = 12.24};

    Tracker tracker(EYE, CAMERA, POSITIONS);

    cv::KalmanFilter KF_reflection = tracker.makeICSKalmanFilter();
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

    cv::cuda::GpuMat spots;
    spots.upload(cv::imread("template.png", cv::IMREAD_GRAYSCALE));
    cv::Ptr<cv::cuda::TemplateMatching> spotsMatcher = cv::cuda::createTemplateMatching(CV_8UC1, cv::TM_CCOEFF_NORMED);

    cv::Mat frameBGRCPU;
    cv::cuda::GpuMat frameBGR, frame, correlation;

    cv::Point2f reflection = None, pupil = None, head = None;

    std::chrono::time_point<std::chrono::steady_clock> last_frame_time = std::chrono::steady_clock::now();
    int frameIndex = 0;

    cv::cuda::Stream streamSpots;
    #ifndef HEADLESS
        cv::cuda::Stream streamDisplay;
        std::ostringstream fpsText;
        fpsText << std::fixed << std::setprecision(2);
    #endif

    while (true) {
        if (!video.read(frameBGRCPU) || frameBGRCPU.empty()) break; // Video has ended
        /* Complicated logic here depending on:
         * - whether we are running in headless mode
         * - whether the input is BGR or monochrome
         * - whether we are recording the input to a file. */
        #ifdef HEADLESS
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
            // Keep the rest of the image for display
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

        spotsMatcher->match(frame, spots, correlation, streamSpots);
        std::vector<RatedCircleCentre> pupils = findCircles(frame, IMAGE_PROPS.pupil);
        std::vector<RatedCircleCentre> irises = findCircles(frame, IMAGE_PROPS.iris);

        if (irises.size() > 0) {
            std::vector<RatedCircleCentre>::const_iterator bestIris = std::max_element(irises.cbegin(), irises.cend());
            for (int i = 0; i < 5; ++i) { // Limit number of iterations
                if (pupils.size() == 0) {
                    KF_pupil.correct(toMat(static_cast<cv::Point2f>(IMAGE_PROPS.ROI.tl()) + bestIris->point));
                    break;
                }
                std::vector<RatedCircleCentre>::const_iterator bestPupil = std::max_element(pupils.cbegin(), pupils.cend());
                if (norm(bestIris->point - bestPupil->point) < IMAGE_PROPS.maxPupilIrisSeparation) {
                    // They should be the same point
                    KF_pupil.correct(toMat(static_cast<cv::Point2f>(IMAGE_PROPS.ROI.tl()) + (bestIris->point + bestPupil->point)/2));
                }
                else {
                    /* The detected pupil and iris are not concentric. Probably, this means that the "pupil" is actually
                     * some other reflection. We treat the detection as a false positive.
                     * NB: It's usually the pupil that's the false positive. But if there were problems with false
                     * positives for the iris too, we could remove one or both of the iris and pupil matches. */
                    pupils.erase(bestPupil);
                }
            }
        }

        #ifndef HEADLESS
            cv::cuda::cvtColor(frame, frame, cv::COLOR_GRAY2BGR, 0, streamDisplay);
            cv::cuda::copyMakeBorder(frame, frame,
                                     IMAGE_PROPS.ROI.y,
                                     CAMERA.resolution.height - IMAGE_PROPS.ROI.y - IMAGE_PROPS.ROI.height,
                                     IMAGE_PROPS.ROI.x,
                                     CAMERA.resolution.width - IMAGE_PROPS.ROI.x - IMAGE_PROPS.ROI.width,
                                     cv::BORDER_CONSTANT, 0, streamDisplay);
            cv::cuda::addWeighted(frameBGR, 0.75, frame, 0.25, 0, frameBGR, -1, streamDisplay);
        #endif

        double maxVal = -1;
        cv::Point2i maxLoc = None;
        streamSpots.waitForCompletion();
        cv::cuda::minMaxLoc(correlation, nullptr, &maxVal, nullptr, &maxLoc);

        if (maxLoc.y > 0 and maxLoc.x > 0 and maxVal > IMAGE_PROPS.templateMatchingThreshold) {
            maxLoc += IMAGE_PROPS.ROI.tl();
            KF_reflection.correct((KFMat(2, 1) << maxLoc.x + spots.cols/2, // integer division intentional
                                                  maxLoc.y + spots.rows/2));
        }

        reflection = toPoint(KF_reflection.predict());
        pupil = toPoint(KF_pupil.predict());
        EyePosition eyePos = tracker.correct(reflection, pupil);
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
            if (reflection != None) cv::circle(frameBGRCPU, reflection,  32, cv::Scalar(0x00, 0x00, 0xFF), 5);
            if (pupil != None) cv::circle(frameBGRCPU, pupil, 32, cv::Scalar(0xFF, 0x00, 0x00), 5);
            if (head != None) cv::circle(frameBGRCPU, head, 32, cv::Scalar(0x00, 0xFF, 0x00), 5);
            cv::putText(frameBGRCPU,
                        fpsText.str(),
                        cv::Point2i(100, 100),
                        cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(0x00, 0x00, 0xFF), 3);

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
