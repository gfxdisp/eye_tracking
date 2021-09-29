#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp> // cv::KalmanFilter
#include <opencv2/videoio.hpp> // cv::VideoCapture
#include <opencv2/videoio/registry.hpp> // cv::videio_registry
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <algorithm> // std::min_element, std::transform
#include <chrono>
#include <iostream>
#include <sstream> // std::ostringstream
#include <string>
#include <vector>
#include <cctype> // std::tolower
#include <cstdlib> // std::strtoul
#include "geometry.hpp"
#include "image_processing.hpp"
using namespace cv;
using namespace std::chrono_literals;
using namespace EyeTracker;
using namespace ImageProcessing;
using namespace Params::Camera;
using namespace Params::Eye;
using Params::makePixelKalmanFilter;

namespace EyeTracker {
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
}

int main(int argc, char* argv[]) {
    // Use a pointer to abstract away the type of VideoCapture (either a video file or an IDS camera)
    VideoCapture video;
    VideoWriter vwInput, vwOutput;
    const int fourcc = VideoWriter::fourcc('m', 'p', '4', 'v');
    bool isRealtime = false;

    // Process arguments
    if (argc < 2) return fail(EyeTracker::Error::ARGUMENTS);
    else {
        std::string mode(argv[1]);
        // Convert to lowercase (for user's convenience)
        std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c){ return std::tolower(c); });
        if (mode == "file") {
            if (argc < 3) return fail(EyeTracker::Error::ARGUMENTS); // No file path specified
            else {
                video.open(argv[2]);
                if (!video.isOpened()) return fail(EyeTracker::Error::FILE_NOT_FOUND);
            }
        }
        else if (mode == "ids" || mode == "ueye") {
            if (videoio_registry::hasBackend(CAP_UEYE)) {
                unsigned long cameraIndex = argc >= 3 ? std::strtoul(argv[2], nullptr, 10) : 0;
                video.open(cameraIndex, CAP_UEYE);
                if (!video.isOpened()) return fail(EyeTracker::Error::UEYE_NOT_FOUND);
                isRealtime = true;
            }
            else return fail(EyeTracker::Error::BUILT_WITHOUT_UEYE);
        }
        else return fail(EyeTracker::Error::ARGUMENTS);
    }

    int nCUDADevices = cuda::getCudaEnabledDeviceCount();
    if (nCUDADevices <= 0) fail(EyeTracker::Error::NO_CUDA);
    else if (argc >= 4) { // Use a non-default CUDA device
        unsigned long CUDAIndex = argc >= 3 ? std::strtoul(argv[2], nullptr, 10) : 0;
        if (CUDAIndex < nCUDADevices) cuda::setDevice(CUDAIndex);
        else return fail(EyeTracker::Error::WRONG_CUDA_INDEX);
    }

    cuda::GpuMat spots;
    spots.upload(imread("double_patch_orig.png", IMREAD_GRAYSCALE));
    // The following line is only necessary if `spots' was captured with a different gamma setting from `video'.
    correct(spots, spots, 1, 0, 2);
    Ptr<cuda::TemplateMatching> spotsMatcher = cuda::createTemplateMatching(CV_8UC1, TM_CCOEFF_NORMED);

    Mat frameBGRCPU;
    cuda::GpuMat frameBGR, frame, correlation;

    KalmanFilter KF_reflection = makePixelKalmanFilter();
    KalmanFilter KF_pupil      = makePixelKalmanFilter();
    KalmanFilter KF_head       = makePixelKalmanFilter();

    Point2i reflection = None, pupil = None, head = None;

    std::chrono::time_point<std::chrono::steady_clock> last_frame_time = std::chrono::steady_clock::now();
    int frameIndex = 0;
    std::ostringstream fpsText;
    fpsText << std::fixed << std::setprecision(2);

    cuda::Stream streamSpots;
    #ifndef HEADLESS
        cuda::Stream streamDisplay;
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
                if (frameBGRCPU.type() == CV_8UC3) cuda::cvtColor(frameBGR, frame, COLOR_BGR2GRAY);
                else frame = frameBGR;
                frame.download(frameBGRCPU);
                vwInput.write(frameBGRCPU);
                frame = frame(ROI);
            }
            else {
                // Crop the ROI immediately, we won't need the rest of the image again
                frameBGR.upload(frameBGRCPU(ROI));
                // frameBGR might not actually be BGR if it's loaded from a greyscale video
                if (frameBGRCPU.type() == CV_8UC3) cuda::cvtColor(frameBGR, frame, COLOR_BGR2GRAY);
                else frame = frameBGR;
            }
        #else
            // Keep the rest of the image for display
            frameBGR.upload(frameBGRCPU);
            if (frameBGRCPU.type() == CV_8UC3) {
                /* If we are recording the input, we need to convert it all to greyscale.
                 * Otherwise, we only need to convert the ROI. */
                if (vwInput.isOpened()) {
                    cuda::cvtColor(frameBGR, frame, COLOR_BGR2GRAY);
                    frame.download(frameBGRCPU);
                    vwInput.write(frameBGRCPU);
                    frame = frame(ROI);
                }
                else cuda::cvtColor(frameBGR(ROI), frame, COLOR_BGR2GRAY);
            }
            else {
                if (vwInput.isOpened()) vwInput.write(frameBGRCPU);
                frame = frameBGR(ROI);
            }
        #endif

        spotsMatcher->match(frame, spots, correlation, streamSpots);
        std::vector<PointWithRating> pupils = findCircles(frame, Pupil::THRESHOLD, Pupil::MIN_RADIUS, Pupil::MAX_RADIUS);
        std::vector<PointWithRating> irises = findCircles(frame, Iris::THRESHOLD, Iris::MIN_RADIUS, Iris::MAX_RADIUS);

        if (irises.size() > 0) {
            std::vector<PointWithRating>::const_iterator bestIris = std::min_element(irises.cbegin(), irises.cend());
            for (int i = 0; i < 5; ++i) { // Limit number of iterations
                if (pupils.size() == 0) {
                    KF_pupil.correct(toMat(static_cast<Point2f>(ROI.tl()) + bestIris->point));
                    break;
                }
                std::vector<PointWithRating>::const_iterator bestPupil = std::min_element(pupils.cbegin(), pupils.cend());
                if (norm(bestIris->point - bestPupil->point) < MAX_PUPIL_IRIS_SEPARATION) {
                    // They should be the same point
                    KF_pupil.correct(toMat(static_cast<Point2f>(ROI.tl()) + (bestIris->point + bestPupil->point)/2));
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
            cuda::cvtColor(frame, frame, COLOR_GRAY2BGR, 0, streamDisplay);
            cuda::copyMakeBorder(frame, frame, ROI.y, RESOLUTION_Y - ROI.y - ROI.height, ROI.x, RESOLUTION_X - ROI.x - ROI.width, BORDER_CONSTANT, 0, streamDisplay);
            cuda::addWeighted(frameBGR, 0.75, frame, 0.25, 0, frameBGR, -1, streamDisplay);
        #endif

        double maxVal = -1;
        Point2i maxLoc = None;
        streamSpots.waitForCompletion();
        cuda::minMaxLoc(correlation, nullptr, &maxVal, nullptr, &maxLoc);

        if (maxLoc.y > 0 and maxLoc.x > 0 and maxVal > TEMPLATE_MATCHING_THRESHOLD) {
            maxLoc += ROI.tl();
            KF_reflection.correct((KFMat(2, 1) << maxLoc.x + spots.cols/2, // integer division intentional
                                                  maxLoc.y + spots.rows/2));
        }

        reflection = toPoint(KF_reflection.predict());
        pupil = toPoint(KF_pupil.predict());

        EyePosition eyePos = Geometry::eyePosition(reflection, pupil);
        if (eyePos.eyeCentre) {
            Point2i headPixel = Geometry::unproject(*eyePos.eyeCentre);
            KF_head.correct((KFMat(2, 1) << float(headPixel.x), float(headPixel.y)));
        }

        head = toPoint(KF_head.predict());

        if (++frameIndex == FRAMES_FOR_FPS_MEASUREMENT) {
            const std::chrono::duration<float> frame_time = std::chrono::steady_clock::now() - last_frame_time;
            fpsText.str(""); // Clear contents of fpsText
            fpsText << 1s/(frame_time/FRAMES_FOR_FPS_MEASUREMENT);
            frameIndex = 0;
            last_frame_time = std::chrono::steady_clock::now();
        }
        #ifdef HEADLESS
            std::cerr << "FPS = " << fpsText.str() << '\n';
        #else
            streamDisplay.waitForCompletion();
            /* We could avoid having to download the frame from the GPU by using OpenGL.
             * This requires OpenCV to be build with OpenGL support, and the line
             * namedWindow(windowName, WINDOW_AUTOSIZE | WINDOW_OPENGL)
             * to be run before the start of the main loop.
             * Then, cv::imshow can be called directly on cv::cuda::GpuMat.
             * However, instead of using cv::circle and cv::putText, we would have to use OpenGL functions directly.
             * This would be a lot of effort for a very marginal performance benefit. */
            frameBGR.download(frameBGRCPU);
            if (reflection != None)  circle(frameBGRCPU, reflection,  32, Scalar(0x00, 0x00, 0xFF), 5);
            if (pupil != None) circle(frameBGRCPU, pupil, 32, Scalar(0xFF, 0x00, 0x00), 5);
            if (head != None) circle(frameBGRCPU, head, 32, Scalar(0x00, 0xFF, 0x00), 5);
            putText(frameBGRCPU, fpsText.str(), Point2i(100, 100), FONT_HERSHEY_SIMPLEX, 3, Scalar(0x00, 0x00, 0xFF), 3);

            imshow(windowName, frameBGRCPU);
            if (vwOutput.isOpened()) vwOutput.write(frameBGRCPU);

            bool quitting = false;
            switch (waitKey(1) & 0xFF) {
            case 27: // Esc
            case 'q':
                quitting = true;
                break;
            case 's':
                imwrite("frame.png", frameBGRCPU);
                break;
            case 'v': // Record input video; only if the input is a live feed
                if (isRealtime) vwInput.open("recorded_input.mp4", fourcc, FPS, {RESOLUTION_X, RESOLUTION_Y}, false);
                break;
            case 'w': // Record output video
                vwOutput.open("recorded_output.mp4", fourcc, FPS, {RESOLUTION_X, RESOLUTION_Y}, true);
                break;
            }

            if (quitting || getWindowProperty(windowName, WND_PROP_AUTOSIZE) == -1) break; // Window closed by user
        #endif
    }

    video.release();
    #ifndef HEADLESS
        destroyAllWindows();
    #endif
    return 0;
}
