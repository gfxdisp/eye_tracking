#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <algorithm> // std::min_element, std::transform
#include <chrono>
#include <iostream>
#include <memory> // std::unique_ptr
#include <sstream> // std::ostringstream
#include <string>
#include <vector>
#include <cctype> // std::tolower
#include <cstdlib> // std::strtoul
#include "geometry.hpp"
#include "image_processing.hpp"
#ifdef HAVE_UEYE
    #include "idslib/IDSVideoCapture.h"
#endif
using namespace cv;
using namespace std::chrono_literals;
using namespace EyeTracker;
using namespace Camera;
using namespace ImageProcessing;

namespace EyeTracker {
    const static Point2i None = {-1, -1};
    using KFMat = Mat_<float>;
    #ifndef HEADLESS
        const char* windowName = "frame";
    #endif

    inline Point toPoint(Mat m) {
        return Point(m.at<float>(0, 0), m.at<float>(0, 1));
    }

    inline Mat toMat(Point p) {
        return (Mat_<float>(2, 1) << p.x, p.y);
    }

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
                std::cerr << "Error: use of IDS camera requested, but the program was compiled without uEye support.";
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
    std::unique_ptr<VideoCapture> video(nullptr);

    // Process arguments
    if (argc < 2) return fail(EyeTracker::Error::ARGUMENTS);
    else {
        std::string mode(argv[1]);
        // Convert to lowercase (for user's convenience)
        std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c){ return std::tolower(c); });
        if (mode == "file") {
            if (argc < 3) return fail(EyeTracker::Error::ARGUMENTS); // No file path specified
            else {
                video = std::make_unique<VideoCapture>(argv[2]);
                if (!video->isOpened()) return fail(EyeTracker::Error::FILE_NOT_FOUND);
            }
        }
        else if (mode == "ids" || mode == "ueye") {
            #ifdef HAVE_UEYE
                unsigned long cameraIndex = argc >= 3 ? std::strtoul(argv[2], nullptr, 10) : 0;
                video = std::make_unique<IDSVideoCapture>(cameraIndex);
                if (!video->isOpened()) return fail(EyeTracker::Error::UEYE_NOT_FOUND);
            #else
                return fail(EyeTracker::Error::BUILT_WITHOUT_UEYE);
            #endif
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

    constexpr float DT = 1/FPS;
    constexpr float VELOCITY_DECAY = 0.9;
    const Mat TRANSITION_MATRIX  = (KFMat(4, 4) << 1, 0, DT, 0,
                                                   0, 1, 0, DT,
                                                   0, 0, VELOCITY_DECAY, 0,
                                                   0, 0, 0, VELOCITY_DECAY);
    const Mat MEASUREMENT_MATRIX = (KFMat(2, 4) << 1, 0, 0, 0,
                                                   0, 1, 0, 0);
    const Mat PROCESS_NOISE_COV     = Mat::eye(4, 4, CV_32F) * 100;
    const Mat MEASUREMENT_NOISE_COV = Mat::eye(2, 2, CV_32F) * 50;
    const Mat ERROR_COV_POST = Mat::eye(4, 4, CV_32F) * 0.1;
    const Mat STATE_POST = (KFMat(4, 1) << RESOLUTION_X/2.0, RESOLUTION_Y/2.0, 0, 0);

    KalmanFilter KF_reflection(4, 2);
    // clone() is needed as, otherwise, the matrices will be used by reference, and all the filters will be the same
    KF_reflection.transitionMatrix  = TRANSITION_MATRIX.clone();
    KF_reflection.measurementMatrix = MEASUREMENT_MATRIX.clone();
    KF_reflection.processNoiseCov   = PROCESS_NOISE_COV.clone();
    KF_reflection.measurementNoiseCov = MEASUREMENT_NOISE_COV.clone();
    KF_reflection.errorCovPost      = ERROR_COV_POST.clone();
    KF_reflection.statePost         = STATE_POST.clone();
    KF_reflection.predict(); // Without this line, OpenCV complains about incorrect matrix dimensions
    KalmanFilter KF_pupil(4, 2);
    KF_pupil.transitionMatrix  = TRANSITION_MATRIX.clone();
    KF_pupil.measurementMatrix = MEASUREMENT_MATRIX.clone();
    KF_pupil.processNoiseCov   = PROCESS_NOISE_COV.clone();
    KF_pupil.measurementNoiseCov = MEASUREMENT_NOISE_COV.clone();
    KF_pupil.errorCovPost      = ERROR_COV_POST.clone();
    KF_pupil.statePost         = STATE_POST.clone();
    KF_pupil.predict();
    KalmanFilter KF_head(4, 2);
    KF_head.transitionMatrix  = TRANSITION_MATRIX;
    KF_head.measurementMatrix = MEASUREMENT_MATRIX;
    KF_head.processNoiseCov   = PROCESS_NOISE_COV;
    KF_head.measurementNoiseCov = MEASUREMENT_NOISE_COV;
    KF_head.errorCovPost      = ERROR_COV_POST;
    KF_head.statePost         = STATE_POST;
    KF_head.predict();

    Point2i reflection = None, pupil = None, head = None;

    std::chrono::time_point<std::chrono::steady_clock> last_frame_time = std::chrono::steady_clock::now();
    constexpr int FRAMES_FOR_FPS_MEASUREMENT = 8;
    int frameIndex = 0;
    std::ostringstream fpsText;
    fpsText << std::fixed << std::setprecision(2);

    cuda::Stream streamSpots;
    #ifndef HEADLESS
        cuda::Stream streamDisplay;
    #endif
    while (true) {
        if (!video->read(frameBGRCPU) || frameBGRCPU.empty()) break; // Video has ended
        #ifdef HEADLESS
            // Crop the ROI immediately, we won't need the rest of the image again
            frameBGR.upload(frameBGRCPU(ROI));
            cuda::cvtColor(frameBGR, frame, COLOR_BGR2GRAY);
        #else
            // Keep the rest of the image for display
            frameBGR.upload(frameBGRCPU);
            cuda::cvtColor(frameBGR(ROI), frame, COLOR_BGR2GRAY);
        #endif

        spotsMatcher->match(frame, spots, correlation, streamSpots);
        std::vector<PointWithRating> pupils = findCircles(frame, 2, 66, 90);
        std::vector<PointWithRating> irises = findCircles(frame, 5, 110, 150);

        if (irises.size() > 0) {
            std::vector<PointWithRating>::const_iterator bestIris = std::min_element(irises.cbegin(), irises.cend());
            for (int i = 0; i < 5; ++i) { // Limit number of iterations
                if (pupils.size() == 0) {
                    KF_pupil.correct(toMat(static_cast<Point2f>(ROI.tl()) + bestIris->point));
                    break;
                }
                std::vector<PointWithRating>::const_iterator bestPupil = std::min_element(pupils.cbegin(), pupils.cend());
                if (norm(bestIris->point - bestPupil->point) < 12.24) {
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

        if (maxLoc.y > 0 and maxLoc.x > 0 and maxVal > 0.9) {
            maxLoc += ROI.tl();
            KF_reflection.correct((KFMat(2, 1) << maxLoc.x + spots.cols/2, // integer division intentional
                                                  maxLoc.y + spots.rows/2));
        }

        reflection = toPoint(KF_reflection.predict());
        pupil = toPoint(KF_pupil.predict());

        Geometry::EyePosition eyePos = Geometry::eyePosition(reflection, pupil);
        if (eyePos.eyeCentre) {
            Geometry::Vector headImage = (*eyePos.eyeCentre - 12.79 * Geometry::o)/-11.79;
            Point2i headPixel = Geometry::WCStoPixel(headImage);
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

            bool quitting = false;
            switch (waitKey(1) & 0xFF) {
            case 27: // Esc
            case 'q':
                quitting = true;
                break;
            case 's':
                imwrite("frame.png", frameBGRCPU);
                break;
            }

            if (quitting || getWindowProperty(windowName, WND_PROP_AUTOSIZE) == -1) break; // Window closed by user
        #endif
    }

    video->release();
    #ifndef HEADLESS
        destroyAllWindows();
    #endif
    return 0;
}
