#ifndef VISUALIZER_H
#define VISUALIZER_H

#include "EyeTracker.hpp"
#include "FeatureDetector.hpp"

#include <opencv2/opencv.hpp>

#include <string_view>

namespace et {
class Visualizer {
public:
    Visualizer(FeatureDetector *feature_detector, EyeTracker *eye_tracker);
    void drawUi(const cv::Mat& image, int camera_id);
    void show();
    void calculateFramerate();
    void printFramerateInterval();
    cv::Mat getUiImage(int camera_id);
    float getAvgFramerate();
    bool isWindowOpen();

private:
    static constexpr int FRAMES_FOR_FPS_MEASUREMENT{8};
    static constexpr std::string_view WINDOW_NAME[]{"Output (left)", "Output (right)"};
    static constexpr std::string_view SLIDER_WINDOW_NAME[]{"Parameters (left)", "Parameters (right)"};

    static constexpr std::string_view PUPIL_THRESHOLD_NAME{"Pupil threshold"};
    static constexpr int PUPIL_THRESHOLD_MAX{255};
    static int pupil_threshold_tracker_[2];

    static constexpr std::string_view GLINT_THRESHOLD_NAME{"Glint threshold"};
    static constexpr int GLINT_THRESHOLD_MAX{255};
    static int glint_threshold_tracker_[2];

    static constexpr std::string_view PUPIL_EXPOSURE_NAME{"Pupil exposure"};
    static constexpr int PUPIL_EXPOSURE_MAX{1000};
    static constexpr int PUPIL_EXPOSURE_MIN{0};
    static int pupil_exposure_tracker_[2];

    static constexpr std::string_view GLINT_EXPOSURE_NAME{"Glint exposure"};
    static constexpr int GLINT_EXPOSURE_MAX{1000};
    static constexpr int GLINT_EXPOSURE_MIN{0};
    static int glint_exposure_tracker_[2];

    static void onPupilLeftThresholdUpdate(int, void *);
    static void onPupilRightThresholdUpdate(int, void *);
    static void onGlintLeftThresholdUpdate(int, void *);
    static void onGlintRightThresholdUpdate(int, void *);
    static void onPupilLeftExposureUpdate(int, void *);
    static void onPupilRightExposureUpdate(int, void *);
    static void onGlintLeftExposureUpdate(int, void *);
    static void onGlintRightExposureUpdate(int, void *);

    cv::Mat image_[2]{};
    FeatureDetector *feature_detector_{};
    EyeTracker *eye_tracker_{};
    std::ostringstream fps_text_{};
    int frame_index_{};
    std::chrono::time_point<std::chrono::steady_clock> last_frame_time_{};
    int total_frames_{0};
    float total_framerate_{0};
};
}// namespace et
#endif