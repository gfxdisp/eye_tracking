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
    void drawUi(const cv::Mat& image);
    void show();
    void calculateFramerate();
    void printFramerateInterval();
    cv::Mat getUiImage();
    float getAvgFramerate();
    bool isWindowOpen();

private:
    static constexpr int FRAMES_FOR_FPS_MEASUREMENT{8};
    static constexpr std::string_view WINDOW_NAME{"Output"};
    static constexpr std::string_view SLIDER_WINDOW_NAME{"Parameters"};

    static constexpr std::string_view PUPIL_THRESHOLD_NAME{"Pupil threshold"};
    static constexpr int PUPIL_THRESHOLD_MAX{255};
    static int pupil_threshold_tracker_;

    static constexpr std::string_view GLINT_THRESHOLD_NAME{"Glint threshold"};
    static constexpr int GLINT_THRESHOLD_MAX{255};
    static int glint_threshold_tracker_;

    static constexpr std::string_view ALPHA_NAME{"Alpha angle"};
    static constexpr int ALPHA_MAX{100};
    static constexpr int ALPHA_MIN{-100};
    static int alpha_tracker_;

    static constexpr std::string_view BETA_NAME{"Beta angle"};
    static constexpr int BETA_MAX{100};
    static constexpr int BETA_MIN{-100};
    static int beta_tracker_;

    static void onPupilThresholdUpdate(int, void *);
    static void onGlintThresholdUpdate(int, void *);
    static void onAlphaUpdate(int, void *);
    static void onBetaUpdate(int, void *);

    cv::Mat image_{};
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