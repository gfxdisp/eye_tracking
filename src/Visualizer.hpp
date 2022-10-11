#ifndef VISUALIZER_H
#define VISUALIZER_H

#include "Settings.hpp"

#include <opencv2/opencv.hpp>

#include <string_view>

namespace et {
class Visualizer {
public:
    void initialize(int camera_id);
    void prepareImage(const cv::Mat& image);
    void drawPupil(cv::Point2f pupil, int radius);
    void drawGlints(std::vector<cv::Point2f> *glints);
    void drawBoundingCircle(cv::Point2f centre, int radius);
    void drawEyeCentre(cv::Point2f eye_centre);
    void drawCorneaCentre(cv::Point2f cornea_centre);
    void drawGlintEllipse(cv::RotatedRect ellipse);
    void drawFps();
    void show();
    static void calculateFramerate();
    void printFramerateInterval();
    cv::Mat getUiImage();
    static float getAvgFramerate();
    bool isWindowOpen();

private:
    static constexpr int FRAMES_FOR_FPS_MEASUREMENT{8};
    static constexpr std::string_view SIDE_NAMES[]{"Left ", "Right "};
    static constexpr std::string_view WINDOW_NAME{"output"};
    static constexpr std::string_view SLIDER_WINDOW_NAME{"Parameters"};

    static constexpr std::string_view PUPIL_THRESHOLD_NAME{"Pupil threshold"};
    static constexpr int PUPIL_THRESHOLD_MAX{255};

    static constexpr std::string_view GLINT_THRESHOLD_NAME{"Glint threshold"};
    static constexpr int GLINT_THRESHOLD_MAX{255};

    static constexpr std::string_view PUPIL_EXPOSURE_NAME{"Pupil exposure"};
    static constexpr int PUPIL_EXPOSURE_MAX{1000};
    static constexpr int PUPIL_EXPOSURE_MIN{0};

    static void onPupilThresholdUpdate(int value, void *ptr);
    void onPupilThresholdUpdate(int value);
    static void onGlintThresholdUpdate(int value, void *ptr);
    void onGlintThresholdUpdate(int value);
    static void onExposureUpdate(int value, void *ptr);
    void onExposureUpdate(int value);

    cv::Mat image_{};

    std::string full_output_window_name_{};
    std::string full_parameters_window_name_{};

    static std::ostringstream fps_text_;
    static int frame_index_;
    static std::chrono::time_point<std::chrono::steady_clock> last_frame_time_;
    static int total_frames_;
    static float total_framerate_;

    CameraParams *camera_params_{};
    int *pupil_threshold_{};
    int *glint_threshold_{};
};
}// namespace et
#endif