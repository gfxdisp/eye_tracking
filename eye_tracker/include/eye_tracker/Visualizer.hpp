#ifndef HDRMFS_EYE_TRACKER_VISUALIZER_HPP
#define HDRMFS_EYE_TRACKER_VISUALIZER_HPP

#include <eye_tracker/Settings.hpp>

#include <opencv2/opencv.hpp>

#include <string_view>

namespace et {
    class Visualizer {
    public:
        Visualizer(int camera_id, bool headless);

        void prepareImage(const cv::Mat& image);

        void drawPupil(cv::Point2d pupil, int radius, double diameter_mm);

        void drawGlints(const std::vector<cv::Point2d>& glints, std::vector<bool>& glints_validity);

        void drawBoundingCircle(cv::Point2d centre, int radius);

        void drawEyeCentre(cv::Point2d eye_centre);

        void drawCorneaCentre(cv::Point2d cornea_centre);

        void drawGlintEllipse(const cv::RotatedRect& ellipse);

        void drawCorneaTrace();

        void drawGaze(cv::Point2d normalized_gaze_point);

        void drawMarker(cv::Point2d normalized_marker);

        void drawGazeTrace();

        void drawFps();

        void show() const;

        void calculateFramerate();

        void printFramerateInterval();

        cv::Mat getUiImage() const;

        bool isWindowOpen() const;

    private:
        static constexpr int FRAMES_FOR_FPS_MEASUREMENT{8};

        static constexpr int game_window_size{300};

        static constexpr int game_window_offset{50};

        static constexpr std::string_view SIDE_NAMES[]{"Left ", "Right "};

        static constexpr std::string_view WINDOW_NAME{"output"};

        static constexpr std::string_view PUPIL_THRESHOLD_NAME{"Pupil threshold"};

        static constexpr int PUPIL_THRESHOLD_MAX{255};

        static constexpr std::string_view GLINT_THRESHOLD_NAME{"Glint threshold"};

        static constexpr int GLINT_THRESHOLD_MAX{255};

        static constexpr std::string_view EXPOSURE_NAME{"Exposure"};

        static constexpr int EXPOSURE_MAX{1000};

        static constexpr int EXPOSURE_MIN{0};

        static void onPupilThresholdUpdate(int value, void* ptr);

        void onPupilThresholdUpdate(int value) const;

        static void onGlintThresholdUpdate(int value, void* ptr);

        void onGlintThresholdUpdate(int value) const;

        static void onExposureUpdate(int value, void* ptr);

        void onExposureUpdate(int value) const;

        cv::Mat image_{};

        std::string full_output_window_name_{};

        std::ostringstream fps_text_;

        int frame_index_{};

        std::chrono::time_point<std::chrono::steady_clock> last_frame_time_;

        int total_frames_{};

        double total_framerate_{};

        FeaturesParams* user_params_{};

        int* pupil_threshold_{};

        int* glint_threshold_{};

        std::chrono::steady_clock::time_point framerate_timer_{};

        bool headless_{};

        constexpr static int CORNEA_HISTORY_SIZE = 50;
        cv::Point2d previous_cornea_centres_[CORNEA_HISTORY_SIZE]{};
        int cornea_history_index_{};
        bool cornea_history_full_{};

        constexpr static int GAZE_HISTORY_SIZE = 50;
        cv::Point2d previous_gaze_points_[GAZE_HISTORY_SIZE]{};
        int gaze_point_index_{};
        bool gaze_point_history_full_{};
    };
} // namespace et
#endif //HDRMFS_EYE_TRACKER_VISUALIZER_HPP
