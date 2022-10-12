#ifndef HDRMFS_EYE_TRACKER_VISUALIZER_HPP
#define HDRMFS_EYE_TRACKER_VISUALIZER_HPP

#include "Settings.hpp"

#include <opencv2/opencv.hpp>

#include <string_view>

namespace et {
/**
 * Presents camera's output and allows changing individual detection parameters.
 */
class Visualizer {
public:
    /**
     * Creates a window and trackbar used to change individual
     * detection parameters.
     * @param camera_id An id of the camera to which the object corresponds.
     */
    void initialize(int camera_id);
    /**
     * Loads an image to present and converts it to RGB scale.
     * @param image Image to load.
     */
    void prepareImage(const cv::Mat &image);
    /**
     * Draw a circle around a pupil on the image loaded with prepareImage().
     * @param pupil Centre of the pupil.
     * @param radius Radius of the pupil.
     */
    void drawPupil(cv::Point2f pupil, int radius);
    /**
     * Draws circles around glint positions in the image loaded
     * with prepareImage().
     * @param glints Vector of glints to draw.
     */
    void drawGlints(std::vector<cv::Point2f> *glints);
    /**
     * Draws a circle around the hole in the view piece in the image loaded
     * with prepareImage().
     * @param centre Centre of the hole.
     * @param radius Radius of the hole.
     */
    void drawBoundingCircle(cv::Point2f centre, int radius);
    /**
     * Draws a circle around the eye centre in the image loaded
     * with prepareImage().
     * @param eye_centre Eye centre position in image space.
     */
    void drawEyeCentre(cv::Point2f eye_centre);
    /**
     * Draws a circle around the cornea centre in the image loaded
     * with prepareImage().
     * @param cornea_centre Cornea centre position in image space.
     */
    void drawCorneaCentre(cv::Point2f cornea_centre);
    /**
     * Draws an ellipse formed from glints in the image loaded with prepareImage().
     * @param ellipse Ellipse formed from glints.
     */
    void drawGlintEllipse(cv::RotatedRect ellipse);
    /**
     * Draws a text with measured framerate in the image loaded with prepareImage().
     */
    void drawFps();
    /**
     * Updates the window image to the image loaded with prepareImage().
     */
    void show();
    /**
     * Calculates framerate between subsequent calls of this method.
     */
    static void calculateFramerate();
    /**
     * Prints framarate calculated in calculateFramerate().
     */
    static void printFramerateInterval();
    /**
     * Retrieves image that is drawn to using prepareImage() and all draw methods.
     * @return Image matrix.
     */
    cv::Mat getUiImage();
    /**
     * Retrieves an average framerate since the start of the application.
     * @return Average framerate value.
     */
    static float getAvgFramerate();
    /**
     * Checks if the managed window was closed.
     * @return True if the window was closed. False otherwise.
     */
    bool isWindowOpen();

private:
    // Sets how often the framerate is updated.
    static constexpr int FRAMES_FOR_FPS_MEASUREMENT{8};
    // Prefixes of the window names depending on the eye.
    static constexpr std::string_view SIDE_NAMES[]{"Left ", "Right "};
    // Window name.
    static constexpr std::string_view WINDOW_NAME{"output"};

    // Name of the pupil threshold parameter in the trackbar.
    static constexpr std::string_view PUPIL_THRESHOLD_NAME{"Pupil threshold"};
    // Maximal value of the pupil threshold parameter.
    static constexpr int PUPIL_THRESHOLD_MAX{255};

    // Name of the glint threshold parameter in the trackbar.
    static constexpr std::string_view GLINT_THRESHOLD_NAME{"Glint threshold"};
    // Maximal value of the glint threshold parameter.
    static constexpr int GLINT_THRESHOLD_MAX{255};

    // Name of the camera's exposure parameter in the trackbar
    static constexpr std::string_view EXPOSURE_NAME{"Exposure"};
    // Maximal value of the camera's exposure in milliseconds.
    static constexpr int EXPOSURE_MAX{1000};
    // Minimal value of the camera's exposure in milliseconds.
    static constexpr int EXPOSURE_MIN{0};

    /**
     * Called whenever a pupil threshold bar is moved.
     * @param value New value of the trackbar.
     * @param ptr Pointer to the Visualizer object.
     */
    static void onPupilThresholdUpdate(int value, void *ptr);
    /**
     * Updates the pupil threshold value.
     * @param value Updated pupil threshold value.
     */
    void onPupilThresholdUpdate(int value);
    /**
     * Called whenever a glint threshold bar is moved.
     * @param value New value of the trackbar.
     * @param ptr Pointer to the Visualizer object.
     */
    static void onGlintThresholdUpdate(int value, void *ptr);
    /**
     * Updates the glint threshold value.
     * @param value Updated glint threshold value.
     */
    void onGlintThresholdUpdate(int value);
    /**
     * Called whenever a pupil threshold bar is moved.
     * @param value New value of the trackbar.
     * @param ptr Pointer to the Visualizer object.
     */
    static void onExposureUpdate(int value, void *ptr);
    /**
     * Updates the camera's exposure value.
     * @param value Updated camera's exposure value.
     */
    void onExposureUpdate(int value);

    // Image that is drawn to and shown in the window.
    cv::Mat image_{};

    // Name of the window including the eye side.
    std::string full_output_window_name_{};

    // Framerate converted to text calculate in calculateFramerate().
    static std::ostringstream fps_text_;
    // Frame counter used to update the framerate every
    // FRAMES_FOR_FPS_MEASUREMENT frames.
    static int frame_index_;
    // Time of the last framerate measurement.
    static std::chrono::time_point<std::chrono::steady_clock> last_frame_time_;
    // Total number of frames since the application start.
    static int total_frames_;
    // Summed up framerate measurements after each calculateFramerate() call.
    static float total_framerate_;

    // Detection parameters for the current user.
    FeaturesParams *user_params_{};
    // Threshold value for pupil detection used by FeatureDetector.
    int *pupil_threshold_{};
    // Threshold value for glints detection used by FeatureDetector.
    int *glint_threshold_{};
};
} // namespace et
#endif //HDRMFS_EYE_TRACKER_VISUALIZER_HPP