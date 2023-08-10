#ifndef HDRMFS_EYE_TRACKER_EYE_TRACKER_HPP
#define HDRMFS_EYE_TRACKER_EYE_TRACKER_HPP

#include "EyeEstimator.hpp"
#include "FeatureDetector.hpp"
#include "Visualizer.hpp"

#include <fstream>
#include <vector>

namespace et {

/**
 * Type of visualization shown in the application window.
 */
enum class VisualizationType {
    // Visualization disabled, only prints framerate to console.
    DISABLED,
    // Shows raw image from the camera.
    CAMERA_IMAGE,
    // Shows video feed image after thresholding that is used to calculate
    // pupil position.
    THRESHOLD_PUPIL,
    // Shows video feed image after thresholding that is used to calculate
    // glints positions.
    THRESHOLD_GLINTS
};

/**
 * Pulls images from the camera (or video/folder), detects
 * eye features, and estimates eye position. It also controls the state of the
 * window showing the detection results.
 */
class EyeTracker {
public:
    /**
     * Initializes all class components. Needs to be run before any
     * other class method.
     * @param image_provider pointer to the object used to serve as a video feed.
     * @param settings_path Path to a folder containing all settings files.
     * @param input_path Path to a folder with csv files.
     * @param enabled_cameras 2-element boolean vector signifying, whether
     * left and right cameras (in this order) are enabled .
     * @param ellipse_fitting 2-element boolean vector signifying, whether
     * left and right cameras (in this order) use ellipse polynomial fitting as
     * eye-detection algorithm. False values would use the model based
     * on 2006 Guestrin et al. model.
     * @param enabled_kalman 2-element boolean vector signifying, whether to
     * enable Kalman filtering for feature and eye position estimation.
     * @param enabled_template_matching 2-element boolean vector signifying,
     * whether to use template matching for finding glints.
     * @param headless If true, the application will not show any windows.
     * @param distorted 2-element boolean vector signifying, whether the images require applying undistortion.
     */
    void initialize(ImageProvider *image_provider, const std::string &settings_path, const std::string &input_path,
                    const bool enabled_cameras[], const bool ellipse_fitting[], const bool enabled_kalman[],
                    const bool enabled_template_matching[], const bool distorted[], bool headless);

    /**
     * Takes the next frame from the image provider, find its features, and calculates
     * eye parameters.
     * @return True, if the next frame was successfully captured from the image
     * provider. Otherwise, it returns false.
     */
    bool analyzeNextFrame();

    /**
     * Retrieves pupil position in image space that was previously
     * calculated in analyzeNextFrame().
     * @param pupil Variable that will contain the pupil position.
     * @param camera_id An id of the camera for which the value is returned.
     */
    void getPupil(cv::Point2f &pupil, int camera_id);

    /**
     * Retrieves pupil position in image space that was previously
     * calculated in analyzeNextFrame(). Value is averaged across a number
     * of frames set using the setGazeBufferSize() method.
     * @param pupil Variable that will contain the pupil position.
     * @param camera_id An id of the camera for which the value is returned.
     */
    void getPupilFiltered(cv::Point2f &pupil, int camera_id);

    /**
     * Retrieves a vector between a specific glint and pupil position
     * in image space based on parameters that were previously calculated
     * in analyzeNextFrame().
     * @param pupil_glint_vector Variable that will contain the vector.
     * @param camera_id An id of the camera for which the value is returned.
     */
    void getPupilGlintVector(cv::Vec2f &pupil_glint_vector, int camera_id);

    /**
     * Retrieves a vector between a specific glint and pupil position
     * in image space based on parameters that were previously calculated
     * in analyzeNextFrame(). Value is averaged across a number of frames set
     * using the setGazeBufferSize() method.
     * @param pupil_glint_vector Variable that will contain the vector.
     * @param camera_id An id of the camera for which the value is returned.
     */
    void getPupilGlintVectorFiltered(cv::Vec2f &pupil_glint_vector, int camera_id);

    /**
     * Retrieves pupil diameter in millimeters that was previously calculated
     * in analyzeNextFrame().
     * @param pupil_diameter Variable that will contain the pupil diameter.
     * @param camera_id An id of the camera for which the value is returned.
     */
    void getPupilDiameter(float &pupil_diameter, int camera_id);

    /**
     * Used to set the number of frames, across which getPupilFiltered() and
     * getPupilGlintVectorFiltered() are calculated.
     * @param value Number of frames.
     */
    void setGazeBufferSize(uint8_t value);

    /**
     * Retrieves eye centre position in camera space that was previously
     * calculated in analyzeNextFrame().
     * @param eye_centre Variable that will contain the eye centre position.
     * @param camera_id An id of the camera for which the value is returned.
     */
    void getEyeCentrePosition(cv::Vec3d &eye_centre, int camera_id);

    /**
     * Retrieves cornea centre position in camera space that was previously
     * calculated in analyzeNextFrame().
     * @param cornea_centre Variable that will contain the cornea centre position.
     * @param camera_id An id of the camera for which the value is returned.
     */
    void getCorneaCentrePosition(cv::Vec3d &cornea_centre, int camera_id);

    /**
     * Calculates the gaze direction in camera space based on parameters that
     * were previously calculated in analyzeNextFrame().
     * @param gaze_direction Variable that will contain the normalized
     * gaze direction.
     * @param camera_id An id of the camera for which the value is returned.
     */
    void getGazeDirection(cv::Vec3f &gaze_direction, int camera_id);

    /**
     * Writes detected features (pupil, glints) to output.
     * @param output A stream to which the parameters will be written.
     * @param camera_id An id of the camera for which the log is saved.
     */
    void logDetectedFeatures(std::ostream &output, int camera_id);

    /**
     * Writes detected eye centre and cornea centre positions to output.
     * @param output A stream to which the parameters will be written.
     * @param camera_id An id of the camera for which the log is saved.
     */
    void logEyePosition(std::ostream &output, int camera_id);

    /**
     * Enables and disables recording of current feed, along with feed with UI
     * elements added to it (fps counter, eye position, etc.). Videos will be
     * saved to the videos/ directory at the location from which the app was run.
     */
    void switchVideoRecordingState();

    /**
     * Stops any enabled video recordings and saves the results.
     */
    void stopVideoRecording();

    /**
     * Captures the image from the current video feed. It will be
     * saved to the images/ directory at the location from which the app was run.
     */
    void captureCameraImage();

    /**
     * Updates the window UI elements based on the parameters that were previously
     * calculated in analyzeNextFrame(). If the window updates are disabled,
     * prints the framerate to console once every second.
     */
    void updateUi();

    /**
     * Disables updates to the window.
     */
    void disableImageUpdate();

    /**
     * Shows raw image from video feed in the window.
     */
    void switchToCameraImage();

    /**
     * Shows video feed image after thresholding that is used to calculate
     * pupil position.
     */
    void switchToPupilThreshImage();

    /**
     * Shows video feed image after thresholding that is used to calculate
     * glints positions.
     */
    void switchToGlintThreshImage();

    /**
     * Checks, whether any of the windows was closed.
     * @return True if any window was closed, false otherwise.
     */
    bool shouldAppClose();

    /**
     * Retrieves number of frames analyzed in one second.
     * @return Calculated framerate.
     */
    static float getAvgFramerate();

    /**
     * Starts a recording of eyes through a remote application.
     * @param folder_name Folder to which the videos will be saved.
     */
    std::string startEyeVideoRecording(const std::string &folder_name);

    /**
     * Stops a video recording started through startEyeVideoRecording().
     */
    void stopEyeVideoRecording();

    /**
     * Saves current frame along with external data initialized by
     * startEyeVideoRecording().
     * @param eye_data External string with all the information that is saved.
     */
    void saveEyeData(const std::string &eye_data);

    void calibrateTransform(const cv::Mat &M_et_left, const cv::Mat &M_et_right,
                            const std::string &plane_calibration_path, const std::string &gaze_calibration_path);

private:
    // Object serving as a video feed.
    ImageProvider *image_provider_{};
    // Objects estimating image features (pupils and glints). One object per eye.
    FeatureDetector feature_detectors_[2]{};
    // Objects calculating eye positions. One object per eye.
    EyeEstimator eye_estimators_[2]{};
    // Objects showing windows with output. One object per window of one eye.
    Visualizer visualizer_[2]{};

    // Path to a folder with all configuration files.
    std::string settings_path_{};

    // Raw images retrieved directly from image_provider_. Two images per eye: one for pupil, one for glints.
    ImageToProcess analyzed_frame_[2]{};
    // Status of ellipse fitting algorithm (enabled/disabled). One value per eye.
    bool ellipse_fitting_[2]{};
    // Set to true if initialize() method was run. False otherwise.
    bool initialized_{false};
    // Number of frames analyzed in analyzeNextFrame() since the object creation.
    int frame_counter_{0};
    // Objects writing raw camera images to video output. One per eye.
    cv::VideoWriter output_video_[2]{};
    // Objects writing images with UI features to video output. One per eye.
    cv::VideoWriter output_video_ui_[2]{};

    // Type of visualization currently shown in the window.
    VisualizationType visualization_type_{};
    // The ids of the enabled cameras (0 - left, 1 - right).
    std::vector<int> camera_ids_{};

    // Objects writing raw camera images to output, captured through a remote app.
    cv::VideoWriter eye_video_[2]{};
    // Objects for writing external data to file, synchronized with eye_video_.
    std::ofstream eye_data_{};
    // Counts number of frames and rows written to eye_video_ and eye_data_.
    int eye_frame_counter_{};

    // Disables access to eye-tracking components during transformation calibration.
    std::mutex per_user_transformation_blocker_{};

    cv::Mat M_left_{};
    cv::Mat M_right_{};
};

} // namespace et

#endif //HDRMFS_EYE_TRACKER_EYE_TRACKER_HPP
