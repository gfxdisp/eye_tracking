#ifndef HDRMFS_EYE_TRACKER_FRAMEWORK_HPP
#define HDRMFS_EYE_TRACKER_FRAMEWORK_HPP

#include "eye_tracker/input/ImageProvider.hpp"
#include "eye_tracker/image/FeatureAnalyser.hpp"
#include "eye_tracker/Visualizer.hpp"
#include "eye_tracker/image/preprocess/ImagePreprocessor.hpp"
#include "eye_tracker/image/temporal_filter/TemporalFilterer.hpp"
#include "eye_tracker/eye/EyeEstimator.hpp"
#include "eye_tracker/optimizers/MetaModel.hpp"

#include <fstream>
#include <vector>

namespace et
{

/**
 * Type of visualization shown in the application window.
 */
    enum class VisualizationType
    {
        // Visualization disabled, only prints framerate to console.
        DISABLED, // Shows raw image from the camera.
        CAMERA_IMAGE, // Shows video feed image after thresholding that is used to calculate
        // pupil position.
        THRESHOLD_PUPIL, // Shows video feed image after thresholding that is used to calculate
        // glints positions.
        THRESHOLD_GLINTS
    };

    struct EyeDataToSend
    {
        cv::Point3d cornea_centre;
        cv::Point3d eye_centre;
        cv::Point2d pupil;
        cv::RotatedRect ellipse;
        cv::Vec2d pupil_glint_vector;
        double pupil_diameter;
        cv::Vec3d gaze_direction;
        int frame_num;
    };

    struct EyeDataToReceive
    {
        std::vector<cv::Point3d> front_corners = std::vector<cv::Point3d>(4);
        std::vector<cv::Point3d> back_corners = std::vector<cv::Point3d>(4);
        cv::Point3d marker_position;
    };

/**
 * Pulls images from the camera (or video/folder), detects
 * eye features, and estimates eye position. It also controls the state of the
 * window showing the detection results.
 */
    class Framework
    {
    public:
        /**
         * Initializes all class components. Needs to be run before any
         * other class method.
         * @param image_provider pointer to the object used to serve as a video feed.
         * @param settings_path Path to a folder containing all settings files.
         * @param input_path Path to a folder with csv files.
         * @param enabled_cameras 2-element boolean vector signifying, whether
         * left and right cameras (in this order) are enabled .
         * @param enabled_ellipse_fitting 2-element boolean vector signifying, whether
         * left and right cameras (in this order) use ellipse polynomial fitting as
         * eye-detection algorithm. False values would use the model based
         * on 2006 Guestrin et al. model.
         * @param enabled_temporal_filter 2-element boolean vector signifying, whether to
         * enable Kalman filtering for feature and eye position estimation.
         * @param enabled_template_matching 2-element boolean vector signifying,
         * whether to use template matching for finding glints.
         * @param headless If true, the application will not show any windows.
         * @param enabled_undistort 2-element boolean vector signifying, whether the images require applying undistortion.
         */
        Framework(int camera_id, bool headless);

        ~Framework();

        /**
         * Takes the next frame from the image provider, find its features, and calculates
         * eye parameters.
         * @return True, if the next frame was successfully captured from the image
         * provider. Otherwise, it returns false.
         */
        bool analyzeNextFrame();

        void getEyeDataPackage(EyeDataToSend &eye_data_package);

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
        double getAvgFramerate();

        /**
         * Starts a recording of eyes through a remote application.
         * @param folder_name Folder to which the videos will be saved.
         */
        std::string startEyeVideoRecording(const std::string &folder_name);

        /**
         * Stops a video recording started through startEyeVideoRecording().
         */
        void stopEyeVideoRecording();

        void addEyeVideoData(const EyeDataToReceive &eye_data);

        cv::Point3d setMetaModel(const std::string &input_path);

        bool wereFeaturesFound();

        static std::mutex mutex;

    protected:
        // Object serving as a video feed.
        std::shared_ptr<ImageProvider> image_provider_{};
        // Object estimating image features (pupils and glints).
        std::shared_ptr<FeatureAnalyser> feature_detector_{};
        // Object showing windows with output.
        std::shared_ptr<Visualizer> visualizer_{};

        std::shared_ptr<EyeEstimator> eye_estimator_{};

        std::shared_ptr<MetaModel> meta_model_{};

        // Path to a folder with all configuration files.
        std::string settings_path_{};

        // Raw images retrieved directly from image_provider_. Two images per eye: one for pupil, one for glints.
        EyeImage analyzed_frame_{};
        // Status of ellipse fitting algorithm (enabled/disabled). One value per eye.
        bool ellipse_fitting_{};
        // Set to true if initialize() method was run. False otherwise.
        bool initialized_{false};
        // Number of frames analyzed in analyzeNextFrame() since the object creation.
        int frame_counter_{0};
        // Objects writing raw camera images to video output. One per eye.
        cv::VideoWriter output_video_{};
        // Objects writing images with UI features to video output. One per eye.
        cv::VideoWriter output_video_ui_{};

        // Type of visualization currently shown in the window.
        VisualizationType visualization_type_{};

        // Objects writing raw camera images to output, captured through a remote app.
        cv::VideoWriter eye_video_{};
        // Objects for writing external data to file, synchronized with eye_video_.
        std::ofstream eye_data_{};
        // Counts number of frames and rows written to eye_video_ and eye_data_.
        int eye_frame_counter_{};

        int camera_id_{};

        bool features_found_{false};

    };

} // namespace et

#endif //HDRMFS_EYE_TRACKER_FRAMEWORK_HPP
