#ifndef HDRMFS_EYE_TRACKER_FRAMEWORK_HPP
#define HDRMFS_EYE_TRACKER_FRAMEWORK_HPP

#include <eye_tracker/Visualizer.hpp>
#include <eye_tracker/optimizers/FineTuner.hpp>

#include <fstream>
#include <vector>

namespace et {
    enum class VisualizationType {
        DISABLED,
        CAMERA_IMAGE,
        THRESHOLD_PUPIL,
        THRESHOLD_GLINTS
    };

    struct EyeDataToSend {
        cv::Point3d cornea_centre;
        cv::Point3d eye_centre;
        cv::Point2d pupil;
        cv::Vec3d gaze_direction;
        double pupil_diameter;
        int frame_num;
    };

    /**
     * @brief The Framework class is the main class of the eye tracker. It is responsible for the initialization of the different components (image provider, feature detector, visualizer, fine tuner) and the analysis of the camera images.
     */
    class Framework {
    public:
        Framework(int camera_id, bool headless);

        virtual ~Framework();

        /**
         * Analyzes the next frame of the camera image. The frame is grabbed from the image provider, preprocessed, and the features are detected. The eye estimator is then used to estimate the eye position.
         * @return True if the frame was successfully captured and analyzed, false otherwise.
         */
        virtual bool analyzeNextFrame();

        void getEyeDataPackage(EyeDataToSend& eye_data_package) const;

        /**
         * Starts the recording of the eye video. The video will be saved in the "results/" directory.
         * @param name The name of the video file.
         * @param record_ui If true, two videos will be recorded: one with the raw eye video and one with the UI overlay, showing the detected features.
         */
        void startRecording(std::string const& name = "", bool record_ui = true);

        /**
         * Stops the recording of the eye video.
         */
        void stopRecording();

        /**
         * Captures an image from the camera and saves it in the "images/" directory.
         */
        void captureCameraImage() const;

        /**
         * Updates the UI with the latest information about the eye position, the cornea position, and the gaze direction.
         */
        void updateUi();

        /**
         * Pauses the image update in the UI. This is useful for increasing the performance of the eye tracker.
         */
        void disableImageUpdate();

        /**
         * Shows current image from the video feed.
         */
        void switchToCameraImage();

        /**
         * Shows the thresholded image which is used for pupil detection.
         */
        void switchToPupilThreshImage();

        /**
         * Shows the thresholded image which is used for glint detection.
         */
        void switchToGlintThreshImage();

        /**
         * Closes the application if the user presses the "q" key.
         * @return True if the application should close, false otherwise.
         */
        bool shouldAppClose() const;

        /**
         * Starts the calibration process. Before it is stopped, all estimations (i.e. eye centre, cornea centre, gaze direction) are continuously saved along with the timestamp.
         * @param name Starting the calibration process will also start the recording of the eye video. This parameter is used to name the video file.
         */
        void startCalibration(std::string const& name = "");

        /**
         * Stops the calibration process. Based on the collected data during the calibration process, the fine tuning is calculated. From that point on, the eye estimator will use the fine tuned parameters for the eye estimation.
         * @param calibration_output The output of the calibration process. It should contain the list of real eye positions and the marker positions and specific timestamps.
         */
        void stopCalibration(const CalibrationOutput& calibration_output);

        void stopEyeVideoRecording();

        virtual cv::Point2d getMarkerPosition();

        static std::mutex mutex;
        std::shared_ptr<EyeEstimator> eye_estimator_{};

    protected:
        std::shared_ptr<ImageProvider> image_provider_{};
        std::shared_ptr<FeatureAnalyser> feature_detector_{};
        std::shared_ptr<Visualizer> visualizer_{};
        std::shared_ptr<FineTuner> fine_tuner_{};

        EyeImage analyzed_frame_{};

        cv::VideoWriter output_video_{};
        cv::VideoWriter output_video_ui_{};
        std::string output_video_name_{};
        int output_video_frame_counter_{};

        VisualizationType visualization_type_{};

        cv::VideoWriter eye_video_{};

        std::ofstream eye_data_{};

        std::vector<CalibrationInput> calibration_input_{};

        bool calibration_running_{};
        std::chrono::high_resolution_clock::time_point calibration_start_time_{};

        int camera_id_{};

        bool features_found_{};
    };
} // namespace et

#endif //HDRMFS_EYE_TRACKER_FRAMEWORK_HPP
