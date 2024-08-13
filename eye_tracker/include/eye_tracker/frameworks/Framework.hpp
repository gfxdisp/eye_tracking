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

    class Framework {
    public:
        Framework(int camera_id, bool headless);

        virtual ~Framework();

        virtual bool analyzeNextFrame();

        void getEyeDataPackage(EyeDataToSend& eye_data_package) const;

        void startRecording(std::string const& name = "", bool record_ui = true);

        void stopRecording();

        void captureCameraImage() const;

        void updateUi();

        void disableImageUpdate();

        void switchToCameraImage();

        void switchToPupilThreshImage();

        void switchToGlintThreshImage();

        bool shouldAppClose() const;

        void startCalibration(std::string const& name = "");

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

        bool calibration_running_{false};
        std::chrono::high_resolution_clock::time_point calibration_start_time_{};

        int camera_id_{};

        bool features_found_{false};
    };
} // namespace et

#endif //HDRMFS_EYE_TRACKER_FRAMEWORK_HPP
