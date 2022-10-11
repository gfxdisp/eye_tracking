#ifndef HDRMFS_EYE_TRACKER_EYETRACKER_HPP
#define HDRMFS_EYE_TRACKER_EYETRACKER_HPP

#include "EyeEstimator.hpp"
#include "FeatureDetector.hpp"
#include "Visualizer.hpp"

#include <vector>

namespace et {

enum class VisualizationType {
    DISABLED,
    CAMERA_IMAGE,
    THRESHOLD_PUPIL,
    THRESHOLD_GLINTS
};

class EyeTracker {
public:
    void initialize(ImageProvider *image_provider, bool ellipse_fitting[]);
    bool analyzeNextFrame();
    void getPupil(cv::Point2f &pupil, int camera_id);
    void getPupilFiltered(cv::Point2f &pupil, int camera_id);
    void getPupilDiameter(float &pupil_diameter, int camera_id);
    void getGazeDirection(cv::Vec3f &gaze_direction, int camera_id);
    void setGazeBufferSize(uint8_t value);
    void getPupilGlintVector(cv::Vec2f &pupil_glint_vector, int camera_id);
    void getPupilGlintVectorFiltered(cv::Vec2f &pupil_glint_vector, int camera_id);

    void getEyeCentrePosition(cv::Vec3d &eye_centre, int camera_id);
    void getCorneaCurvaturePosition(cv::Vec3d &cornea_centre, int camera_id);

    void logEyeFeatures(std::ostream &output);

    void startVideoRecording();

    void stopVideoRecording();

    void captureCameraImage();

    void updateUi();

    void disableImageUpdate();

    void switchToCameraImage();

    void switchToPupilThreshImage();

    void switchToGlintThreshImage();

    bool shouldAppClose();

    float getAvgFramerate();


private:
    ImageProvider *image_provider_{};
    FeatureDetector feature_detectors_[2]{};
    EyeEstimator eye_estimators_[2]{};
    Visualizer visualizer_[2]{};

    cv::Mat analyzed_frame_[2]{};
    bool ellipse_fitting_[2]{};
    bool initialized_{false};
    int frame_counter_{0};
    cv::VideoWriter output_video_[2]{};
    cv::VideoWriter output_video_ui_[2]{};
    VisualizationType visualization_type_{};
};

} // namespace et

#endif //HDRMFS_EYE_TRACKER_EYETRACKER_HPP
