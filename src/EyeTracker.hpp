#ifndef HDRMFS_EYE_TRACKER_EYETRACKER_HPP
#define HDRMFS_EYE_TRACKER_EYETRACKER_HPP

#include "EyeEstimator.hpp"
#include "FeatureDetector.hpp"

#include <vector>

namespace et {

class EyeTracker {
public:
    void initialize(ImageProvider *image_provider);
    bool findPupil(const cv::Mat &image, int camera_id);
    bool findGlints(const cv::Mat &image, int camera_id);
    bool findEllipse(const cv::Mat &image, const cv::Point2f &pupil, int camera_id);
    cv::Point2f getPupil(int camera_id);
    void getPupil(cv::Point2f &pupil, int camera_id);
    void getPupilFiltered(cv::Point2f &pupil, int camera_id);
    int getPupilRadius(int camera_id);
    void getPupilDiameter(float &pupil_diameter, int camera_id);
    void getGazeDirection(cv::Vec3f &gaze_direction, int camera_id);

    void setGazeBufferSize(uint8_t value);

    std::vector<cv::Point2f> *getGlints(int camera_id);
    void getPupilGlintVector(cv::Vec2f &pupil_glint_vector, int camera_id);
    void getPupilGlintVectorFiltered(cv::Vec2f &pupil_glint_vector, int camera_id);
    cv::RotatedRect getEllipse(int camera_id);

    cv::Mat getThresholdedPupilImage(int camera_id);

    cv::Mat getThresholdedGlintsImage(int camera_id);

    void updateGazeBuffer(int camera_id);

    void getEyeFromModel(cv::Point2f pupil_pix_position,
                         std::vector<cv::Point2f> *glint_pix_positions,
                         int pupil_radius, int camera_id);

    void getEyeFromPolynomial(cv::Point2f pupil_pix_position,
                              cv::RotatedRect ellipse, int camera_id);

    cv::Point2f getCorneaCurvaturePixelPosition(int camera_id);

    cv::Point2f getEyeCentrePixelPosition(int camera_id);

    void getEyeCentrePosition(cv::Vec3d &eye_centre, int camera_id);
    void getCorneaCurvaturePosition(cv::Vec3d &cornea_centre, int camera_id);


private:
    ImageProvider *image_provider_{};
    FeatureDetector feature_detectors_[2]{};
    EyeEstimator eye_estimators_[2]{};
};

} // namespace et

#endif //HDRMFS_EYE_TRACKER_EYETRACKER_HPP
