#include "EyeTracker.hpp"

namespace et {
void EyeTracker::initialize(ImageProvider *image_provider) {
    image_provider = image_provider_;
    for (int i = 0; i < 2; i++) {
        feature_detectors_[i].initialize(i);
        eye_estimators_[i].initialize(i);
    }
}

void EyeTracker::preprocessGlintEllipse(const cv::Mat &image, int camera_id)
{
    feature_detectors_[camera_id].preprocessGlintEllipse(image);
}

void EyeTracker::preprocessIndivGlints(const cv::Mat &image, int camera_id)
{
    feature_detectors_[camera_id].preprocessIndivGlints(image);
}

bool EyeTracker::findPupil(int camera_id) {
    return feature_detectors_[camera_id].findPupil();
}

bool EyeTracker::findGlints(int camera_id) {
    return feature_detectors_[camera_id].findGlints();
}

bool EyeTracker::findEllipse(const cv::Point2f &pupil, int camera_id) {
    return feature_detectors_[camera_id].findEllipse(pupil);
}

cv::Point2f EyeTracker::getPupil(int camera_id) {
    return feature_detectors_[camera_id].getPupil();
}

void EyeTracker::getPupil(cv::Point2f &pupil, int camera_id) {
    feature_detectors_[camera_id].getPupil(pupil);
}

void EyeTracker::getPupilFiltered(cv::Point2f &pupil, int camera_id) {
    feature_detectors_[camera_id].getPupilFiltered(pupil);
}

int EyeTracker::getPupilRadius(int camera_id) {
    return feature_detectors_[camera_id].getPupilRadius();
}

void EyeTracker::getPupilDiameter(float &pupil_diameter, int camera_id) {
    eye_estimators_[camera_id].getPupilDiameter(pupil_diameter);
}

void EyeTracker::getGazeDirection(cv::Vec3f &gaze_direction, int camera_id) {
    eye_estimators_[camera_id].getGazeDirection(gaze_direction);
}

void EyeTracker::setGazeBufferSize(uint8_t value) {
    for (auto & feature_detector : feature_detectors_) {
        feature_detector.setGazeBufferSize(value);
    }
}

std::vector<cv::Point2f> *EyeTracker::getGlints(int camera_id) {
    return feature_detectors_[camera_id].getGlints();
}

void EyeTracker::getPupilGlintVector(cv::Vec2f &pupil_glint_vector, int camera_id) {
    feature_detectors_[camera_id].getPupilGlintVector(pupil_glint_vector);
}

void EyeTracker::getPupilGlintVectorFiltered(cv::Vec2f &pupil_glint_vector,
                                             int camera_id) {
    feature_detectors_[camera_id].getPupilGlintVectorFiltered(pupil_glint_vector);
}


cv::RotatedRect EyeTracker::getEllipse(int camera_id) {
    return feature_detectors_[camera_id].getEllipse();
}

cv::Mat EyeTracker::getThresholdedPupilImage(int camera_id) {
    return feature_detectors_[camera_id].getThresholdedPupilImage();
}

cv::Mat EyeTracker::getThresholdedGlintsImage(int camera_id) {
    return feature_detectors_[camera_id].getThresholdedGlintsImage();
}

void EyeTracker::updateGazeBuffer(int camera_id) {
    feature_detectors_[camera_id].updateGazeBuffer();
}

void EyeTracker::getEyeFromModel(cv::Point2f pupil_pix_position,
                                 std::vector<cv::Point2f> *glint_pix_positions,
                                 int pupil_radius, int camera_id) {
    eye_estimators_[camera_id].getEyeFromModel(
        pupil_pix_position, glint_pix_positions, pupil_radius);
}

void EyeTracker::getEyeFromPolynomial(cv::Point2f pupil_pix_position,
                                      cv::RotatedRect ellipse, int camera_id) {
    eye_estimators_[camera_id].getEyeFromPolynomial(pupil_pix_position,
                                                    ellipse);
}

cv::Point2f EyeTracker::getCorneaCurvaturePixelPosition(int camera_id) {
    return eye_estimators_[camera_id].getCorneaCurvaturePixelPosition();
}

cv::Point2f EyeTracker::getEyeCentrePixelPosition(int camera_id) {
    return eye_estimators_[camera_id].getEyeCentrePixelPosition();
}

void EyeTracker::getEyeCentrePosition(cv::Vec3d &eye_centre, int camera_id) {
    eye_estimators_[camera_id].getEyeCentrePosition(eye_centre);
}

void EyeTracker::getCorneaCurvaturePosition(cv::Vec3d &cornea_centre,
                                            int camera_id) {
    eye_estimators_[camera_id].getCorneaCurvaturePosition(cornea_centre);
}

} // namespace et