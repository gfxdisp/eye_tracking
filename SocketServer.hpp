#ifndef EYE_TRACKER_SERVER_H
#define EYE_TRACKER_SERVER_H

#include "EyeTracker.hpp"
#include "FeatureDetector.hpp"

#include <netinet/in.h>
#include <opencv2/opencv.hpp>
#include <vector>

namespace et {
class SocketServer {
public:
    SocketServer(EyeTracker *tracker, FeatureDetector *feature_detector);

    void startServer();
    void openSocket();
    void closeSocket();

    bool finished{false};

private:
    EyeTracker *eye_tracker_{};
    FeatureDetector *feature_detector_{};

    cv::Vec3d eye_position_{};
    cv::Vec3f gaze_direction_{};

    EyeData eye_data_{};

    float pupil_diameter_{};
    cv::Point2f pupil_location_{};
    std::vector<cv::Point2f> glint_locations_{};

    sockaddr_in address_{};
    int server_handle_{};
    int socket_handle_{-1};
};
}// namespace et

#endif
