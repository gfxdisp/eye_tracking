#ifndef EYE_TRACKER_SERVER_H
#define EYE_TRACKER_SERVER_H

#include "EyeEstimator.hpp"
#include "EyeTracker.hpp"
#include "FeatureDetector.hpp"

#include <netinet/in.h>
#include <opencv2/opencv.hpp>
#include <vector>

namespace et {
class SocketServer {
public:
    explicit SocketServer(EyeTracker *eye_tracker);

    void startServer();
    void openSocket();
    void closeSocket();

    bool isClientConnected();

    bool finished{false};

private:
    EyeTracker *eye_tracker_{};

    cv::Vec3d eye_position_{};
    cv::Vec3f gaze_direction_{};

    float pupil_diameter_{};
    cv::Point2f pupil_location_{};
    cv::Vec2f pupil_glint_vector_{};
    std::vector<cv::Point2f> glint_locations_{};

    static constexpr std::string_view CAMERA_MAPPING[] = {"left", "right"};

    sockaddr_in address_{};
    int server_handle_{};
    int socket_handle_{-1};
};
}// namespace et

#endif
