#ifndef HDRMFS_EYE_TRACKER_SOCKET_SERVER_HPP
#define HDRMFS_EYE_TRACKER_SOCKET_SERVER_HPP

#include "EyeEstimator.hpp"
#include "EyeTracker.hpp"
#include "FeatureDetector.hpp"

#include <netinet/in.h>
#include <opencv2/opencv.hpp>
#include <vector>

namespace et {
/**
 * Communicates with a client to exchange eye parameters and tracking
 * parameters.
 */
class SocketServer {
public:
    /**
     * Assigns loaded EyeTracker object to the Socket server.
     * @param eye_tracker
     */
    explicit SocketServer(EyeTracker *eye_tracker);

    /**
     * Opens a connection and allows clients to connect.
     */
    void startServer();
    /**
     * Listens to incoming client connections and manages their messages.
     */
    void openSocket();
    /**
     * Closes a socket connection.
     */
    void closeSocket() const;

    /**
     * Checks if all the clients disconnected.
     * @return True if client closed their application. False otherwise.
     */
    [[nodiscard]] bool isClientConnected() const;

    // Set to true after sending a finish request from the client, running out
    // of camera images, or manually closing the application.
    bool finished{false};

private:
    bool sendAll(void *input, size_t bytes_count) const;

    bool readAll(void *output, size_t bytes_count) const;

    // Request code to close connection and the application.
    static constexpr int MSG_CLOSE_CONNECTION = 0;
    // Request code to send cornea centre position in camera space from both eyes.
    static constexpr int MSG_GET_CORNEA_CENTRE_POS = 1;
    // Request code to send eye centre position in camera space from both eyes.
    static constexpr int MSG_GET_EYE_CENTRE_POS = 2;
    // Request code to send pupil-glint vector in image space from both eyes.
    static constexpr int MSG_GET_PUPIL_GLINT_VEC = 3;
    // Request code to send pupil diameter in millimeters from both eyes.
    static constexpr int MSG_GET_PUPIL_DIAM = 4;
    // Request code to send gaze direction in camera space from both eyes.
    static constexpr int MSG_GET_GAZE_DIR = 5;
    // Request code to set a new size of the buffer for gaze tracking.
    static constexpr int MSG_SET_MOVING_AVG_SIZE = 6;
    // Request code to send pupil-glint vector in image space from both eyes
    // averaged across multiple frames.
    static constexpr int MSG_GET_PUPIL_GLINT_VEC_FLTR = 7;
    // Request code to send pupil position in image space from both eyes.
    static constexpr int MSG_GET_PUPIL = 8;
    // Request code to send pupil position in image space from both eyes
    // averaged across multiple frames.
    static constexpr int MSG_GET_PUPIL_FLTR = 9;
    // Starts recording of the video during gaze calibration.
    static constexpr int MSG_START_EYE_VIDEO = 10;
    // Stops recording of the video during gaze calibration.
    static constexpr int MSG_STOP_EYE_VIDEO = 11;
    // Saves current camera image to png file along with the calibrated eye positions.
    static constexpr int MSG_SAVE_EYE_DATA = 12;
    // Calibrates the transformation between Blender and real data.
    static constexpr int MSG_CALIBRATE_TRANSFORM = 13;
    EyeTracker *eye_tracker_{};

    // Eye position to be sent.
    cv::Vec3d eye_position_{};
    // Gaze direction to be sent.
    cv::Vec3f gaze_direction_{};

    // Pupil diameter in millimeters to be sent.
    float pupil_diameter_{};
    // Pupil location in image space to be spent.
    cv::Point2f pupil_location_{};
    // Pupil-glint vector to be sent.
    cv::Vec2f pupil_glint_vector_{};
    // Vector of glint locations to be sent.
    std::vector<cv::Point2f> glint_locations_{};

    // Address of an opened socket server.
    sockaddr_in address_{};
    // Handle of the opened server connection.
    int server_handle_{};
    // Handle of the opened socket.
    int socket_handle_{-1};
    char message_buffer_[10000]{};
};
} // namespace et

#endif //HDRMFS_EYE_TRACKER_SOCKET_SERVER_HPP
