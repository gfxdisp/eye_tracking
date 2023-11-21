#ifndef HDRMFS_EYE_TRACKER_SOCKET_SERVER_HPP
#define HDRMFS_EYE_TRACKER_SOCKET_SERVER_HPP

#include "eye_tracker/frameworks/Framework.hpp"

#include <netinet/in.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <thread>

namespace et
{


/**
 * Communicates with a client to exchange eye parameters and tracking
 * parameters.
 */
    class SocketServer
    {
    public:
        /**
         * Assigns loaded EyeTracker object to the Socket server.
         * @param eye_tracker
         */
        SocketServer(std::shared_ptr<Framework> eye_tracker_left, std::shared_ptr<Framework> eye_tracker_right);

        ~SocketServer();

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
        static constexpr int MSG_GET_EYE_DATA = 1;
        // Starts recording of the video during gaze calibration.
        static constexpr int MSG_START_CALIBRATION = 2;
        // Stops recording of the video during gaze calibration.
        static constexpr int MSG_STOP_CALIBRATION = 3;
        // Add data to the video and csv.
        static constexpr int MSG_ADD_CALIBRATION_DATA = 4;
        // Saves current camera image to png file along with the calibrated eye positions.
        static constexpr int MSG_SET_META_MODEL = 5;

        std::shared_ptr<Framework> eye_trackers_[2]{};

        // Gaze direction to be sent.
        cv::Vec3d gaze_direction_{};

        // Pupil-glint vector to be sent.
        cv::Vec2d pupil_glint_vector_{};
        // Vector of glint locations to be sent.
        std::vector<cv::Point2d> glint_locations_{};

        EyeDataToSend eye_data_to_send_{};
        EyeDataToReceive eye_data_to_receive_{};

        // Address of an opened socket server.
        sockaddr_in address_{};
        // Handle of the opened server connection.
        int server_handle_{};
        // Handle of the opened socket.
        int socket_handle_{-1};
        char message_buffer_[10000]{};

        std::thread server_thread_{};
    };
} // namespace et

#endif //HDRMFS_EYE_TRACKER_SOCKET_SERVER_HPP
