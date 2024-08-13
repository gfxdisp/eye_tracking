#ifndef HDRMFS_EYE_TRACKER_SOCKET_SERVER_HPP
#define HDRMFS_EYE_TRACKER_SOCKET_SERVER_HPP

#include <eye_tracker/frameworks/Framework.hpp>

#include <netinet/in.h>

#include <vector>
#include <thread>

namespace et {
    class SocketServer {
    public:
        SocketServer(const std::shared_ptr<Framework>& eye_tracker_left, const std::shared_ptr<Framework>& eye_tracker_right);

        ~SocketServer();

        void startServer();

        void openSocket();

        void closeSocket() const;

        bool isClientConnected() const;

        bool finished{false};

    private:
        bool sendAll(void* input, size_t bytes_count) const;

        bool readAll(void* output, size_t bytes_count) const;

        static constexpr int MSG_CLOSE_CONNECTION = 0;

        static constexpr int MSG_GET_EYE_DATA = 1;

        static constexpr int MSG_START_CALIBRATION = 2;

        static constexpr int MSG_STOP_CALIBRATION = 3;

        static constexpr int MSG_START_RECORDING = 4;

        static constexpr int MSG_STOP_RECORDING = 5;

        std::shared_ptr<Framework> eye_trackers_[2]{};

        cv::Vec3d gaze_direction_{};

        cv::Vec2d pupil_glint_vector_{};

        std::vector<cv::Point2d> glint_locations_{};

        EyeDataToSend eye_data_to_send_{};
        CalibrationOutput online_calibration_data_received_{};

        sockaddr_in address_{};

        int server_handle_{};

        int socket_handle_{-1};
        char message_buffer_[10000]{};

        std::thread server_thread_{};
    };
} // namespace et

#endif //HDRMFS_EYE_TRACKER_SOCKET_SERVER_HPP
