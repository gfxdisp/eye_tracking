#include "SocketServer.hpp"

#include <arpa/inet.h>
#include <chrono>
#include <fcntl.h>
#include <iostream>
#include <thread>
#include <unistd.h>

using cv::Vec3d;

namespace et {
SocketServer::SocketServer(EyeTracker *eye_tracker)
    : eye_tracker_(eye_tracker) {
}

void SocketServer::startServer() {
    int opt{1};

    if ((server_handle_ = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        std::clog << "Failed to create a socket.\n";
        return;
    }

    fcntl(server_handle_, F_SETFL, O_NONBLOCK);

    if (setsockopt(server_handle_, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT,
                   &opt, sizeof(opt))) {
        std::clog << "Failed to set socket options.\n";
        close(server_handle_);
        return;
    }
    address_.sin_family = AF_INET;
    address_.sin_addr.s_addr = inet_addr("127.0.0.1");
    address_.sin_port = htons(8080);

    if (bind(server_handle_, (sockaddr *) &address_, sizeof(address_)) < 0) {
        std::clog << "Failed to bind a socket address.\n";
        close(server_handle_);
        return;
    }
    if (listen(server_handle_, 3) < 0) {
        std::clog << "Failed to start socket listening.\n";
        close(server_handle_);
    }
}

void SocketServer::openSocket() {
    std::clog << "Socket opened.\n";
    char buffer[1]{1};
    while (!finished) {
        int address_length = sizeof(address_);
        while (socket_handle_ < 0 && !finished) {
            socket_handle_ =
                accept(server_handle_, (struct sockaddr *) &address_,
                       (socklen_t *) &address_length);
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        while (!finished) {
            if (!readAll(buffer, 1)) {
                break;
            }
            if (buffer[0] == MSG_CLOSE_CONNECTION) {
                finished = true;
            } else if (buffer[0] == MSG_GET_CORNEA_CENTRE_POS) {
                for (int i = 0; i < 2; i++) {
                    eye_tracker_->getCorneaCentrePosition(eye_position_, i);
                    if (!sendAll(&eye_position_, sizeof(eye_position_))) {
                        break;
                    }
                }
            } else if (buffer[0] == MSG_GET_EYE_CENTRE_POS) {
                for (int i = 0; i < 2; i++) {
                    eye_tracker_->getEyeCentrePosition(eye_position_, i);
                    if (!sendAll(&eye_position_, sizeof(eye_position_))) {
                        break;
                    }
                }
            } else if (buffer[0] == MSG_GET_PUPIL_GLINT_VEC) {
                for (int i = 0; i < 2; i++) {
                    eye_tracker_->getPupilGlintVector(pupil_glint_vector_, i);
                    if (!sendAll(&pupil_glint_vector_,
                                 sizeof(pupil_glint_vector_))) {
                        break;
                    }
                }
            } else if (buffer[0] == MSG_GET_PUPIL_DIAM) {
                for (int i = 0; i < 2; i++) {
                    eye_tracker_->getPupilDiameter(pupil_diameter_, i);
                    if (!sendAll(&pupil_diameter_, sizeof(pupil_diameter_))) {
                        break;
                    }
                }
            } else if (buffer[0] == MSG_GET_GAZE_DIR) {
                for (int i = 0; i < 2; i++) {
                    eye_tracker_->getGazeDirection(gaze_direction_, i);
                    if (!sendAll(&gaze_direction_, sizeof(gaze_direction_))) {
                        break;
                    }
                }

            } else if (buffer[0] == MSG_SET_MOVING_AVG_SIZE) {
                uint8_t moving_average_size{};
                if (!readAll(&moving_average_size,
                             sizeof(moving_average_size))) {
                    break;
                }
                eye_tracker_->setGazeBufferSize(moving_average_size);
            } else if (buffer[0] == MSG_GET_PUPIL_GLINT_VEC_FLTR) {
                for (int i = 0; i < 2; i++) {
                    eye_tracker_->getPupilGlintVectorFiltered(
                        pupil_glint_vector_, i);
                    if (!sendAll(&pupil_glint_vector_,
                                 sizeof(pupil_glint_vector_))) {
                        break;
                    }
                }
            } else if (buffer[0] == MSG_GET_PUPIL) {
                for (int i = 0; i < 2; i++) {
                    eye_tracker_->getPupil(pupil_location_, i);
                    if (!sendAll(&pupil_location_, sizeof(pupil_location_))) {
                        break;
                    }
                }
            } else if (buffer[0] == MSG_GET_PUPIL_FLTR) {
                for (int i = 0; i < 2; i++) {
                    eye_tracker_->getPupilFiltered(pupil_location_, i);
                    if (!sendAll(&pupil_location_, sizeof(pupil_location_))) {
                        break;
                    }
                }
            } else if (buffer[0] == MSG_START_EYE_VIDEO) {
                int path_length{};
                if (!readAll(&path_length, sizeof(path_length))) {
                    break;
                }
                if (!readAll(message_buffer_, path_length)) {
                    break;
                }
                message_buffer_[path_length] = '\0';
                eye_tracker_->startEyeVideoRecording(message_buffer_);
            } else if (buffer[0] == MSG_STOP_EYE_VIDEO) {
                eye_tracker_->stopEyeVideoRecording();
            } else if (buffer[0] == MSG_SAVE_EYE_DATA) {
                int message_length{};
                if (!readAll(&message_length, sizeof(message_length))) {
                    break;
                }
                if (!readAll(message_buffer_, message_length)) {
                    break;
                }
                message_buffer_[message_length] = '\0';
                eye_tracker_->saveEyeData(message_buffer_);
            }
        }
        close(socket_handle_);
    }
    close(socket_handle_);
    std::cout << "Socket closed.\n";
}

void SocketServer::closeSocket() const {
    close(server_handle_);
}

bool SocketServer::isClientConnected() const {
    return socket_handle_ > -1 && !finished;
}

bool SocketServer::sendAll(void *input, size_t bytes_count) const {
    size_t size_to_send = bytes_count;
    size_t sent_bytes = 0;
    while (sent_bytes < size_to_send) {
        ssize_t new_bytes = send(socket_handle_, (char *) input + sent_bytes,
                                 size_to_send - sent_bytes, 0);
        if (new_bytes < 0) {
            return false;
        }
        sent_bytes += new_bytes;
    }
    return true;
}

bool SocketServer::readAll(void *output, size_t bytes_count) const {
    size_t size_to_read = bytes_count;
    size_t read_bytes = 0;
    while (read_bytes < size_to_read) {
        ssize_t new_bytes = read(socket_handle_, (char *) output + read_bytes,
                                 size_to_read - read_bytes);
        if (new_bytes < 0) {
            return false;
        }
        read_bytes += new_bytes;
    }
    return true;
}

} // namespace et
