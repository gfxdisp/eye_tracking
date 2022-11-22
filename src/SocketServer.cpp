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
            auto read_bytes = (int) read(socket_handle_, buffer, 1);
            if (read_bytes != 1) {
                break;
            }
            if (buffer[0] == MSG_CLOSE_CONNECTION) {
                finished = true;
            } else if (buffer[0] == MSG_GET_CORNEA_CENTRE_POS) {
                for (int i = 0; i < 2; i++) {
                    eye_tracker_->getCorneaCentrePosition(eye_position_, i);
                    uint32_t sent{0};
                    uint32_t size_to_send{sizeof(eye_position_)};
                    while (sent < size_to_send) {
                        ssize_t new_bytes{send(socket_handle_,
                                               (char *) &eye_position_ + sent,
                                               size_to_send - sent, 0)};
                        if (new_bytes < 0) {
                            break;
                        }
                        sent += new_bytes;
                    }
                }
            } else if (buffer[0] == MSG_GET_EYE_CENTRE_POS) {
                for (int i = 0; i < 2; i++) {
                    eye_tracker_->getEyeCentrePosition(eye_position_, i);
                    uint32_t sent{0};
                    uint32_t size_to_send{sizeof(eye_position_)};
                    while (sent < size_to_send) {
                        ssize_t new_bytes{send(socket_handle_,
                                               (char *) &eye_position_ + sent,
                                               size_to_send - sent, 0)};
                        if (new_bytes < 0) {
                            break;
                        }
                        sent += new_bytes;
                    }
                }
            } else if (buffer[0] == MSG_GET_PUPIL_GLINT_VEC) {
                for (int i = 0; i < 2; i++) {
                    eye_tracker_->getPupilGlintVector(pupil_glint_vector_, i);
                    uint32_t sent{0};
                    uint32_t size_to_send{sizeof(pupil_glint_vector_)};
                    while (sent < size_to_send) {
                        ssize_t new_bytes{
                            send(socket_handle_,
                                 (char *) &pupil_glint_vector_ + sent,
                                 size_to_send - sent, 0)};
                        if (new_bytes < 0) {
                            break;
                        }
                        sent += new_bytes;
                    }
                }
            } else if (buffer[0] == MSG_GET_PUPIL_DIAM) {
                for (int i = 0; i < 2; i++) {
                    eye_tracker_->getPupilDiameter(pupil_diameter_, i);
                    uint32_t sent{0};
                    uint32_t size_to_send{sizeof(pupil_diameter_)};
                    while (sent < size_to_send) {
                        ssize_t new_bytes{send(socket_handle_,
                                               (char *) &pupil_diameter_ + sent,
                                               size_to_send - sent, 0)};
                        if (new_bytes < 0) {
                            break;
                        }
                        sent += new_bytes;
                    }
                }
            } else if (buffer[0] == MSG_GET_GAZE_DIR) {
                for (int i = 0; i < 2; i++) {
                    eye_tracker_->getGazeDirection(gaze_direction_, i);
                    uint32_t sent{0};
                    uint32_t size_to_send{sizeof(gaze_direction_)};
                    while (sent < size_to_send) {
                        ssize_t new_bytes{send(socket_handle_,
                                               (char *) &gaze_direction_ + sent,
                                               size_to_send - sent, 0)};
                        if (new_bytes < 0) {
                            break;
                        }
                        sent += new_bytes;
                    }
                }

            } else if (buffer[0] == MSG_SET_MOVING_AVG_SIZE) {
                uint8_t moving_average_size{};
                read_bytes =
                    (int) read(socket_handle_, &moving_average_size, 1);
                if (read_bytes != 1) {
                    break;
                }
                eye_tracker_->setGazeBufferSize(moving_average_size);
            } else if (buffer[0] == MSG_GET_PUPIL_GLINT_VEC_FLTR) {
                for (int i = 0; i < 2; i++) {
                    eye_tracker_->getPupilGlintVectorFiltered(
                        pupil_glint_vector_, i);
                    uint32_t sent{0};
                    uint32_t size_to_send{sizeof(pupil_glint_vector_)};
                    while (sent < size_to_send) {
                        ssize_t new_bytes{
                            send(socket_handle_,
                                 (char *) &pupil_glint_vector_ + sent,
                                 size_to_send - sent, 0)};
                        if (new_bytes < 0) {
                            break;
                        }
                        sent += new_bytes;
                    }
                }
            } else if (buffer[0] == MSG_GET_PUPIL) {
                for (int i = 0; i < 2; i++) {
                    eye_tracker_->getPupil(pupil_location_, i);
                    uint32_t sent{0};
                    uint32_t size_to_send{sizeof(pupil_location_)};
                    while (sent < size_to_send) {
                        ssize_t new_bytes{send(socket_handle_,
                                               (char *) &pupil_location_ + sent,
                                               size_to_send - sent, 0)};
                        if (new_bytes < 0) {
                            break;
                        }
                        sent += new_bytes;
                    }
                }
            } else if (buffer[0] == MSG_GET_PUPIL_FLTR) {
                for (int i = 0; i < 2; i++) {
                    eye_tracker_->getPupilFiltered(pupil_location_, i);
                    uint32_t sent{0};
                    uint32_t size_to_send{sizeof(pupil_location_)};
                    while (sent < size_to_send) {
                        ssize_t new_bytes{send(socket_handle_,
                                               (char *) &pupil_location_ + sent,
                                               size_to_send - sent, 0)};
                        if (new_bytes < 0) {
                            break;
                        }
                        sent += new_bytes;
                    }
                }
            } else if (buffer[0] == MSG_SAVE_EYE_DATA) {
                uint32_t size_to_read;
                cv::Point3f left_eye_pos{}, right_eye_pos{}, marker_pos{};

                read_bytes = 0;
                size_to_read = sizeof(left_eye_pos);
                while (read_bytes < size_to_read) {
                    ssize_t new_bytes{read(socket_handle_,
                                           (char *) &left_eye_pos + read_bytes,
                                           size_to_read - read_bytes)};
                    if (new_bytes < 0) {
                        break;
                    }
                    read_bytes += (int) new_bytes;
                }

                read_bytes = 0;
                size_to_read = sizeof(right_eye_pos);
                while (read_bytes < size_to_read) {
                    ssize_t new_bytes{read(socket_handle_,
                                           (char *) &right_eye_pos + read_bytes,
                                           size_to_read - read_bytes)};
                    if (new_bytes < 0) {
                        break;
                    }
                    read_bytes += (int) new_bytes;
                }

                read_bytes = 0;
                size_to_read = sizeof(marker_pos);
                while (read_bytes < size_to_read) {
                    ssize_t new_bytes{read(socket_handle_,
                                           (char *) &marker_pos + read_bytes,
                                           size_to_read - read_bytes)};
                    if (new_bytes < 0) {
                        break;
                    }
                    read_bytes += (int) new_bytes;
                }
                eye_tracker_->saveEyeData(left_eye_pos, right_eye_pos,
                                          marker_pos);
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
} // namespace et
