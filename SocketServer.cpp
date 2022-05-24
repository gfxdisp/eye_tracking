#include "SocketServer.hpp"

#include <arpa/inet.h>
#include <chrono>
#include <cstdio>
#include <fcntl.h>
#include <iostream>
#include <thread>
#include <unistd.h>

using cv::Vec3d;

namespace et {
SocketServer::SocketServer(EyeTracker *eye_tracker,
                           FeatureDetector *feature_detector)
    : eye_tracker_(eye_tracker), feature_detector_(feature_detector) {
}

void SocketServer::startServer() {
    int opt{1};

    if ((server_handle_ = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket failed");
        return;
    }

    fcntl(server_handle_, F_SETFL, O_NONBLOCK);

    if (setsockopt(server_handle_, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT,
                   &opt, sizeof(opt))) {
        perror("setsockopt");
        close(server_handle_);
        return;
    }
    address_.sin_family = AF_INET;
    address_.sin_addr.s_addr = inet_addr("127.0.0.1");
    address_.sin_port = htons(8080);

    if (bind(server_handle_, (sockaddr *) &address_, sizeof(address_)) < 0) {
        perror("bind failed");
        close(server_handle_);
        return;
    }
    if (listen(server_handle_, 3) < 0) {
        perror("listen");
        close(server_handle_);
    }
}

void SocketServer::openSocket() {
    std::clog << "Socket opened.\n";
    char buffer[1]{1};
    while (!finished) {
        int addrlen = sizeof(address_);
        while (socket_handle_ < 0 && !finished) {
            socket_handle_ =
                accept(server_handle_, (struct sockaddr *) &address_,
                       (socklen_t *) &addrlen);
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        while (!finished) {
            int valread = read(socket_handle_, buffer, 1);
            if (valread != 1) {
                break;
            }

            if (buffer[0] == 0) {
                finished = true;
            } else if (buffer[0] == 1) {
                eye_tracker_->getCorneaCurvaturePosition(eye_position_);
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

            } else if (buffer[0] == 2) {
                feature_detector_->getGlints(glint_locations_);
                feature_detector_->getPupil(pupil_location_);
                uint32_t sent{0};
                uint32_t size_to_send{0};
                char *variables[]{(char *) glint_locations_.data(),
                                  (char *) &pupil_location_};
                uint32_t sizes[]{
                    static_cast<uint32_t>(sizeof(glint_locations_[0])
                                          * glint_locations_.size()),
                    sizeof(pupil_location_)};
                for (int i = 0; i < sizeof(variables) / sizeof(variables[0]);
                     i++) {
                    sent = 0;
                    size_to_send = sizes[i];
                    while (sent < size_to_send) {
                        ssize_t new_bytes{send(socket_handle_,
                                               variables[i] + sent,
                                               size_to_send - sent, 0)};
                        if (new_bytes < 0) {
                            break;
                        }
                        sent += new_bytes;
                    }
                }
            } else if (buffer[0] == 4) {
                eye_tracker_->getPupilDiameter(pupil_diameter_);
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
            } else if (buffer[0] == 5) {
                eye_tracker_->getGazeDirection(gaze_direction_);
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
            } else if (buffer[0] == 8) {
                feature_detector_->getPupil(pupil_location_);
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
        }
        close(socket_handle_);
    }
    close(socket_handle_);
    std::cout << "Socket closed.\n";
}

void SocketServer::closeSocket() {
    close(server_handle_);
}
}// namespace et
