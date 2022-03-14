#include "SocketServer.hpp"

#include <arpa/inet.h>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <fcntl.h>
#include <thread>
#include <unistd.h>



using cv::Vec3d;

namespace et
{
    SocketServer::SocketServer(EyeTracker *eye_tracker, FeatureDetector *feature_detector) : eye_tracker_(eye_tracker), feature_detector_(feature_detector) {}

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

        if (bind(server_handle_, (sockaddr *) &address_,
                 sizeof(address_)) < 0) {
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
            while (socket_handle_ < 0 && !finished)
            {
                socket_handle_ = accept(server_handle_, (struct sockaddr *) &address_, (socklen_t *) &addrlen);
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }

            while (!finished) {
                int valread = read(socket_handle_, buffer, 1);
                if (valread != 1) 
                {
                    break;
                }
                
                if (buffer[0] == 0) 
                {
                    std::cout << "Socket closed.\n";
                    finished = true;
                }
                else if (buffer[0] == 1) 
                {
                    eye_tracker_->getCorneaCurvaturePosition(eye_position_);
                    uint32_t sent{0};
                    uint32_t size_to_send{sizeof(eye_position_)};
                    while (sent < size_to_send) 
                    {
                        ssize_t new_bytes{send(socket_handle_, (char*)&eye_position_ + sent, size_to_send - sent, 0)};
                        if (new_bytes < 0) 
                        {
                            break;
                        }
                        sent += new_bytes;
                    }

                }
                else if (buffer[0] == 2) 
                {
                    feature_detector_->getLeds(leds_locations_);
                    feature_detector_->getPupil(pupil_location_);
                    uint32_t sent{0};
                    uint32_t size_to_send{0};
                    char* variables[]{(char*)&leds_locations_, (char*)&pupil_location_};
                    uint32_t sizes[]{sizeof(leds_locations_), sizeof(pupil_location_)};
                    for (int i = 0; i < sizeof(variables) / sizeof(variables[0]); i++) 
                    {
                        sent = 0;
                        size_to_send = sizes[i];
                        while (sent < size_to_send) 
                        {
                            ssize_t new_bytes{send(socket_handle_, variables[i] + sent, size_to_send - sent, 0)};
                            if (new_bytes < 0) 
                            {
                                break;
                            }
                            sent += new_bytes;
                        }
                    }
                }
                else if (buffer[0] == 3) 
                {
                    SetupLayout setup_layout{};

                    uint32_t bytes_read{0};
                    uint32_t size_to_read{0};
                    char* variables[]{(char*)&setup_layout.camera_lambda, (char*)&setup_layout.camera_eye_distance, (char*)&setup_layout.camera_nodal_point_position, (char*)setup_layout.led_positions};
                    uint32_t sizes[]{sizeof(setup_layout.camera_lambda), sizeof(setup_layout.camera_eye_distance), sizeof(setup_layout.camera_nodal_point_position), sizeof(setup_layout.led_positions)};
                    for (int i = 0; i < sizeof(variables) / sizeof(variables[0]); i++) 
                    {
                        bytes_read = 0;
                        size_to_read = sizes[i];
                        while (bytes_read < size_to_read) 
                        {
                            ssize_t new_bytes{read(socket_handle_, variables[i] + bytes_read, size_to_read - bytes_read)};
                            if (new_bytes < 0) 
                            {
                                break;
                            }
                            bytes_read += new_bytes;
                        }
                    }
                    std::clog << "Received: " << setup_layout.camera_lambda << " " << setup_layout.camera_eye_distance << " " << setup_layout.camera_nodal_point_position << " " << setup_layout.led_positions[0] << " " << setup_layout.led_positions[1] << std::endl;
                    setup_layout.camera_eye_projection_factor = setup_layout.camera_eye_distance / setup_layout.camera_lambda;


                    eye_tracker_->setNewSetupLayout(setup_layout);
                }
                else if (buffer[0] == 4) 
                {
                    feature_detector_->getPupilRadius(pupil_radius_);
                    pupil_radius_ *= 20.0f / 175.0f;
                    uint32_t sent{0};
                    uint32_t size_to_send{sizeof(pupil_radius_)};
                    while (sent < size_to_send) 
                    {
                        ssize_t new_bytes{send(socket_handle_, (char*)&pupil_radius_ + sent, size_to_send - sent, 0)};
                        if (new_bytes < 0) 
                        {
                            break;
                        }
                        sent += new_bytes;
                    }
                }
                else if (buffer[0] == 5) 
                {
                    eye_tracker_->getGazeDirection(gaze_direction_);
                    uint32_t sent{0};
                    uint32_t size_to_send{sizeof(gaze_direction_)};
                    while (sent < size_to_send) 
                    {
                        ssize_t new_bytes{send(socket_handle_, (char*)&gaze_direction_ + sent, size_to_send - sent, 0)};
                        if (new_bytes < 0) 
                        {
                            break;
                        }
                        sent += new_bytes;
                    }
                }
                else if (buffer[0] == 6)
                {
                    int threshold;
                    uint32_t bytes_read{0};
                    uint32_t size_to_read{sizeof(threshold)};
                    while (bytes_read < size_to_read)
                    {
                        ssize_t new_bytes{read(socket_handle_, (char*)&threshold + bytes_read, size_to_read - bytes_read)};
                        if (new_bytes < 0) 
                        {
                            break;
                        }
                        bytes_read += new_bytes;
                    }
                    feature_detector_->pupil_threshold = threshold;
                }
                else if (buffer[0] == 7)
                {
                    int threshold = feature_detector_->pupil_threshold;
                    uint32_t sent{0};
                    uint32_t size_to_send{sizeof(threshold)};
                    while (sent < size_to_send) 
                    {
                        ssize_t new_bytes{send(socket_handle_, (char*)&threshold + sent, size_to_send - sent, 0)};
                        if (new_bytes < 0) 
                        {
                            break;
                        }
                        sent += new_bytes;
                    }
                }
            }
            close(socket_handle_);
        }
        close(socket_handle_);
    }

    void SocketServer::closeSocket() {
        close(server_handle_);
    }
}
