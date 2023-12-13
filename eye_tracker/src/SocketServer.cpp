#include "eye_tracker/SocketServer.hpp"

#include <arpa/inet.h>
#include <chrono>
#include <fcntl.h>
#include <iostream>
#include <thread>
#include <unistd.h>

namespace et
{
    SocketServer::SocketServer(std::shared_ptr<Framework> eye_tracker_left,
                               std::shared_ptr<Framework> eye_tracker_right)
    {
        eye_trackers_[0] = eye_tracker_left;
        eye_trackers_[1] = eye_tracker_right;
    }

    void SocketServer::startServer()
    {
        int opt{1};

        if ((server_handle_ = socket(AF_INET, SOCK_STREAM, 0)) == 0)
        {
            std::clog << "Failed to create a socket.\n";
            return;
        }

        fcntl(server_handle_, F_SETFL, O_NONBLOCK);

        if (setsockopt(server_handle_, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt)))
        {
            std::clog << "Failed to set socket options.\n";
            close(server_handle_);
            return;
        }
        address_.sin_family = AF_INET;
        address_.sin_addr.s_addr = inet_addr("127.0.0.1");
        address_.sin_port = htons(8080);

        if (bind(server_handle_, (sockaddr *) &address_, sizeof(address_)) < 0)
        {
            std::clog << "Failed to bind a socket address.\n";
            close(server_handle_);
            return;
        }
        if (listen(server_handle_, 3) < 0)
        {
            std::clog << "Failed to start socket listening.\n";
            close(server_handle_);
        }

        server_thread_ = std::thread{&et::SocketServer::openSocket, this};
    }

    void SocketServer::openSocket()
    {
        std::clog << "Socket opened.\n";
        char buffer[1]{1};
        while (!finished)
        {
            int address_length = sizeof(address_);
            while (socket_handle_ < 0 && !finished)
            {
                socket_handle_ = accept(server_handle_, (struct sockaddr *) &address_, (socklen_t *) &address_length);
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }

            while (!finished)
            {
                if (!readAll(buffer, 1))
                {
                    break;
                }
                switch (buffer[0])
                {
                    case MSG_CLOSE_CONNECTION:
                    {
                        finished = true;
                        break;
                    }
                    case MSG_GET_EYE_DATA:
                    {
                        char camera_id;
                        if (!readAll(&camera_id, sizeof(camera_id)))
                        {
                            goto connect_failure;
                        }

                        if (camera_id != 0 && camera_id != 1)
                        {
                            goto connect_failure;
                        }

                        eye_trackers_[camera_id]->getEyeDataPackage(eye_data_to_send_);
                        if (!sendAll(&eye_data_to_send_, sizeof(eye_data_to_send_)))
                        {
                            goto connect_failure;
                        }
                        break;
                    }
                    case MSG_START_CALIBRATION:
                    {
                        char camera_id;
                        if (!readAll(&camera_id, sizeof(camera_id)))
                        {
                            goto connect_failure;
                        }

                        if (camera_id != 0 && camera_id != 1)
                        {
                            goto connect_failure;
                        }

                        int path_length{};
                        if (!readAll(&path_length, sizeof(path_length)))
                        {
                            goto connect_failure;
                        }
                        if (!readAll(message_buffer_, path_length))
                        {
                            goto connect_failure;
                        }
                        message_buffer_[path_length] = '\0';
                        std::string video_folder = eye_trackers_[camera_id]->startEyeVideoRecording(message_buffer_);
                        int video_folder_length = video_folder.length();
                        if (!sendAll(&video_folder_length, sizeof(video_folder_length)))
                        {
                            goto connect_failure;
                        }
                        if (!sendAll((void *) video_folder.c_str(), video_folder_length))
                        {
                            goto connect_failure;
                        }
                        break;
                    }
                    case MSG_STOP_CALIBRATION:
                    {
                        char camera_id;
                        if (!readAll(&camera_id, sizeof(camera_id)))
                        {
                            goto connect_failure;
                        }

                        if (camera_id != 0 && camera_id != 1)
                        {
                            goto connect_failure;
                        }
                        eye_trackers_[camera_id]->stopEyeVideoRecording();
                        break;
                    }
                    case MSG_ADD_CALIBRATION_DATA:
                    {
                        char camera_id;
                        if (!readAll(&camera_id, sizeof(camera_id)))
                        {
                            goto connect_failure;
                        }

                        if (camera_id != 0 && camera_id != 1)
                        {
                            goto connect_failure;
                        }

                        if (!readAll(eye_data_to_receive_.front_corners.data(), eye_data_to_receive_.front_corners.size() *
                                                                          sizeof(eye_data_to_receive_.front_corners[0])))
                        {
                            goto connect_failure;
                        }

                        if (!readAll(eye_data_to_receive_.back_corners.data(), eye_data_to_receive_.back_corners.size() *
                                                                                sizeof(eye_data_to_receive_.back_corners[0])))
                        {
                            goto connect_failure;
                        }

                        if (!readAll(&eye_data_to_receive_.marker_position, sizeof(eye_data_to_receive_.marker_position)))
                        {
                            goto connect_failure;
                        }

                        eye_trackers_[camera_id]->addEyeVideoData(eye_data_to_receive_);
                        break;
                    }

                    case MSG_SET_META_MODEL:
                    {
                        char camera_id;
                        if (!readAll(&camera_id, sizeof(camera_id)))
                        {
                            goto connect_failure;
                        }

                        if (camera_id != 0 && camera_id != 1)
                        {
                            goto connect_failure;
                        }

                        int path_length{};
                        if (!readAll(&path_length, sizeof(path_length)))
                        {
                            goto connect_failure;
                        }
                        if (!readAll(message_buffer_, path_length))
                        {
                            goto connect_failure;
                        }
                        std::string meta_model_path = message_buffer_;

                        int user_id_length{};
                        if (!readAll(&user_id_length, sizeof(user_id_length)))
                        {
                            goto connect_failure;
                        }
                        if (!readAll(message_buffer_, user_id_length))
                        {
                            goto connect_failure;
                        }

                        message_buffer_[user_id_length] = '\0';
                        std::string user_id = message_buffer_;
                        cv::Point3d eye_centre = eye_trackers_[camera_id]->setMetaModel(meta_model_path, user_id);
                        if (!sendAll(&eye_centre, sizeof(eye_centre)))
                        {
                            goto connect_failure;
                        }
                        break;
                    }
                }
            }
            connect_failure:
            close(socket_handle_);
        }

        close(socket_handle_);
        std::cout << "Socket closed.\n";
    }

    void SocketServer::closeSocket() const
    {
        close(server_handle_);
    }

    bool SocketServer::isClientConnected() const
    {
        return socket_handle_ > -1 && !finished;
    }

    bool SocketServer::sendAll(void *input, size_t bytes_count) const
    {
        size_t size_to_send = bytes_count;
        size_t sent_bytes = 0;
        while (sent_bytes < size_to_send)
        {
            ssize_t new_bytes = send(socket_handle_, (char *) input + sent_bytes, size_to_send - sent_bytes, 0);
            if (new_bytes < 0)
            {
                return false;
            }
            sent_bytes += new_bytes;
        }
        return true;
    }

    bool SocketServer::readAll(void *output, size_t bytes_count) const
    {
        size_t size_to_read = bytes_count;
        size_t read_bytes = 0;
        while (read_bytes < size_to_read)
        {
            ssize_t new_bytes = read(socket_handle_, (char *) output + read_bytes, size_to_read - read_bytes);
            if (new_bytes < 0)
            {
                return false;
            }
            read_bytes += new_bytes;
        }
        return true;
    }

    SocketServer::~SocketServer()
    {
        if (server_thread_.joinable())
        {
            server_thread_.join();
        }
    }

} // namespace et
