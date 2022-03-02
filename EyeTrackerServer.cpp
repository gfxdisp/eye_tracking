#include "EyeTrackerServer.h"
#include <cstdio>
#include <arpa/inet.h>
#include <unistd.h>
#include <iostream>

using cv::Vec3d;

EyeTrackerServer::EyeTrackerServer(EyeTracking::Tracker *tracker) : tracker(tracker) {}

void EyeTrackerServer::startServer() {
    int opt = 1;

    if ((serverHandle = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket failed");
        return;
    }

    if (setsockopt(serverHandle, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT,
                   &opt, sizeof(opt))) {
        perror("setsockopt");
        close(serverHandle);
        return;
    }
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = inet_addr("127.0.0.1");
    address.sin_port = htons(8080);

    if (bind(serverHandle, (sockaddr *) &address,
             sizeof(address)) < 0) {
        perror("bind failed");
        close(serverHandle);
        return;
    }
    if (listen(serverHandle, 3) < 0) {
        perror("listen");
        close(serverHandle);
    }
}

void EyeTrackerServer::openSocket() {
    std::cout << "Socket opened.\n";
    char buffer[1] = { 1 };
    while (!finished) {
        int addrlen = sizeof(address);
        if ((socketHandle = accept(serverHandle, (struct sockaddr *) &address,
                                   (socklen_t *) &addrlen)) < 0) {
            perror("accept");
            return;
        }
        while (!finished) {
            int valread = read(socketHandle, buffer, 1);
            if (valread != 1) {
                break;
            }
            if (buffer[0] == 1) {
                tracker->getEyePosition(eyePosition);
                uint32_t sent = 0;
                uint32_t size_to_send = sizeof(*eyePosition.eyeCentre);
                while (sent < size_to_send) {
                    sent += send(socketHandle, (char*)&*eyePosition.eyeCentre + sent, size_to_send - sent, 0);
                }

            }
            else if (buffer[0] == 2) {
                tracker->getImagePositions(imagePositions);
                uint32_t sent = 0;
                uint32_t size_to_send = sizeof(imagePositions);
                while (sent < size_to_send) {
                    sent += send(socketHandle, (char*)&imagePositions + sent, size_to_send - sent, 0);
                }
            }
            else if (buffer[0] == 3) {
                float lambda;
                float cameraEyeDistance;
                Vec3d nodalPoint;
                Vec3d light1;
                Vec3d light2;
                uint32_t bytes_read = 0;
                uint32_t size_to_read = 0;
                char* variables[] = {(char*)&lambda, (char*)&cameraEyeDistance, (char*)&nodalPoint, (char*)&light1, (char*)&light2};
                uint32_t sizes[] = {sizeof(lambda), sizeof(cameraEyeDistance), sizeof(nodalPoint), sizeof(light1), sizeof(light2)};
                for (int i = 0; i < 5; i++) {
                    bytes_read = 0;
                    size_to_read = sizes[i];
                    while (bytes_read < size_to_read) {
                        bytes_read += read(socketHandle, variables[i] + bytes_read, size_to_read - bytes_read);
                    }
                }
                std::cout << "Received: " << lambda << " " << cameraEyeDistance << " " << nodalPoint << " " << light1 << " " << light2 << std::endl;
                tracker->setNewParameters(lambda, cameraEyeDistance, nodalPoint, light1, light2);
            }
            else if (buffer[0] == 4) {
                tracker->getPupilDiameter(pupilDiameter);
                uint32_t sent = 0;
                uint32_t size_to_send = sizeof(pupilDiameter);
                while (sent < size_to_send) {
                    sent += send(socketHandle, (char*)&pupilDiameter + sent, size_to_send - sent, 0);
                }
            }
            else if (buffer[0] == 0) {
                std::cout << "Socket closed.\n";
                finished = true;
            }
        }
        close(socketHandle);
    }
    close(socketHandle);
}

void EyeTrackerServer::closeSocket() {
    close(serverHandle);
}

