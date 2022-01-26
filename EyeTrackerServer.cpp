#include "EyeTrackerServer.h"
#include <cstdio>
#include <arpa/inet.h>
#include <unistd.h>
#include <iostream>

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
            if (buffer[0] != 0) {
                tracker->getEyePosition(eyePosition);
                send(socketHandle, &*eyePosition.eyeCentre, sizeof(*eyePosition.eyeCentre), 0);
            } else {
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

