#ifndef EYE_TRACKER_EYETRACKERSERVER_H
#define EYE_TRACKER_EYETRACKERSERVER_H

#include <netinet/in.h>
#include "eye_tracking.hpp"

class EyeTrackerServer {
public:
    EyeTrackerServer(EyeTracking::Tracker *tracker);

    void startServer();
    void openSocket();
    void closeSocket();

    bool finished = false;
private:
    EyeTracking::Tracker* tracker;
    EyeTracking::EyePosition eyePosition;
    EyeTracking::ImagePositions imagePositions;
    sockaddr_in address;
    int serverHandle;
    int socketHandle;
};


#endif //EYE_TRACKER_EYETRACKERSERVER_H
