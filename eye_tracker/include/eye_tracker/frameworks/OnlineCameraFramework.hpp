#ifndef EYE_TRACKER_ONLINECAMERAFRAMEWORK_HPP
#define EYE_TRACKER_ONLINECAMERAFRAMEWORK_HPP

#include "eye_tracker/frameworks/Framework.hpp"

namespace et {
    class OnlineCameraFramework : public Framework {
    public:
        OnlineCameraFramework(int camera_id, bool headless);
    };
} // et

#endif //EYE_TRACKER_ONLINECAMERAFRAMEWORK_HPP
