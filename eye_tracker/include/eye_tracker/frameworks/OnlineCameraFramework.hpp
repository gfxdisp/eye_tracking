#ifndef HDRMFS_EYE_TRACKER_ONLINE_CAMERA_FRAMEWORK_HPP
#define HDRMFS_EYE_TRACKER_ONLINE_CAMERA_FRAMEWORK_HPP

#include <eye_tracker/frameworks/Framework.hpp>

namespace et {
    /**
     * @brief The OnlineCameraFramework class is a specialization of the Framework class for eye-tracking using an online camera feed.
     */
    class OnlineCameraFramework : public Framework {
    public:
        OnlineCameraFramework(int camera_id, bool headless);
    };
} // et

#endif //HDRMFS_EYE_TRACKER_ONLINE_CAMERA_FRAMEWORK_HPP
