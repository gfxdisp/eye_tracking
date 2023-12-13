#ifndef EYE_TRACKER_VIDEOCAMERAFRAMEWORK_HPP
#define EYE_TRACKER_VIDEOCAMERAFRAMEWORK_HPP

#include "eye_tracker/frameworks/Framework.hpp"

namespace et
{

    class VideoCameraFramework : public Framework {
    public:
        VideoCameraFramework(int camera_id, bool headless, const std::string &input_video_path, bool loop);
    };

} // et

#endif //EYE_TRACKER_VIDEOCAMERAFRAMEWORK_HPP
