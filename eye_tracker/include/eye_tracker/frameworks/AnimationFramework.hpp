#ifndef EYE_TRACKER_ANIMATIONFRAMEWORK_HPP
#define EYE_TRACKER_ANIMATIONFRAMEWORK_HPP

#include "eye_tracker/frameworks/Framework.hpp"

namespace et
{

    class AnimationFramework : public Framework
    {
    public:
        AnimationFramework(int camera_id, bool headless, const std::string &input_video_path);
    };

} // et

#endif //EYE_TRACKER_ANIMATIONFRAMEWORK_HPP
