#ifndef EYE_TRACKER_RANDOMIMAGESFRAMEWORK_HPP
#define EYE_TRACKER_RANDOMIMAGESFRAMEWORK_HPP

#include "eye_tracker/frameworks/Framework.hpp"

namespace et
{

    class RandomImagesFramework : public Framework
    {
    public:
        RandomImagesFramework(int camera_id, bool headless, const std::string &images_folder_path);
    };

} // et


#endif //EYE_TRACKER_RANDOMIMAGESFRAMEWORK_HPP
