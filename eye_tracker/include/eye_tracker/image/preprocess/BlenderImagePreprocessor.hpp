#ifndef HDRMFS_EYE_TRACKER_BLENDERIMAGEPREPROCESSOR_HPP
#define HDRMFS_EYE_TRACKER_BLENDERIMAGEPREPROCESSOR_HPP

#include "eye_tracker/image/preprocess/ImagePreprocessor.hpp"

namespace et
{

    class BlenderImagePreprocessor : public ImagePreprocessor
    {
    public:
        BlenderImagePreprocessor(int camera_id);

        void preprocess(const EyeImage &input, EyeImage &output) override;
    };

} // et

#endif //HDRMFS_EYE_TRACKER_BLENDERIMAGEPREPROCESSOR_HPP
