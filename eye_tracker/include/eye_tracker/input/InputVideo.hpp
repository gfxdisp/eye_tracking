#ifndef HDRMFS_EYE_TRACKER_INPUT_VIDEO_HPP
#define HDRMFS_EYE_TRACKER_INPUT_VIDEO_HPP

#include "eye_tracker/input/ImageProvider.hpp"

#include <opencv2/opencv.hpp>

namespace et
{
/**
 * Loads a video from a file to use as a feed.
 */
    class InputVideo : public ImageProvider
    {
    public:
        /**
         * Detects if the videos for one or two eyes are available.
         * @param input_video_path Path to video file.
         */
        explicit InputVideo(const std::string &input_video_path, bool loop = false);

        /**
         * Returns next image from the video feed. If the whole video has been
         * analyzed, it loops back to the beginning.
         * @param camera_id An id of the camera for which the value is returned.
         * @return A pair of images: one for detecting pupil and one for detecting glints.
         */
        EyeImage grabImage() override;

        /**
         * Releases all opened video files.
         */
        void close() override;

    private:
        // Paths to all opened videos. One per eye.
        std::string input_video_path_{};
        // Opened videos. One per eye.
        cv::VideoCapture video_capture_{};

        int frame_num_{0};
        bool loop_{false};
    };
} // namespace et

#endif //HDRMFS_EYE_TRACKER_INPUT_VIDEO_HPP