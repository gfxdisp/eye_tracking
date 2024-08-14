#ifndef HDRMFS_EYE_TRACKER_VIDEO_CAMERA_FRAMEWORK_HPP
#define HDRMFS_EYE_TRACKER_VIDEO_CAMERA_FRAMEWORK_HPP

#include <eye_tracker/frameworks/Framework.hpp>

namespace et {
    /**
     * @brief The VideoCameraFramework class is a specialization of the Framework class for eye-tracking using an already recorded video.
     */
    class VideoCameraFramework : public Framework {
    public:
        /**
         * @brief Construct a new Video Camera Framework object
         * @param camera_id Number of the camera to use: 0 for the left camera, 1 for the right camera.
         * @param headless If true, the tracking will be done without any visualization.
         * @param loop If true, the video will be looped back to the beginning once it has finished. If false, the program will exit once the video has finished.
         * @param input_video_path Path to the video file to be used for showing the video.
         * @param csv_file_path Path to the CSV file containing the calibration data, specifically the real eye positions, marker positions, and timestamps. Used for displaying the marker positions in the "Gaze position" part of the UI. If no CSV file is provided, the marker positions will not be displayed.
         */
        VideoCameraFramework(int camera_id, bool headless, bool loop, const std::string& input_video_path, const std::string& csv_file_path = "");

        cv::Point2d getMarkerPosition() override;

        bool analyzeNextFrame() override;

    protected:
        std::vector<std::vector<double> > csv_data_{};
        cv::Point2d marker_position_{};
        bool markers_from_csv_{};
    };
} // et

#endif //HDRMFS_EYE_TRACKER_VIDEO_CAMERA_FRAMEWORK_HPP
