#ifndef HDRMFS_EYE_TRACKER_VIDEO_CAMERA_FRAMEWORK_HPP
#define HDRMFS_EYE_TRACKER_VIDEO_CAMERA_FRAMEWORK_HPP

#include <eye_tracker/frameworks/Framework.hpp>

namespace et {
    class VideoCameraFramework : public Framework {
    public:
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
