#ifndef EYE_TRACKER_VIDEOCAMERAFRAMEWORK_HPP
#define EYE_TRACKER_VIDEOCAMERAFRAMEWORK_HPP

#include "eye_tracker/frameworks/Framework.hpp"

namespace et
{

    class VideoCameraFramework : public Framework {
    public:
        VideoCameraFramework(int camera_id, bool headless, const std::string &input_video_path, bool loop);

        cv::Point2d getMarkerPosition() override;

        bool analyzeNextFrame() override;
    protected:
        std::vector<std::vector<double>> csv_data_{};
        cv::Point2d marker_position_{};

        double min_width_{};
        double max_width_{};
        double min_height_{};
        double max_height_{};
    };

} // et

#endif //EYE_TRACKER_VIDEOCAMERAFRAMEWORK_HPP
