#include "eye_tracker/frameworks/VideoCameraFramework.hpp"
#include "eye_tracker/input/InputVideo.hpp"
#include "eye_tracker/image/CameraFeatureAnalyser.hpp"
#include "eye_tracker/eye/PolynomialEyeEstimator.hpp"
#include "eye_tracker/Utils.hpp"

namespace et
{
    VideoCameraFramework::VideoCameraFramework(int camera_id, bool headless, const std::string &input_video_path, bool loop) : Framework(camera_id, headless)
    {
        image_provider_ = std::make_shared<InputVideo>(input_video_path, loop);
        feature_detector_ = std::make_shared<CameraFeatureAnalyser>(camera_id);
        eye_estimator_ = std::make_shared<ModelEyeEstimator>(camera_id);

        std::string csv_file_path = input_video_path.substr(0, input_video_path.find_last_of('.')) + ".csv";
        csv_data_ = Utils::readFloatRowsCsv(csv_file_path, true);

        min_width_ = 130;
        max_width_ = 330;
        min_height_ = 50;
        max_height_ = 250;

//        min_width_ -= 100;
//        max_width_ -= 100;
//        min_height_ -= 100;
//        max_height_ -= 100;
    }

    cv::Point2d VideoCameraFramework::getMarkerPosition() {
        return marker_position_;
    }

    bool VideoCameraFramework::analyzeNextFrame() {
        static int frame_counter = 0;
        bool return_value = Framework::analyzeNextFrame();
        if (return_value) {
            marker_position_ = cv::Point2d(csv_data_[frame_counter][3], csv_data_[frame_counter][4]);
//            marker_position_ = {160.2219,142.57618};

            marker_position_.x = (marker_position_.x - min_width_) / (max_width_ - min_width_);
            marker_position_.y = (marker_position_.y - min_height_) / (max_height_ - min_height_);
            frame_counter = (frame_counter + 1) % csv_data_.size();
        }
        return return_value;
    }
} // et