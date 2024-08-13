#include <eye_tracker/frameworks/VideoCameraFramework.hpp>
#include <eye_tracker/eye/EyeEstimator.hpp>
#include <eye_tracker/input/InputVideo.hpp>
#include <eye_tracker/image/FeatureAnalyser.hpp>
#include <eye_tracker/Utils.hpp>

namespace et {
    VideoCameraFramework::VideoCameraFramework(int camera_id, bool headless, bool loop, const std::string& input_video_path, const std::string& csv_file_path) : Framework(camera_id, headless) {
        image_provider_ = std::make_shared<InputVideo>(input_video_path, loop);
        feature_detector_ = std::make_shared<FeatureAnalyser>(camera_id);
        eye_estimator_ = std::make_shared<EyeEstimator>(camera_id);

        csv_data_ = Utils::readCsv(csv_file_path, true);
        markers_from_csv_ = !csv_data_.empty();
        marker_position_ = cv::Point2d(-1, -1);
    }

    cv::Point2d VideoCameraFramework::getMarkerPosition() {
        return marker_position_;
    }

    bool VideoCameraFramework::analyzeNextFrame() {
        static int frame_counter = 0;
        const bool return_value = Framework::analyzeNextFrame();
        if (return_value && markers_from_csv_) {
            marker_position_ = cv::Point2d(csv_data_[frame_counter][3], csv_data_[frame_counter][4]);
            frame_counter = (frame_counter + 1) % static_cast<int>(csv_data_.size());

            marker_position_.x = (marker_position_.x - EyeEstimator::bottom_left_gaze_window_x_mm_) / (EyeEstimator::upper_right_gaze_window_x_mm_ - EyeEstimator::bottom_left_gaze_window_x_mm_);
            marker_position_.y = (marker_position_.y - EyeEstimator::bottom_left_gaze_window_y_mm_) / (EyeEstimator::upper_right_gaze_window_y_mm_ - EyeEstimator::bottom_left_gaze_window_y_mm_);
        }
        return return_value;
    }
} // et
