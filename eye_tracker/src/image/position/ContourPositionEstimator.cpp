#include "eye_tracker/image/position/ContourPositionEstimator.hpp"
#include "eye_tracker/Settings.hpp"

#include <opencv2/imgproc.hpp>

namespace et
{
    ContourPositionEstimator::ContourPositionEstimator(int camera_id) : FeatureEstimator(camera_id)
    {
        pupil_search_centre_ = &Settings::parameters.detection_params[camera_id].pupil_search_centre;
        pupil_search_radius_ = &Settings::parameters.detection_params[camera_id].pupil_search_radius;
        min_pupil_radius_ = &Settings::parameters.detection_params[camera_id].min_pupil_radius;
        max_pupil_radius_ = &Settings::parameters.detection_params[camera_id].max_pupil_radius;
    }

    bool ContourPositionEstimator::findPupil(cv::Mat &image, cv::Point2f &pupil_position, float &radius)
    {
        cv::findContours(image, contours_, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::Point2f best_pupil_position{};
        float best_radius{};
        float best_rating{0};

        cv::Point2f image_centre{*pupil_search_centre_};
        auto max_distance = (float) (*pupil_search_radius_);

        // All the contours are analyzed
        for (const std::vector<cv::Point> &contour: contours_)
        {
            cv::Point2f est_pupil_position;
            float est_radius;

            cv::Rect bound_rect = cv::boundingRect(contour);
            est_pupil_position = 0.5 * (bound_rect.tl() + bound_rect.br());
            est_radius = (float) std::max(bound_rect.width, bound_rect.height) / 2;

            // Contours forming too small or too large pupils are rejected.
            if (est_radius < *min_pupil_radius_ or est_radius > *max_pupil_radius_)
            {
                continue;
            }

            // Contours outside the hole in the view piece are rejected.
            float distance = euclideanDistance(est_pupil_position, image_centre);
            if (distance > max_distance)
            {
                continue;
            }

            // Contours are rated according to their similarity to the circle.
            const float contour_area = static_cast<float>(cv::contourArea(contour));
            const float circle_area = 3.1415926f * powf(est_radius, 2);
            float rating = contour_area / circle_area;
            if (rating >= best_rating)
            {
                best_pupil_position = est_pupil_position;
                best_rating = rating;
                best_radius = est_radius;
            }
        }

        if (best_rating == 0)
        {
            return false;
        }

        pupil_position = best_pupil_position;
        radius = best_radius;
        return true;
    }

    bool ContourPositionEstimator::findGlints(cv::Mat &image, std::vector<cv::Point2f> &glints)
    {
        cv::findContours(image, contours_, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        glints.clear();

        // Calculates the central point of each contour.
        for (const auto &contour: contours_)
        {
            cv::Point2d mean_point{};
            for (const auto &point: contour)
            {
                mean_point.x += point.x;
                mean_point.y += point.y;
            }
            mean_point.x /= (double) contour.size();
            mean_point.y /= (double) contour.size();
            glints.push_back(mean_point);
        }

        return glints.size() >= 5;
    }
} // et