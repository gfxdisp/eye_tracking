#include "eye_tracker/image/temporal_filter/DiscreteTemporalFilterer.hpp"

namespace et
{
    DiscreteTemporalFilterer::DiscreteTemporalFilterer(int camera_id) : TemporalFilterer(camera_id)
    {

    }

    void DiscreteTemporalFilterer::filterPupil(cv::Point2d &pupil, double &radius)
    {
        // Pupil position is not filtered for the discrete case.
    }

    void DiscreteTemporalFilterer::filterGlints(std::vector<cv::Point2f> &glints)
    {
        cv::Point2d average_glint_position(0, 0);
        for (const auto &glint : glints)
        {
            average_glint_position += (cv::Point2d) glint;
        }
        average_glint_position.x /= glints.size();
        average_glint_position.y /= glints.size();

        std::vector<double> distances(glints.size());
        for (int i = 0; i < glints.size(); ++i)
        {
            distances[i] = cv::norm((cv::Point2d) glints[i] - average_glint_position);
        }

        std::sort(distances.begin(), distances.end());
        double median_distance = distances[distances.size() / 2];

        for (int i = 0; i < glints.size(); ++i)
        {
            if (cv::norm((cv::Point2d) glints[i] - average_glint_position) > median_distance * 2)
            {
                glints.erase(glints.begin() + i);
                --i;
            }
        }
    }

    void DiscreteTemporalFilterer::filterEllipse(cv::RotatedRect &ellipse)
    {
        // Ellipse is not filtered for the discrete case.
    }
} // et