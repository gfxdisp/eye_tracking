#include "eye_tracker/image/temporal_filter/DiscreteTemporalFilterer.hpp"

namespace et
{
    DiscreteTemporalFilterer::DiscreteTemporalFilterer(int camera_id) : TemporalFilterer(camera_id)
    {

    }

    void DiscreteTemporalFilterer::filterPupil(cv::Point2f &pupil, float &radius)
    {
        // Pupil position is not filtered for the discrete case.
    }

    void DiscreteTemporalFilterer::filterGlints(std::vector<cv::Point2f> &glints)
    {
        cv::Point2f average_glint_position(0, 0);
        for (const auto &glint : glints)
        {
            average_glint_position += glint;
        }
        average_glint_position.x /= glints.size();
        average_glint_position.y /= glints.size();

        std::vector<float> distances(glints.size());
        for (int i = 0; i < glints.size(); ++i)
        {
            distances[i] = cv::norm(glints[i] - average_glint_position);
        }

        std::sort(distances.begin(), distances.end());
        float median_distance = distances[distances.size() / 2];

        for (int i = 0; i < glints.size(); ++i)
        {
            if (cv::norm(glints[i] - average_glint_position) > median_distance * 2)
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