#pragma once
#include <opencv2/core.hpp>
#include <xtensor/xfixed.hpp>
namespace EyeTracker {
    template<unsigned long... I> using Matrix = xt::xtensor_fixed<float, xt::xshape<I...>>;
    using Vector = Matrix<3>;

    struct EyePosition {
        std::optional<Vector> corneaCurvatureCentre, pupilCentre, eyeCentre; // c, p, d
        inline operator bool() const {
            return eyeCentre and pupilCentre and corneaCurvatureCentre;
        }
    };

    struct PointWithRating {
        cv::Point2f point = {-1, -1};
        float rating = std::numeric_limits<float>::infinity();
        inline bool operator<(const PointWithRating& other) const {
            return rating < other.rating;
        }
    };

    const static cv::Point2i None = {-1, -1};
    using KFMat = cv::Mat_<float>;

    inline cv::Point toPoint(cv::Mat m) {
        return cv::Point(m.at<float>(0, 0), m.at<float>(0, 1));
    }

    inline cv::Mat toMat(cv::Point p) {
        return (cv::Mat_<float>(2, 1) << p.x, p.y);
    }
}
