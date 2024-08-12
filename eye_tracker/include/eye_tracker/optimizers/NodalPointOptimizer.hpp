#ifndef HDRMFS_EYE_TRACKER_RAY_POINT_MINIMIZER_HPP
#define HDRMFS_EYE_TRACKER_RAY_POINT_MINIMIZER_HPP

#include <opencv2/opencv.hpp>

#include <vector>

namespace et {
    class NodalPointOptimizer {
    public:
        explicit NodalPointOptimizer(int camera_id);

        void initialize();

        void setParameters(const cv::Vec3d& np2c_dir, const cv::Vec3d* screen_glint, const std::vector<cv::Vec3d>& lp, const cv::Vec3d& np, double cornea_radius);


        double goldenSectionSearch(double minimum, double maximum, double tolerance = 1e-5) const;

    private:
        static constexpr double golden_ratio = 1.618033988749895;

        cv::Vec3d np_{};

        cv::Vec3d np2c_dir_{};

        std::vector<cv::Vec3d> screen_glint_{};

        std::vector<cv::Vec3d> lp_{};

        std::vector<cv::Vec3d> ray_dir_{};

        int camera_id_{};

        double cornea_radius_{};

        double calculateError(const double& x) const;
    };
} // namespace et

#endif // HDRMFS_EYE_TRACKER_RAY_POINT_MINIMIZER_HPP
