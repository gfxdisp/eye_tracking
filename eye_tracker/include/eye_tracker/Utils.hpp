#ifndef HDRMFS_EYE_TRACKER_UTILS_HPP
#define HDRMFS_EYE_TRACKER_UTILS_HPP

#include <opencv2/core/types.hpp>
#include <string>
#include <vector>
#include <random>

namespace et {
    class Utils {
    public:
        static std::vector<std::vector<double> > readCsv(const std::string& filename, bool ignore_first_line = false);

        static void writeCsv(std::vector<std::vector<double> >& data, const std::string& filename, bool append = false, const std::string& header = "");

        static std::string getCurrentTimeText();

        static bool getRaySphereIntersection(const cv::Vec3d& ray_pos, const cv::Vec3d& ray_dir, const cv::Vec3d& sphere_pos, double sphere_radius, double& t);

        static cv::Point3d opticalToVisualAxis(const cv::Point3d& optical_axis, double alpha, double beta);

        static double getAngleBetweenVectors(cv::Vec3d a, cv::Vec3d b);

        static cv::Vec3d getRefractedRay(const cv::Vec3d& direction, const cv::Vec3d& normal, double refraction_index);

        template<typename T>
        static T getMean(const std::vector<T>& values) {
            T sum{};
            for (auto& value: values) {
                sum += value;
            }
            int n = values.size();
            return sum / n;
        }

        static cv::Point3d getTrimmmedMean(const std::vector<cv::Point3d>& values, double trim_ratio);

        static cv::Vec3d getMedian(const std::vector<cv::Vec3d>& values);

        template<typename T>
        static T getStdDev(const std::vector<T>& values) {
            T mean = Utils::getMean<T>(values);
            T std{};
            for (int i = 0; i < values.size(); i++) {
                std += (values[i] - mean) * (values[i] - mean);
            }
            std /= values.size();
            std = std::sqrt(std);
            return std;
        }

        static void vectorToAngles(cv::Vec3d vector, cv::Vec2d& angles);

        static void anglesToVector(cv::Vec2d angles, cv::Vec3d& vector);

    private:
        static std::mt19937::result_type seed;
        static std::mt19937 gen;
    };
} // namespace et

#endif //HDRMFS_EYE_TRACKER_UTILS_HPP
