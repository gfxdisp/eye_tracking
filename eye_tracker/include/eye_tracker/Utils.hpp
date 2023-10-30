#ifndef HDRMFS_EYE_TRACKER_UTILS_HPP
#define HDRMFS_EYE_TRACKER_UTILS_HPP

#include <opencv2/core/types.hpp>
#include <string>
#include <vector>
#include <random>

namespace et
{

    class Utils
    {
    public:

        /**
         * Reads the CSV file with float values in the column-major order.
         * @param filename Path to the CSV file.
         * @return Vector of columns of the CSV content.
         */
        static std::vector<std::vector<float>>
        readFloatColumnsCsv(const std::string &filename, bool ignore_first_line = false);

        /**
         * Reads the CSV file with float values in the row-major order.
         * @param filename Path to the CSV file.
         * @return Vector of rows of the CSV content.
         */
        static std::vector<std::vector<float>>
        readFloatRowsCsv(const std::string &filename, bool ignore_first_line = false);

        /**
         * Writes the float matrix to CSV file in row-major order.
         * @param data Matrix with the float data.
         * @param filename Path to CSV file to be saved.
         */
        static void writeFloatCsv(std::vector<std::vector<float>> &data, const std::string &filename);

        /**
         * Converts current timestamp to a human-readable format.
         * @return A string with a converted timestamp.
         */
        static std::string getCurrentTimeText();

        static std::vector<std::string> split(std::string to_split, char i);

        static cv::Point3f calculateNodalPointPosition(cv::Point3_<float> observed_point, cv::Point3_<float> eye_centre,
                                                       float nodal_dist = 5.3f);

        static cv::Mat findTransformationMatrix(const cv::Mat &mat_from, const cv::Mat &mat_to);

        static cv::Mat convertToHomogeneous(cv::Mat mat);


        /**
         * Calculates the intersection point between a ray and a sphere.
         * @param ray_pos Position of the ray origin.
         * @param ray_dir Direction of the ray.
         * @param sphere_pos Centre position of the sphere.
         * @param sphere_radius Radius of the sphere.
         * @param t The distance between ray origin and an intersection point.
         * @return True if the intersection is found. False otherwise.
         */
        static bool
        getRaySphereIntersection(const cv::Vec3f &ray_pos, const cv::Vec3d &ray_dir, const cv::Vec3f &sphere_pos,
                                 double sphere_radius, double &t);

        static cv::Point3f visualToOpticalAxis(const cv::Point3f &visual_axis, float alpha, float beta);

        static cv::Point3f opticalToVisualAxis(const cv::Point3f &optical_axis, float alpha, float beta);

        static float pointToLineDistance(cv::Vec3f origin, cv::Vec3f direction, cv::Vec3f point);

        static cv::Point2f findEllipseIntersection(cv::RotatedRect &ellipse, float angle);

        static float getAngleBetweenVectors(cv::Vec3f a, cv::Vec3f b);

        static void getCrossValidationIndices(std::vector<int>& indices, int n_data_points, int n_folds);

        static cv::Point3f findGridIntersection(std::vector<cv::Point3f> &front_corners, std::vector<cv::Point3f> &back_corners);

    private:
        static std::mt19937::result_type seed;
        static std::mt19937 gen;
    };

} // namespace et

#endif //HDRMFS_EYE_TRACKER_UTILS_HPP
