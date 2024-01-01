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
        static std::vector<std::vector<double>>
        readFloatColumnsCsv(const std::string &filename, bool ignore_first_line = false);

        /**
         * Reads the CSV file with float values in the row-major order.
         * @param filename Path to the CSV file.
         * @return Vector of rows of the CSV content.
         */
        static std::vector<std::vector<double>>
        readFloatRowsCsv(const std::string &filename, bool ignore_first_line = false);

        /**
         * Writes the float matrix to CSV file in row-major order.
         * @param data Matrix with the float data.
         * @param filename Path to CSV file to be saved.
         */
        static void writeFloatCsv(std::vector<std::vector<double>> &data, const std::string &filename);

        /**
         * Converts current timestamp to a human-readable format.
         * @return A string with a converted timestamp.
         */
        static std::string getCurrentTimeText();

        static std::vector<std::string> split(std::string to_split, char i);

        static cv::Point3d
        calculateNodalPointPosition(cv::Point3d observed_point, cv::Point3d eye_centre, double nodal_dist = 5.3);

        static cv::Mat convertToHomogeneous(cv::Mat mat);

        static cv::Mat convertToHomogeneous(cv::Point3d point, double w = 1.0);


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
        getRaySphereIntersection(const cv::Vec3d &ray_pos, const cv::Vec3d &ray_dir, const cv::Vec3d &sphere_pos,
                                 double sphere_radius, double &t);

        static cv::Point3d visualToOpticalAxis(const cv::Point3d &visual_axis, double alpha, double beta);

        static cv::Point3d opticalToVisualAxis(const cv::Point3d &optical_axis, double alpha, double beta);

        static double pointToLineDistance(cv::Vec3d origin, cv::Vec3d direction, cv::Vec3d point);

        static cv::Point2d findEllipseIntersection(cv::RotatedRect &ellipse, double angle);

        static double getAngleBetweenVectors(cv::Vec3d a, cv::Vec3d b);

        static void getCrossValidationIndices(std::vector<int> &indices, int n_data_points, int n_folds);

        static cv::Point3d
        findGridIntersection(std::vector<cv::Point3d> &front_corners, std::vector<cv::Point3d> &back_corners);

        static cv::Mat getRotX(double angle);

        static cv::Mat getRotY(double angle);

        static cv::Mat getRotZ(double angle);

        static cv::Mat convertAxisAngleToRotationMatrix(cv::Mat axis, double angle);

        static void getAnglesBetweenVectors(cv::Vec3d a, cv::Vec3d b, double &alpha, double &beta);

        static void getAnglesBetweenVectorsAlt(cv::Vec3d visual_axis, cv::Vec3d optical_axis, double &alpha, double &beta);

        static cv::Vec3d getRefractedRay(const cv::Vec3d &direction, const cv::Vec3d &normal, double refraction_index);

        static double getStdDev(const std::vector<double>& values);

        static double getPercentile(const std::vector<double>& values, double percentile);

        static cv::Mat getTransformationBetweenMatrices(std::vector<cv::Point3d> from, std::vector<cv::Point3d> to);

        static cv::Point3d convertFromHomogeneous(cv::Mat mat);

    private:
        static std::mt19937::result_type seed;
        static std::mt19937 gen;
    };

} // namespace et

#endif //HDRMFS_EYE_TRACKER_UTILS_HPP
