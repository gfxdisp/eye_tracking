#ifndef HDRMFS_EYE_TRACKER_UTILS_HPP
#define HDRMFS_EYE_TRACKER_UTILS_HPP

#include <opencv2/core/types.hpp>
#include <string>
#include <vector>
#include <random>

namespace et
{

    class Utils {
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

        static cv::Point3d visualToOpticalAxis2(const cv::Point3d &visual_axis, double alpha, double beta);

        static cv::Point3d opticalToVisualAxis(const cv::Point3d &optical_axis, double alpha, double beta);

        static cv::Point3d opticalToVisualAxis2(const cv::Point3d &optical_axis, double alpha, double beta);

        static int solveQuartic(double r0, double r1, double r2, double r3, double r4, double *roots);

        static cv::Vec2d project3Dto2D(cv::Vec3d point, cv::Vec3d normal);

        static cv::Vec3d unproject2Dto3D(cv::Vec2d point, cv::Vec3d normal, double depth);

        static cv::Point2d getReflectionPoint(cv::Vec2d source, cv::Vec2d destination);

        static cv::Point3d getReflectionPoint(cv::Vec3d source, cv::Vec3d sphere_centre, cv::Vec3d destination, double radius);


        static double pointToLineDistance(cv::Vec3d origin, cv::Vec3d direction, cv::Vec3d point);

        static cv::Point2d findEllipseIntersection(cv::RotatedRect &ellipse, double angle);

        static double getAngleBetweenVectors(cv::Vec3d a, cv::Vec3d b);

        static void getVectorFromAngles(double alpha, double beta, cv::Vec3d &vector);

        static void getCrossValidationIndices(std::vector<int> &indices, int n_data_points, int n_folds);

        static cv::Point3d
        findGridIntersection(std::vector<cv::Point3d> &front_corners, std::vector<cv::Point3d> &back_corners);

        static cv::Mat getRotX(double angle);

        static cv::Mat getRotXd(double angle);

        static cv::Mat getRotY(double angle);

        static cv::Mat getRotYd(double angle);

        static cv::Mat getRotZ(double angle);

        static cv::Mat getRotZd(double angle);

        static cv::Mat convertAxisAngleToRotationMatrix(cv::Point3d axis, double angle);

        static cv::Mat convertAxisAngleToRotationMatrix(cv::Mat axis, double angle);

        static void getAnglesBetweenVectors(cv::Vec3d a, cv::Vec3d b, double &alpha, double &beta);

        static void getAnglesBetweenVectorsAlt(cv::Vec3d visual_axis, cv::Vec3d optical_axis, double &alpha, double &beta);

        static cv::Vec3d getRefractedRay(const cv::Vec3d &direction, const cv::Vec3d &normal, double refraction_index);

        template<typename T>
        static T getMean(const std::vector<T> &values)
        {
            T sum{};
            for (auto &value: values) {
                sum += value;
            }
            int n = values.size();
            return sum / n;
        }

        template<typename T>
        static T getStdDev(const std::vector<T> &values)
        {
            T mean = Utils::getMean<T>(values);
            T std{};
            for (int i = 0; i < values.size(); i++)
            {
                std += (values[i] - mean) * (values[i] - mean);
            }
            std /= values.size();
            std = std::sqrt(std);
            return std;
        }

        static cv::Point3d getStdDev(const std::vector<cv::Point3d>& values);

        static cv::Point2d getStdDev(const std::vector<cv::Point2d>& values);

        template<typename T>
        static std::vector<double> getDists(const std::vector<T> &points, const T &center)
        {
            std::vector<double> dists;
            for (auto &point: points) {
                dists.push_back(cv::norm(point - center));
            }
            return dists;
        }

        static double getPercentile(const std::vector<double> &values, double percentile);

        static cv::Mat getTransformationBetweenMatrices(std::vector<cv::Point3d> from, std::vector<cv::Point3d> to);

        static cv::Point3d convertFromHomogeneous(cv::Mat mat);

        static void vectorToAngles(cv::Vec3d vector, cv::Vec2d &angles);

        static void anglesToVector(cv::Vec2d angles, cv::Vec3d &vector);

    private:
        static std::mt19937::result_type seed;
        static std::mt19937 gen;
    };

} // namespace et

#endif //HDRMFS_EYE_TRACKER_UTILS_HPP
