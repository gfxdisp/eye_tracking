#include "eye_tracker/Utils.hpp"
#include "eye_tracker/optimizers/OpticalFromVisualAxisOptimizer.hpp"

#include <chrono>
#include <ctime>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <sstream>
#include <numeric>
#include <iostream>

namespace et
{
    std::mt19937::result_type Utils::seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 Utils::gen = std::mt19937(seed);

    std::vector<std::vector<double>> Utils::readFloatColumnsCsv(const std::string &filename, bool ignore_first_line)
    {
        std::ifstream input_file{filename};
        if (!input_file.is_open())
        {
            return {};
        }

        std::string line{};
        std::vector<std::vector<double>> csv_data{};
        int line_num{0};

        if (ignore_first_line)
        {
            std::getline(input_file, line);
        }

        while (std::getline(input_file, line))
        {
            std::string str_value{};
            std::stringstream stream_line{line};
            int val_num{0};
            while (std::getline(stream_line, str_value, ','))
            {
                if (line_num == 0)
                {
                    csv_data.emplace_back();
                }
                csv_data[val_num].push_back(std::stof(str_value));
                val_num++;
            }
            line_num++;
        }
        input_file.close();
        return csv_data;
    }

    std::vector<std::vector<double>> Utils::readFloatRowsCsv(const std::string &filename, bool ignore_first_line)
    {
        std::ifstream input_file{filename};
        if (!input_file.is_open())
        {
            return {};
        }

        std::string line{};
        std::vector<std::vector<double>> csv_data{};

        if (ignore_first_line)
        {
            std::getline(input_file, line);
        }

        while (std::getline(input_file, line))
        {
            std::vector<double> row{};
            std::string str_value{};
            std::stringstream stream_line{line};
            while (std::getline(stream_line, str_value, ','))
            {
                row.push_back(std::stof(str_value));
            }
            csv_data.push_back(row);
        }
        input_file.close();
        return csv_data;
    }

    void Utils::writeFloatCsv(std::vector<std::vector<double>> &data, const std::string &filename)
    {
        std::ofstream file{filename};
        for (auto &row: data)
        {
            for (int i = 0; i < row.size(); i++)
            {
                if (i != 0)
                {
                    file << "," << row[i];
                }
                else
                {
                    file << row[i];
                }
            }
            file << "\n";
        }
        file.close();
    }

    std::string Utils::getCurrentTimeText()
    {
        std::time_t now{std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())};
        static char buffer[80];

        std::strftime(buffer, 80, "%Y-%m-%d_%H-%M-%S", std::localtime(&now));
        std::string s{buffer};
        return s;
    }

    std::vector<std::string> Utils::split(std::string to_split, char i)
    {
        std::vector<std::string> result{};
        std::stringstream ss{to_split};
        std::string item{};
        while (std::getline(ss, item, i))
        {
            result.push_back(item);
        }
        return result;
    }

    cv::Point3d Utils::calculateNodalPointPosition(cv::Point3d observed_point, cv::Point3d eye_centre,
                                                   double nodal_dist)
    {
        cv::Point3d nodal_point{};
        double dist = cv::norm(observed_point - eye_centre);
        double ratio = nodal_dist / dist;
        nodal_point.x = eye_centre.x + ratio * (observed_point.x - eye_centre.x);
        nodal_point.y = eye_centre.y + ratio * (observed_point.y - eye_centre.y);
        nodal_point.z = eye_centre.z + ratio * (observed_point.z - eye_centre.z);
        return nodal_point;
    }

    cv::Mat Utils::findTransformationMatrix(const cv::Mat &mat_from, const cv::Mat &mat_to)
    {
        // Calculate pseudo inverse of mat_from
        cv::Mat mat_from_inv{};
        cv::invert(mat_from, mat_from_inv, cv::DECOMP_SVD);

        // Calculate transformation matrix -> mat_from * transformation_matrix = mat_to
        cv::Mat transformation_matrix = mat_from_inv * mat_to;
        return transformation_matrix;
    }

    cv::Mat Utils::convertToHomogeneous(cv::Mat mat)
    {
        cv::Mat mat_homogeneous(mat.rows, mat.cols + 1, CV_64F);
        for (int i = 0; i < mat.rows; i++)
        {
            mat_homogeneous.at<double>(i, 0) = mat.at<double>(i, 0);
            mat_homogeneous.at<double>(i, 1) = mat.at<double>(i, 1);
            mat_homogeneous.at<double>(i, 2) = mat.at<double>(i, 2);
            mat_homogeneous.at<double>(i, 3) = 1.0;
        }
        return mat_homogeneous;
    }

    bool
    Utils::getRaySphereIntersection(const cv::Vec3d &ray_pos, const cv::Vec3d &ray_dir, const cv::Vec3d &sphere_pos,
                                    double sphere_radius, double &t)
    {
        double A{ray_dir.dot(ray_dir)};
        cv::Vec3d v{ray_pos - sphere_pos};
        double B{2 * v.dot(ray_dir)};
        double C{v.dot(v) - sphere_radius * sphere_radius};
        double delta{B * B - 4 * A * C};
        if (delta > 0)
        {
            double t1{(-B - std::sqrt(delta)) / (2 * A)};
            double t2{(-B + std::sqrt(delta)) / (2 * A)};
            if (t1 < 1e-5)
            {
                t = t2;
            }
            else if (t1 < 1e-5)
            {
                t = t1;
            }
            else
            {
                t = std::min(t1, t2);
            }
        }
        return (delta > 0);
    }

    cv::Point3d Utils::visualToOpticalAxis(const cv::Point3d &visual_axis, double alpha, double beta)
    {
        static int initialized = false;
        static OpticalFromVisualAxisOptimizer *optimizer{};
        static cv::Ptr<cv::DownhillSolver::Function> minimizer_function{};
        static cv::Ptr<cv::DownhillSolver> solver{};
        if (!initialized)
        {
            optimizer = new OpticalFromVisualAxisOptimizer();
            minimizer_function = cv::Ptr<cv::DownhillSolver::Function>{optimizer};
            solver = cv::DownhillSolver::create();
            solver->setFunction(minimizer_function);
            cv::Mat step = (cv::Mat_<double>(1, 2) << 0.1, 0.1);
            solver->setInitStep(step);
            initialized = true;
        }

        optimizer->setParameters(alpha, beta, visual_axis);
        cv::Mat x = (cv::Mat_<double>(1, 2) << 0, 0);
        solver->minimize(x);
        double phi = x.at<double>(0, 0);
        double theta = x.at<double>(0, 1);
        cv::Mat R = Utils::getRotY(theta) * Utils::getRotX(phi);
        cv::Mat optical_axis_mat = R * (cv::Mat_<double>(3, 1) << 0, 0, -1);
        cv::Point3d optical_axis = cv::Point3d(optical_axis_mat.at<double>(0, 0), optical_axis_mat.at<double>(1, 0),
                                               optical_axis_mat.at<double>(2, 0));
        optical_axis = optical_axis / cv::norm(optical_axis);
        return optical_axis;
    }

    cv::Point3d Utils::opticalToVisualAxis(const cv::Point3d &optical_axis, double alpha, double beta)
    {
        cv::Point3d norm_optical_axis = optical_axis / cv::norm(optical_axis);

        double theta = std::atan2(norm_optical_axis.x, norm_optical_axis.z) - M_PI;
        double phi = -(std::acos(norm_optical_axis.y) - M_PI / 2);

        cv::Mat R = getRotY(theta) * getRotX(phi);
        cv::Mat right = R * (cv::Mat_<double>(3, 1) << 1, 0, 0);
        cv::Mat up = R * (cv::Mat_<double>(3, 1) << 0, 1, 0);
        cv::Mat forward = (cv::Mat_<double>(3, 1) << norm_optical_axis.x, norm_optical_axis.y, norm_optical_axis.z);

        R = Utils::convertAxisAngleToRotationMatrix(up, -alpha * M_PI / 180);
        forward = R * forward;
        right = R * right;

        R = Utils::convertAxisAngleToRotationMatrix(right, -beta * M_PI / 180);
        forward = R * forward;

        cv::Point3d visual_axis = {forward.at<double>(0, 0), forward.at<double>(1, 0), forward.at<double>(2, 0)};
        visual_axis = visual_axis / cv::norm(visual_axis);
        return visual_axis;
    }

    double Utils::pointToLineDistance(cv::Vec3d origin, cv::Vec3d direction, cv::Vec3d point)
    {
        cv::Vec3d V = origin - point;
        cv::Vec3d projection = V.dot(direction) / (cv::norm(direction) * cv::norm(direction)) * direction;
        return cv::norm(V - projection);
    }

    cv::Point2d Utils::findEllipseIntersection(cv::RotatedRect &ellipse, double angle)
    {
        double a = ellipse.size.width / 2.0;
        double b = ellipse.size.height / 2.0;
        double x0 = ellipse.center.x;
        double y0 = ellipse.center.y;
        double theta = ellipse.angle * M_PI / 180;
        double cos_theta = cos(theta);
        double sin_theta = sin(theta);
        double cos_angle = cos(angle);
        double sin_angle = sin(angle);
        double x = x0 + a * cos_theta * cos_angle - b * sin_theta * sin_angle;
        double y = y0 + a * sin_theta * cos_angle + b * cos_theta * sin_angle;
        return cv::Point2d(x, y);
    }

    double Utils::getAngleBetweenVectors(cv::Vec3d a, cv::Vec3d b)
    {
        double dot = a.dot(b);
        double det = a[0] * b[1] - a[1] * b[0];
        if (dot == 0 && det == 0)
        {
            return M_PI;
        }

        return atan2(det, dot);
    }

    void Utils::getCrossValidationIndices(std::vector<int>& indices, int n_data_points, int n_folds)
    {
        indices.resize(n_data_points);
        for (int i = 0; i < n_data_points; i++)
        {
            indices[i] = i % n_folds;
        }
        std::random_shuffle(indices.begin(), indices.end());
    }

    cv::Point3d
    Utils::findGridIntersection(std::vector<cv::Point3d> &front_corners, std::vector<cv::Point3d> &back_corners)
    {
        int n = front_corners.size();
        cv::Mat S = cv::Mat::zeros(3, 3, CV_64F);
        cv::Mat C = cv::Mat::zeros(3, 1, CV_64F);
        cv::Mat directions = cv::Mat::zeros(n, 3, CV_64F);
        cv::Mat origins = cv::Mat::zeros(n, 3, CV_64F);

        for (int i = 0; i < n; i++)
        {
            cv::Vec3d optical_axis = {back_corners[i].x - front_corners[i].x, back_corners[i].y - front_corners[i].y,
                                     back_corners[i].z - front_corners[i].z};
            double norm = std::sqrt(optical_axis[0] * optical_axis[0] + optical_axis[1] * optical_axis[1] +
                                   optical_axis[2] * optical_axis[2]);
            optical_axis[0] /= norm;
            optical_axis[1] /= norm;
            optical_axis[2] /= norm;
            directions.at<double>(i, 0) = optical_axis[0];
            directions.at<double>(i, 1) = optical_axis[1];
            directions.at<double>(i, 2) = optical_axis[2];

            origins.at<double>(i, 0) = back_corners[i].x;
            origins.at<double>(i, 1) = back_corners[i].y;
            origins.at<double>(i, 2) = back_corners[i].z;
        }

        cv::Mat eye = cv::Mat::eye(3, 3, CV_64F);

        for (int i = 0; i < 4; i++)
        {
            S += eye - directions.row(i).t() * directions.row(i);
            C += (eye - directions.row(i).t() * directions.row(i)) * origins.row(i).t();
        }

        cv::Mat intersection = S.inv(cv::DECOMP_SVD) * C;
        cv::Point3d cross_point = {intersection.at<double>(0, 0), intersection.at<double>(1, 0), intersection.at<double>(2, 0)};
        return cross_point;
    }

    cv::Mat Utils::getRotX(double angle)
    {
        cv::Mat rot_x = cv::Mat::eye(3, 3, CV_64F);
        rot_x.at<double>(1, 1) = std::cos(angle);
        rot_x.at<double>(1, 2) = -std::sin(angle);
        rot_x.at<double>(2, 1) = std::sin(angle);
        rot_x.at<double>(2, 2) = std::cos(angle);
        return rot_x;
    }

    cv::Mat Utils::getRotY(double angle)
    {
        cv::Mat rot_y = cv::Mat::eye(3, 3, CV_64F);
        rot_y.at<double>(0, 0) = std::cos(angle);
        rot_y.at<double>(0, 2) = std::sin(angle);
        rot_y.at<double>(2, 0) = -std::sin(angle);
        rot_y.at<double>(2, 2) = std::cos(angle);
        return rot_y;
    }

    cv::Mat Utils::getRotZ(double angle)
    {
        cv::Mat rot_z = cv::Mat::eye(3, 3, CV_64F);
        rot_z.at<double>(0, 0) = std::cos(angle);
        rot_z.at<double>(0, 1) = -std::sin(angle);
        rot_z.at<double>(1, 0) = std::sin(angle);
        rot_z.at<double>(1, 1) = std::cos(angle);
        return rot_z;
    }

    cv::Mat Utils::convertAxisAngleToRotationMatrix(cv::Mat axis, double angle)
    {
        cv::Mat K = cv::Mat::zeros(3, 3, CV_64F);
        K.at<double>(0, 1) = -axis.at<double>(2);
        K.at<double>(0, 2) = axis.at<double>(1);
        K.at<double>(1, 0) = axis.at<double>(2);
        K.at<double>(1, 2) = -axis.at<double>(0);
        K.at<double>(2, 0) = -axis.at<double>(1);
        K.at<double>(2, 1) = axis.at<double>(0);

        cv::Mat R = cv::Mat::eye(3, 3, CV_64F) + std::sin(angle) * K + (1 - std::cos(angle)) * K * K;
        return R;
    }

} // namespace et