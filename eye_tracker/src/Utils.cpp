#include "eye_tracker/Utils.hpp"

#include <chrono>
#include <ctime>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <sstream>
#include <numeric>

namespace et
{
    std::mt19937::result_type Utils::seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 Utils::gen = std::mt19937(seed);

    std::vector<std::vector<float>> Utils::readFloatColumnsCsv(const std::string &filename, bool ignore_first_line)
    {
        std::ifstream input_file{filename};
        if (!input_file.is_open())
        {
            return {};
        }

        std::string line{};
        std::vector<std::vector<float>> csv_data{};
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

    std::vector<std::vector<float>> Utils::readFloatRowsCsv(const std::string &filename, bool ignore_first_line)
    {
        std::ifstream input_file{filename};
        if (!input_file.is_open())
        {
            return {};
        }

        std::string line{};
        std::vector<std::vector<float>> csv_data{};

        if (ignore_first_line)
        {
            std::getline(input_file, line);
        }

        while (std::getline(input_file, line))
        {
            std::vector<float> row{};
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

    void Utils::writeFloatCsv(std::vector<std::vector<float>> &data, const std::string &filename)
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

    cv::Point3f Utils::calculateNodalPointPosition(cv::Point3_<float> observed_point, cv::Point3_<float> eye_centre,
                                                   float nodal_dist)
    {
        cv::Point3f nodal_point{};
        float dist = cv::norm(observed_point - eye_centre);
        float ratio = nodal_dist / dist;
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
        cv::Mat mat_homogeneous(mat.rows, mat.cols + 1, mat.type());
        for (int i = 0; i < mat.rows; i++)
        {
            mat_homogeneous.at<float>(i, 0) = mat.at<float>(i, 0);
            mat_homogeneous.at<float>(i, 1) = mat.at<float>(i, 1);
            mat_homogeneous.at<float>(i, 2) = mat.at<float>(i, 2);
            mat_homogeneous.at<float>(i, 3) = 1.0;

        }
        return mat_homogeneous;
    }

    bool
    Utils::getRaySphereIntersection(const cv::Vec3f &ray_pos, const cv::Vec3d &ray_dir, const cv::Vec3f &sphere_pos,
                                    double sphere_radius, double &t)
    {
        double A{ray_dir.dot(ray_dir)};
        cv::Vec3f v{ray_pos - sphere_pos};
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

    cv::Point3f Utils::visualToOpticalAxis(const cv::Point3f &visual_axis, float alpha, float beta)
    {
        float alpha_r = alpha * M_PI / 180;
        float beta_r = beta * M_PI / 180;

        float phi = std::asin(visual_axis.y);

        float inside_val = visual_axis.x / std::cos(phi);
        if (inside_val > 1.0) {
            inside_val = 1.0;
        } else if (inside_val < -1.0) {
            inside_val = -1.0;
        }

        float theta = std::asin(inside_val);

        theta -= alpha_r;
        phi -= beta_r;

        cv::Point3f optical_axis = {std::cos(phi) * std::sin(theta), std::sin(phi),
                                    -std::cos(phi) * std::cos(theta)};
        optical_axis = optical_axis / cv::norm(optical_axis);
        return optical_axis;
    }

    cv::Point3f Utils::opticalToVisualAxis(const cv::Point3f &optical_axis, float alpha, float beta)
    {
        float alpha_r = alpha * M_PI / 180;
        float beta_r = beta * M_PI / 180;

        float phi = std::asin(optical_axis.y);

        float inside_val = optical_axis.x / std::cos(phi);
        if (inside_val > 1.0) {
            inside_val = 1.0;
        } else if (inside_val < -1.0) {
            inside_val = -1.0;
        }

        float theta = std::asin(inside_val);

        theta += alpha_r;
        phi += beta_r;

        cv::Point3f visual_axis = {std::cos(phi) * std::sin(theta), std::sin(phi),
                                   -std::cos(phi) * std::cos(theta)};
        visual_axis = visual_axis / cv::norm(visual_axis);
        return visual_axis;
    }

    float Utils::pointToLineDistance(cv::Vec3f origin, cv::Vec3f direction, cv::Vec3f point)
    {
        cv::Vec3f V = origin - point;
        cv::Vec3f projection = V.dot(direction) / (cv::norm(direction) * cv::norm(direction)) * direction;
        return cv::norm(V - projection);
    }

    cv::Point2f Utils::findEllipseIntersection(cv::RotatedRect &ellipse, float angle)
    {
        float a = ellipse.size.width / 2;
        float b = ellipse.size.height / 2;
        float x0 = ellipse.center.x;
        float y0 = ellipse.center.y;
        float theta = ellipse.angle * M_PI / 180;
        float cos_theta = cosf(theta);
        float sin_theta = sinf(theta);
        float cos_angle = cosf(angle);
        float sin_angle = sinf(angle);
        float x = x0 + a * cos_theta * cos_angle - b * sin_theta * sin_angle;
        float y = y0 + a * sin_theta * cos_angle + b * cos_theta * sin_angle;
        return cv::Point2f(x, y);
    }

    float Utils::getAngleBetweenVectors(cv::Vec3f a, cv::Vec3f b)
    {
        float dot = a.dot(b);
        float det = a[0] * b[1] - a[1] * b[0];
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

    cv::Point3f
    Utils::findGridIntersection(std::vector<cv::Point3f> &front_corners, std::vector<cv::Point3f> &back_corners)
    {
        int n = front_corners.size();
        cv::Mat S = cv::Mat::zeros(3, 3, CV_32F);
        cv::Mat C = cv::Mat::zeros(3, 1, CV_32F);
        cv::Mat directions = cv::Mat::zeros(n, 3, CV_32F);
        cv::Mat origins = cv::Mat::zeros(n, 3, CV_32F);

        for (int i = 0; i < n; i++)
        {
            cv::Vec3f optical_axis = {back_corners[i].x - front_corners[i].x, back_corners[i].y - front_corners[i].y,
                                     back_corners[i].z - front_corners[i].z};
            float norm = std::sqrt(optical_axis[0] * optical_axis[0] + optical_axis[1] * optical_axis[1] +
                                   optical_axis[2] * optical_axis[2]);
            optical_axis[0] /= norm;
            optical_axis[1] /= norm;
            optical_axis[2] /= norm;
            directions.at<float>(i, 0) = optical_axis[0];
            directions.at<float>(i, 1) = optical_axis[1];
            directions.at<float>(i, 2) = optical_axis[2];

            origins.at<float>(i, 0) = back_corners[i].x;
            origins.at<float>(i, 1) = back_corners[i].y;
            origins.at<float>(i, 2) = back_corners[i].z;
        }

        cv::Mat eye = cv::Mat::eye(3, 3, CV_32F);

        for (int i = 0; i < 4; i++)
        {
            S += eye - directions.row(i).t() * directions.row(i);
            C += (eye - directions.row(i).t() * directions.row(i)) * origins.row(i).t();
        }

        cv::Mat intersection = S.inv(cv::DECOMP_SVD) * C;
        cv::Point3f cross_point = {intersection.at<float>(0, 0), intersection.at<float>(1, 0), intersection.at<float>(2, 0)};
        return cross_point;
    }

} // namespace et