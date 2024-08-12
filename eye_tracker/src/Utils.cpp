#include "eye_tracker/Utils.hpp"

#include <chrono>
#include <ctime>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <random>
#include <sstream>
#include <numeric>
#include <iostream>

namespace et
{
    std::mt19937::result_type Utils::seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 Utils::gen = std::mt19937(seed);

    std::vector<std::vector<double>> Utils::readCsv(const std::string& filename, bool ignore_first_line)
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

    void Utils::writeCsv(std::vector<std::vector<double>>& data, const std::string& filename, const bool append, const std::string& header)
    {
        std::ofstream file;
        if (append)
        {
            file.open(filename, std::ios_base::app);
        }
        else
        {
            file.open(filename);
        }
        if (!header.empty() && !append) {
            file << header << "\n";
        }
        for (auto& row: data)
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
        const std::time_t now{std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())};
        static char buffer[80];

        std::strftime(buffer, 80, "%Y-%m-%d_%H-%M-%S", std::localtime(&now));
        std::string s{buffer};
        return s;
    }

    bool
    Utils::getRaySphereIntersection(const cv::Vec3d& ray_pos, const cv::Vec3d& ray_dir, const cv::Vec3d& sphere_pos,
                                    const double sphere_radius, double& t)
    {
        const double A{ray_dir.dot(ray_dir)};
        const cv::Vec3d v{ray_pos - sphere_pos};
        const double B{2 * v.dot(ray_dir)};
        const double C{v.dot(v) - sphere_radius * sphere_radius};
        const double delta{B * B - 4 * A * C};
        if (delta > 0)
        {
            const double t1{(-B - std::sqrt(delta)) / (2 * A)};
            const double t2{(-B + std::sqrt(delta)) / (2 * A)};
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
        return delta > 0;
    }

    cv::Point3d Utils::opticalToVisualAxis(const cv::Point3d& optical_axis, const double alpha, const double beta)
    {
        const cv::Point3d norm_optical_axis = optical_axis / cv::norm(optical_axis);

        double theta = std::atan2(-norm_optical_axis.x, -norm_optical_axis.z);
        double phi = std::acos(norm_optical_axis.y);

        theta -= alpha * M_PI / 180;
        phi -= beta * M_PI / 180;

        cv::Point3d visual_axis;
        visual_axis.x = -std::sin(phi) * std::sin(theta);
        visual_axis.y = std::cos(phi);
        visual_axis.z = -std::sin(phi) * std::cos(theta);

        return visual_axis;
    }

    double Utils::getAngleBetweenVectors(cv::Vec3d a, cv::Vec3d b)
    {
        const double dot = a.dot(b);
        const double det = a[0] * b[1] - a[1] * b[0];
        if (dot == 0 && det == 0)
        {
            return M_PI;
        }

        return atan2(det, dot);
    }

    cv::Vec3d Utils::getRefractedRay(const cv::Vec3d& direction, const cv::Vec3d& normal, const double refraction_index)
    {
        const double nr{1 / refraction_index};
        const double m_cos{(-direction).dot(normal)};
        const double m_sin{nr * nr * (1 - m_cos * m_cos)};
        cv::Vec3d t{nr * (direction + m_cos * normal) - std::sqrt(1 - m_sin) * normal};
        cv::normalize(t, t);
        return t;
    }

    cv::Point3d Utils::getTrimmmedMean(std::vector<cv::Point3d> const& values, const double trim_ratio) {
        std::vector<double> x_values;
        std::vector<double> y_values;
        std::vector<double> z_values;
        for (auto const& value : values) {
            x_values.push_back(value.x);
            y_values.push_back(value.y);
            z_values.push_back(value.z);
        }
        std::sort(x_values.begin(), x_values.end());
        std::sort(y_values.begin(), y_values.end());
        std::sort(z_values.begin(), z_values.end());
        const int trim_size = static_cast<int>(static_cast<int>(values.size()) * trim_ratio / 2);
        x_values.erase(x_values.begin(), x_values.begin() + trim_size);
        x_values.erase(x_values.end() - trim_size, x_values.end());
        y_values.erase(y_values.begin(), y_values.begin() + trim_size);
        y_values.erase(y_values.end() - trim_size, y_values.end());
        z_values.erase(z_values.begin(), z_values.begin() + trim_size);
        z_values.erase(z_values.end() - trim_size, z_values.end());
        return {Utils::getMean(x_values), Utils::getMean(y_values), Utils::getMean(z_values)};
    }

    cv::Vec3d Utils::getMedian(std::vector<cv::Vec3d> const& values) {
        std::vector<double> x_values;
        std::vector<double> y_values;
        std::vector<double> z_values;
        for (auto const& value : values) {
            x_values.push_back(value[0]);
            y_values.push_back(value[1]);
            z_values.push_back(value[2]);
        }
        std::sort(x_values.begin(), x_values.end());
        std::sort(y_values.begin(), y_values.end());
        std::sort(z_values.begin(), z_values.end());
        return cv::Point3d(x_values[values.size() / 2], y_values[values.size() / 2], z_values[values.size() / 2]);
    }

    void Utils::vectorToAngles(cv::Vec3d vector, cv::Vec2d& angles)
    {
        const double norm = cv::norm(vector);
        vector = vector / norm;
        angles[0] = std::atan2(-vector[0], -vector[2]);
        angles[1] = std::acos(vector[1]);
    }

    void Utils::anglesToVector(cv::Vec2d angles, cv::Vec3d& vector)
    {
        double x = -std::sin(angles[1]) * std::sin(angles[0]);
        double y = std::cos(angles[1]);
        double z = -std::sin(angles[1]) * std::cos(angles[0]);
        vector = {x, y, z};
    }




} // namespace et
