#include "eye_tracker/Utils.hpp"
#include "eye_tracker/optimizers/OpticalFromVisualAxisOptimizer.hpp"

#include <chrono>
#include <ctime>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <random>
#include <sstream>
#include <numeric>
#include <iostream>
#include <eye_tracker/optimizers/EyeAnglesOptimizer.hpp>

namespace et
{
    std::mt19937::result_type Utils::seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 Utils::gen = std::mt19937(seed);

    std::vector<std::vector<double>> Utils::readFloatColumnsCsv(const std::string& filename, bool ignore_first_line)
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

    std::vector<std::vector<double>> Utils::readFloatRowsCsv(const std::string& filename, bool ignore_first_line)
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

    void Utils::writeFloatCsv(std::vector<std::vector<double>>& data, const std::string& filename, bool append, std::string header)
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
        if (!header.empty()) {
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

    cv::Point3d
    Utils::calculateNodalPointPosition(cv::Point3d observed_point, cv::Point3d eye_centre, double nodal_dist)
    {
        cv::Point3d nodal_point{};
        double dist = cv::norm(observed_point - eye_centre);
        double ratio = nodal_dist / dist;
        nodal_point.x = eye_centre.x + ratio * (observed_point.x - eye_centre.x);
        nodal_point.y = eye_centre.y + ratio * (observed_point.y - eye_centre.y);
        nodal_point.z = eye_centre.z + ratio * (observed_point.z - eye_centre.z);
        return nodal_point;
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

    cv::Mat Utils::convertToHomogeneous(cv::Point3d point, double w)
    {
        cv::Mat mat_homogeneous(1, 4, CV_64F);
        mat_homogeneous.at<double>(0, 0) = point.x;
        mat_homogeneous.at<double>(0, 1) = point.y;
        mat_homogeneous.at<double>(0, 2) = point.z;
        mat_homogeneous.at<double>(0, 3) = w;
        return mat_homogeneous;
    }

    bool
    Utils::getRaySphereIntersection(const cv::Vec3d& ray_pos, const cv::Vec3d& ray_dir, const cv::Vec3d& sphere_pos,
                                    double sphere_radius, double& t)
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

    cv::Point3d Utils::visualToOpticalAxis(const cv::Point3d& visual_axis, double alpha, double beta)
    {
        cv::Point3d norm_visual_axis = visual_axis / cv::norm(visual_axis);

        double theta = std::atan2(-norm_visual_axis.x, -norm_visual_axis.z);
        double phi = std::acos(norm_visual_axis.y);

        theta += alpha * M_PI / 180;
        phi += beta * M_PI / 180;

        cv::Point3d optical_axis;
        optical_axis.x = -std::sin(phi) * std::sin(theta);
        optical_axis.y = std::cos(phi);
        optical_axis.z = -std::sin(phi) * std::cos(theta);

        return optical_axis;
    }

    cv::Point3d Utils::visualToOpticalAxis2(const cv::Point3d& visual_axis, double alpha, double beta)
    {
        static int initialized = false;
        static OpticalFromVisualAxisOptimizer* optimizer{};
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
            solver->setTermCriteria(cv::TermCriteria((cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS), 1e3, 1e-12));
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

    cv::Point3d Utils::opticalToVisualAxis(const cv::Point3d& optical_axis, double alpha, double beta)
    {
        cv::Point3d norm_optical_axis = optical_axis / cv::norm(optical_axis);

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

    cv::Point3d Utils::opticalToVisualAxis2(const cv::Point3d& optical_axis, double alpha, double beta)
    {
        cv::Point3d norm_optical_axis = optical_axis / cv::norm(optical_axis);


        cv::Point3d visual_axis;
        visual_axis.x = std::cos(-beta * M_PI / 180) * std::cos(alpha * M_PI / 180);
        visual_axis.y = std::sin(-beta * M_PI / 180);
        visual_axis.z = std::cos(-beta * M_PI / 180) * std::sin(alpha * M_PI / 180);

        return visual_axis;
    }

    static std::complex<double> complex_sqrt(const std::complex<double>& z)
    {
        return pow(z, 1. / 2.);
    }

    static std::complex<double> complex_cbrt(const std::complex<double>& z)
    {
        return pow(z, 1. / 3.);
    }

    int Utils::solveQuartic(double r0, double r1, double r2, double r3, double r4, double* roots)
    {
        // Solve the quartic equation r4 * x^4 + r3 * x^3 + r2 * x^2 + r1 * x + r0 = 0

        const std::complex<double> a = std::complex<double>(r4, 0);
        const std::complex<double> b = r3 / a;
        const std::complex<double> c = r2 / a;
        const std::complex<double> d = r1 / a;
        const std::complex<double> e = r0 / a;

        const std::complex<double> Q1 = c * c - 3. * b * d + 12. * e;
        const std::complex<double> Q2 = 2. * c * c * c - 9. * b * c * d + 27. * d * d + 27. * b * b * e - 72. * c * e;
        const std::complex<double> Q3 = 8. * b * c - 16. * d - 2. * b * b * b;
        const std::complex<double> Q4 = 3. * b * b - 8. * c;

        const std::complex<double> Q5 = complex_cbrt(Q2 / 2. + complex_sqrt(Q2 * Q2 / 4. - Q1 * Q1 * Q1));
        const std::complex<double> Q6 = (Q1 / Q5 + Q5) / 3.;
        const std::complex<double> Q7 = 2. * complex_sqrt(Q4 / 12. + Q6);

        std::complex<double> roots_complex[4];

        roots_complex[0] = (-b - Q7 - complex_sqrt(4. * Q4 / 6. - 4. * Q6 - Q3 / Q7)) / 4.;
        roots_complex[1] = (-b - Q7 + complex_sqrt(4. * Q4 / 6. - 4. * Q6 - Q3 / Q7)) / 4.;
        roots_complex[2] = (-b + Q7 - complex_sqrt(4. * Q4 / 6. - 4. * Q6 + Q3 / Q7)) / 4.;
        roots_complex[3] = (-b + Q7 + complex_sqrt(4. * Q4 / 6. - 4. * Q6 + Q3 / Q7)) / 4.;

        int n_real_roots = 0;
        for (int i = 0; i < 4; i++)
        {
            if (std::abs(roots_complex[i].imag()) < 1e-10)
            {
                roots[n_real_roots] = roots_complex[i].real();
                n_real_roots++;
            }
        }
        return n_real_roots;
    }

    cv::Vec2d Utils::project3Dto2D(cv::Vec3d point, cv::Vec3d normal)
    {
        cv::Vec3d x = {1, 0, 0};
        cv::Vec3d projected_x = x - x.dot(normal) * normal;
        projected_x /= cv::norm(projected_x);
        cv::Vec3d projected_y = normal.cross(projected_x);
        cv::Vec2d point2d = {projected_x.dot(point), projected_y.dot(point)};
        return point2d;
    }

    cv::Vec3d Utils::unproject2Dto3D(cv::Vec2d point, cv::Vec3d normal, double depth)
    {
        cv::Vec3d x = {1, 0, 0};
        cv::Vec3d projected_x = x - x.dot(normal) * normal;
        projected_x /= cv::norm(projected_x);
        cv::Vec3d projected_y = normal.cross(projected_x);

        cv::Mat A = (cv::Mat_<double>(3, 3) << projected_x[0], projected_x[1], projected_x[2], projected_y[0],
                projected_y[1], projected_y[2], normal[0], normal[1], normal[2]);


        cv::Mat point3d_mat = A.inv(cv::DECOMP_SVD) * (cv::Mat_<double>(2, 1) << point[0], point[1], depth);
        cv::Vec3d point3d = {point3d_mat.at<double>(0, 0), point3d_mat.at<double>(1, 0), point3d_mat.at<double>(2, 0)};
        return point3d;
    }

    cv::Point2d Utils::getReflectionPoint(cv::Vec2d source, cv::Vec2d destination)
    {
        cv::Vec2d A = {source[0], source[1]};
        cv::Vec2d B = {destination[0], destination[1]};


        double u = A[0] * A[0] + A[1] * A[1];
        double v = A[0] * B[0] + A[1] * B[1];
        double w = B[0] * B[0] + B[1] * B[1];

        double r0 = w - 1;
        double r1 = 2 * (w - v);
        double r2 = u + 2 * v + w - 4 * u * w;
        double r3 = 4 * (v * v - u * w);
        double r4 = 4 * u * (u * w - v * v);
        double roots[4];
        int roots_num = Utils::solveQuartic(r0, r1, r2, r3, r4, roots);
        double x, y;
        for (int i = 0; i < roots_num; i++)
        {
            x = roots[i];
            if (x > 0)
            {
                y = (1 + x - 2 * u * x * x) / (1 + 2 * v * x);
                if (y > 0)
                {
                    break;
                }
            }
        }

        return x * A + y * B;
    }

    cv::Point3d Utils::getReflectionPoint(cv::Vec3d source, cv::Vec3d sphere_centre,
                                          cv::Vec3d destination, double radius)
    {
        source = (source - sphere_centre) / radius;
        destination = (destination - sphere_centre) / radius;
        cv::Vec3d plane_normal = (source - destination).cross(source);
        plane_normal /= cv::norm(plane_normal);
        double depth = plane_normal.dot(source);

        cv::Vec2d source_2d = project3Dto2D(source, plane_normal);
        cv::Vec2d destination_2d = project3Dto2D(destination, plane_normal);
        cv::Vec2d reflection_2d = getReflectionPoint(source_2d, destination_2d);
        cv::Vec3d reflection = unproject2Dto3D(reflection_2d, plane_normal, depth) * radius + sphere_centre;
        return reflection;
    }

    double Utils::pointToLineDistance(cv::Vec3d origin, cv::Vec3d direction, cv::Vec3d point)
    {
        cv::Vec3d V = origin - point;
        cv::Vec3d projection = V.dot(direction) / (cv::norm(direction) * cv::norm(direction)) * direction;
        return cv::norm(V - projection);
    }

    cv::Point2d Utils::findEllipseIntersection(cv::RotatedRect& ellipse, double angle)
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
        std::shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device()()));
    }

    cv::Point3d
    Utils::findGridIntersection(std::vector<cv::Point3d>& front_corners, std::vector<cv::Point3d>& back_corners)
    {
        int n = front_corners.size();
        cv::Mat S = cv::Mat::zeros(3, 3, CV_64F);
        cv::Mat C = cv::Mat::zeros(3, 1, CV_64F);
        cv::Mat directions = cv::Mat::zeros(n, 3, CV_64F);
        cv::Mat origins = cv::Mat::zeros(n, 3, CV_64F);

        for (int i = 0; i < n; i++)
        {
            cv::Vec3d optical_axis = {
                    back_corners[i].x - front_corners[i].x, back_corners[i].y - front_corners[i].y,
                    back_corners[i].z - front_corners[i].z
            };
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
        cv::Point3d cross_point = {
                intersection.at<double>(0, 0), intersection.at<double>(1, 0),
                intersection.at<double>(2, 0)
        };
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

    cv::Mat Utils::getRotXd(double angle)
    {
        return getRotX(angle * M_PI / 180);
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

    cv::Mat Utils::getRotYd(double angle)
    {
        return getRotY(angle * M_PI / 180);
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

    cv::Mat Utils::getRotZd(double angle)
    {
        return getRotZ(angle * M_PI / 180);
    }

    cv::Mat Utils::convertAxisAngleToRotationMatrix(cv::Mat axis, double angle)
    {
        return convertAxisAngleToRotationMatrix(cv::Point3d(axis.at<double>(0, 0), axis.at<double>(1, 0), axis.at<double>(2, 0)), angle);
    }

    cv::Mat Utils::convertAxisAngleToRotationMatrix(cv::Point3d axis, double angle)
    {
        cv::Mat R = cv::Mat::eye(3, 3, CV_64F);
        R.at<double>(0, 0) = std::cos(angle) + axis.x * axis.x * (1 - std::cos(angle));
        R.at<double>(0, 1) = axis.x * axis.y * (1 - std::cos(angle)) - axis.z * std::sin(angle);
        R.at<double>(0, 2) = axis.x * axis.z * (1 - std::cos(angle)) + axis.y * std::sin(angle);
        R.at<double>(1, 0) = axis.y * axis.x * (1 - std::cos(angle)) + axis.z * std::sin(angle);
        R.at<double>(1, 1) = std::cos(angle) + axis.y * axis.y * (1 - std::cos(angle));
        R.at<double>(1, 2) = axis.y * axis.z * (1 - std::cos(angle)) - axis.x * std::sin(angle);
        R.at<double>(2, 0) = axis.z * axis.x * (1 - std::cos(angle)) - axis.y * std::sin(angle);
        R.at<double>(2, 1) = axis.z * axis.y * (1 - std::cos(angle)) + axis.x * std::sin(angle);
        R.at<double>(2, 2) = std::cos(angle) + axis.z * axis.z * (1 - std::cos(angle));
        return R;
    }

    cv::Vec3d Utils::getRefractedRay(const cv::Vec3d& direction, const cv::Vec3d& normal, double refraction_index)
    {
        double nr{1 / refraction_index};
        double m_cos{(-direction).dot(normal)};
        double m_sin{nr * nr * (1 - m_cos * m_cos)};
        cv::Vec3d t{nr * (direction + m_cos * normal) - std::sqrt(1 - m_sin) * normal};
        cv::normalize(t, t);
        return t;
    }

    cv::Vec3d Utils::getTrimmmedMean(std::vector<cv::Vec3d> const& values, double trim_ratio) {
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
        int trim_size = (int) (values.size() * trim_ratio / 2);
        x_values.erase(x_values.begin(), x_values.begin() + trim_size);
        x_values.erase(x_values.end() - trim_size, x_values.end());
        y_values.erase(y_values.begin(), y_values.begin() + trim_size);
        y_values.erase(y_values.end() - trim_size, y_values.end());
        z_values.erase(z_values.begin(), z_values.begin() + trim_size);
        z_values.erase(z_values.end() - trim_size, z_values.end());
        return {Utils::getMean(x_values), Utils::getMean(y_values), Utils::getMean(z_values)};
    }

    cv::Point3d Utils::getTrimmmedMean(std::vector<cv::Point3d> const& values, double trim_ratio) {
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
        int trim_size = (int) (values.size() * trim_ratio / 2);
        x_values.erase(x_values.begin(), x_values.begin() + trim_size);
        x_values.erase(x_values.end() - trim_size, x_values.end());
        y_values.erase(y_values.begin(), y_values.begin() + trim_size);
        y_values.erase(y_values.end() - trim_size, y_values.end());
        z_values.erase(z_values.begin(), z_values.begin() + trim_size);
        z_values.erase(z_values.end() - trim_size, z_values.end());
        return {Utils::getMean(x_values), Utils::getMean(y_values), Utils::getMean(z_values)};
    }


    std::vector<int> Utils::getOutliers(std::vector<cv::Point3d> const& values, double threshold) {
        std::vector<int> outliers{};
        cv::Point3d mean = Utils::getMean<cv::Point3d>(values);
        cv::Point3d std = Utils::getStdDev(values);
        for (int i = 0; i < values.size(); i++) {
            if (std::abs(values[i].x - mean.x) > threshold * std.x ||
                std::abs(values[i].y - mean.y) > threshold * std.y ||
                std::abs(values[i].z - mean.z) > threshold * std.z) {
                outliers.push_back(i);
            }
        }
        return outliers;
    }

    std::vector<int> Utils::getOutliers(std::vector<cv::Point2d> const& values, double threshold) {
        std::vector<int> outliers{};
        cv::Point2d mean = Utils::getMean<cv::Point2d>(values);
        cv::Point2d std = Utils::getStdDev(values);
        for (int i = 0; i < values.size(); i++) {
            if (std::abs(values[i].x - mean.x) > threshold * std.x ||
                std::abs(values[i].y - mean.y) > threshold * std.y) {
                outliers.push_back(i);
            }
        }
        return outliers;
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

    double Utils::getPercentile(const std::vector<double>& values, double percentile)
    {
        int index = (int) (percentile * values.size());
        std::vector<double> sorted_values = values;
        std::sort(sorted_values.begin(), sorted_values.end());
        return sorted_values[index];
    }

    cv::Point3d Utils::getStdDev(const std::vector<cv::Point3d>& values)
    {
        cv::Point3d mean = Utils::getMean<cv::Point3d>(values);
        cv::Point3d std{};
        for (int i = 0; i < values.size(); i++)
        {
            std.x += (values[i].x - mean.x) * (values[i].x - mean.x);
            std.y += (values[i].y - mean.y) * (values[i].y - mean.y);
            std.z += (values[i].z - mean.z) * (values[i].z - mean.z);
        }
        std.x /= values.size();
        std.y /= values.size();
        std.z /= values.size();
        std.x = std::sqrt(std.x);
        std.y = std::sqrt(std.y);
        std.z = std::sqrt(std.z);
        return std;
    }

    cv::Point2d Utils::getStdDev(const std::vector<cv::Point2d>& values)
    {
        cv::Point2d mean = Utils::getMean<cv::Point2d>(values);
        cv::Point2d std{};
        for (int i = 0; i < values.size(); i++)
        {
            std.x += (values[i].x - mean.x) * (values[i].x - mean.x);
            std.y += (values[i].y - mean.y) * (values[i].y - mean.y);
        }
        std.x /= values.size();
        std.y /= values.size();
        std.x = std::sqrt(std.x);
        std.y = std::sqrt(std.y);
        return std;
    }

    void Utils::getAnglesBetweenVectors(cv::Vec3d a, cv::Vec3d b, double& alpha, double& beta)
    {
        cv::Vec2d a_xz = {a[0], a[2]};
        cv::Vec2d b_xz = {b[0], b[2]};
        a_xz = a_xz / cv::norm(a_xz);
        b_xz = b_xz / cv::norm(b_xz);

        alpha = std::acos(a_xz.dot(b_xz)) * 180 / M_PI;
        if (a_xz[0] * b_xz[1] - a_xz[1] * b_xz[0] < 0)
        {
            alpha = -alpha;
        }

        cv::Vec2d a_yz = {a[1], a[2]};
        cv::Vec2d b_yz = {b[1], b[2]};
        a_yz = a_yz / cv::norm(a_yz);
        b_yz = b_yz / cv::norm(b_yz);
        beta = std::acos(a_yz.dot(b_yz)) * 180 / M_PI;
        if (a_yz[0] * b_yz[1] - a_yz[1] * b_yz[0] < 0)
        {
            beta = -beta;
        }
    }

    void Utils::getVectorFromAngles(double alpha, double beta, cv::Vec3d& vector)
    {
        auto forward = cv::Mat{cv::Vec3d{0, 0, 1}};
        auto right = cv::Mat{cv::Vec3d{-1, 0, 0}};
        auto up = cv::Mat{cv::Vec3d{0, 1, 0}};

        cv::Mat R = convertAxisAngleToRotationMatrix(up, alpha * M_PI / 180);
        forward = R * forward;
        right = R * right;

        R = convertAxisAngleToRotationMatrix(right, beta * M_PI / 180);
        forward = R * forward;
        vector = {forward.at<double>(0, 0), forward.at<double>(1, 0), forward.at<double>(2, 0)};
    }

    void Utils::getAnglesBetweenVectorsAlt(cv::Vec3d visual_axis, cv::Vec3d optical_axis, double& alpha, double& beta)
    {
        static int initialized = false;
        static EyeAnglesOptimizer* optimizer{};
        static cv::Ptr<cv::DownhillSolver::Function> minimizer_function{};
        static cv::Ptr<cv::DownhillSolver> solver{};
        if (!initialized)
        {
            optimizer = new EyeAnglesOptimizer();
            minimizer_function = cv::Ptr<cv::DownhillSolver::Function>{optimizer};
            solver = cv::DownhillSolver::create();
            solver->setFunction(minimizer_function);
            cv::Mat step = (cv::Mat_<double>(1, 2) << 0.1, 0.1);
            solver->setInitStep(step);
            initialized = true;
        }

        cv::Mat x = (cv::Mat_<double>(1, 2) << 0, 0);
        optimizer->setParameters(visual_axis, optical_axis);
        solver->minimize(x);
        alpha = -x.at<double>(0, 0);
        beta = -x.at<double>(0, 1);
    }

    cv::Mat Utils::getTransformationBetweenMatrices(std::vector<cv::Point3d> from, std::vector<cv::Point3d> to)
    {
        cv::Mat from_mat = cv::Mat(from).reshape(1);
        cv::Mat to_mat = cv::Mat(to).reshape(1);

        from_mat = Utils::convertToHomogeneous(from_mat);
        to_mat = Utils::convertToHomogeneous(to_mat);

        cv::Mat transformation_matrix = from_mat.inv(cv::DECOMP_SVD) * to_mat;
        return transformation_matrix;
    }

    cv::Point3d Utils::convertFromHomogeneous(cv::Mat mat)
    {
        cv::Point3d point{
                mat.at<double>(0, 0) / mat.at<double>(0, 3),
                mat.at<double>(0, 1) / mat.at<double>(0, 3),
                mat.at<double>(0, 2) / mat.at<double>(0, 3)
        };
        return point;
    }

    void Utils::vectorToAngles(cv::Vec3d vector, cv::Vec2d& angles)
    {
        double norm = cv::norm(vector);
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
