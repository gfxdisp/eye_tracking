#include "Utils.hpp"

#include <chrono>
#include <ctime>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <sstream>

namespace et {
std::vector<std::vector<float>>
Utils::readFloatColumnsCsv(const std::string &filename) {
    std::ifstream input_file{filename};
    if (!input_file.is_open()) {
        return {};
    }

    std::string line{};
    std::vector<std::vector<float>> csv_data{};
    int line_num{0};
    while (std::getline(input_file, line)) {
        std::string str_value{};
        std::stringstream stream_line{line};
        int val_num{0};
        while (std::getline(stream_line, str_value, ',')) {
            if (line_num == 0) {
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

std::vector<std::vector<float>>
Utils::readFloatRowsCsv(const std::string &filename) {
    std::ifstream input_file{filename};
    if (!input_file.is_open()) {
        return {};
    }

    std::string line{};
    std::vector<std::vector<float>> csv_data{};
    while (std::getline(input_file, line)) {
        std::vector<float> row{};
        std::string str_value{};
        std::stringstream stream_line{line};
        while (std::getline(stream_line, str_value, ',')) {
            row.push_back(std::stof(str_value));
        }
        csv_data.push_back(row);
    }
    input_file.close();
    return csv_data;
}

void Utils::writeFloatCsv(std::vector<std::vector<float>> &data,
                          const std::string &filename) {
    std::ofstream file{filename};
    for (auto &row : data) {
        for (int i = 0; i < row.size(); i++) {
            if (i != 0) {
                file << "," << row[i];
            } else {
                file << row[i];
            }
        }
        file << "\n";
    }
    file.close();
}

std::string Utils::getCurrentTimeText() {
    std::time_t now{
        std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())};
    static char buffer[80];

    std::strftime(buffer, 80, "%Y-%m-%d_%H-%M-%S", std::localtime(&now));
    std::string s{buffer};
    return s;
}

std::vector<std::string> Utils::split(std::string to_split, char i) {
    std::vector<std::string> result{};
    std::stringstream ss{to_split};
    std::string item{};
    while (std::getline(ss, item, i)) {
        result.push_back(item);
    }
    return result;
}

cv::Point3f Utils::calculateNodalPointPosition(cv::Point3_<float> observed_point, cv::Point3_<float> eye_centre,
                                               float nodal_dist) {
    cv::Point3f nodal_point{};
    float dist = cv::norm(observed_point - eye_centre);
    float ratio = nodal_dist / dist;
    nodal_point.x = eye_centre.x + ratio * (observed_point.x - eye_centre.x);
    nodal_point.y = eye_centre.y + ratio * (observed_point.y - eye_centre.y);
    nodal_point.z = eye_centre.z + ratio * (observed_point.z - eye_centre.z);
    return nodal_point;
}

cv::Mat Utils::findTransformationMatrix(const cv::Mat& mat_from, const cv::Mat& mat_to) {
    // Calculate pseudo inverse of mat_from
    cv::Mat mat_from_inv{};
    cv::invert(mat_from, mat_from_inv, cv::DECOMP_SVD);

    // Calculate transformation matrix -> mat_from * transformation_matrix = mat_to
    cv::Mat transformation_matrix = mat_from_inv * mat_to;
    return transformation_matrix;
}

cv::Mat Utils::convertToHomogeneous(cv::Mat mat) {
    cv::Mat mat_homogeneous(mat.rows, mat.cols + 1, mat.type());
    for (int i = 0; i < mat.rows; i++) {
        mat_homogeneous.at<float>(i, 0) = mat.at<float>(i, 0);
        mat_homogeneous.at<float>(i, 1) = mat.at<float>(i, 1);
        mat_homogeneous.at<float>(i, 2) = mat.at<float>(i, 2);
        mat_homogeneous.at<float>(i, 3) = 1.0;

    }
    return mat_homogeneous;
}

} // namespace et