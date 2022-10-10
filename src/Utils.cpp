#include "Utils.hpp"

#include <fstream>
#include <sstream>

namespace et {
std::vector<std::vector<float>> Utils::readFloatColumnsCsv(const std::string& filename) {
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

std::vector<std::vector<float>> Utils::readFloatRowsCsv(const std::string& filename) {
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

void Utils::writeFloatCsv(std::vector<std::vector<float>> &data, const std::string& filename) {
    std::ofstream file{filename};
    for (auto & row : data) {
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
} // namespace et