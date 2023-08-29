#include "../src/PolynomialFit.hpp"
#include "../src/Utils.hpp"

#include <filesystem>
#include <iostream>
#include <vector>

int main() {
    // Load all fitting models
    std::vector<et::PolynomialFit> cornea_centre_x{};
    std::vector<et::PolynomialFit> cornea_centre_y{};
    std::vector<et::PolynomialFit> cornea_centre_z{};
    std::vector<et::PolynomialFit> eye_centre_x{};
    std::vector<et::PolynomialFit> eye_centre_y{};
    std::vector<et::PolynomialFit> eye_centre_z{};
    std::vector<std::string> paths{};
    for (const auto &entry : std::filesystem::directory_iterator("/mnt/d/Blender/setups_left")) {
        if (entry.is_directory()) {
            if (std::filesystem::exists(entry.path().string() + "/ellipse_fitting_coeffs.csv")) {
                std::vector<std::vector<float>> coeffs =
                    et::Utils::readFloatRowsCsv(entry.path().string() + "/ellipse_fitting_coeffs.csv");
                cornea_centre_x.emplace_back(5, 3);
                cornea_centre_x.back().setCoefficients(coeffs[0]);
                eye_centre_x.emplace_back(5, 3);
                eye_centre_x.back().setCoefficients(coeffs[1]);
                cornea_centre_y.emplace_back(5, 3);
                cornea_centre_y.back().setCoefficients(coeffs[2]);
                eye_centre_y.emplace_back(5, 3);
                eye_centre_y.back().setCoefficients(coeffs[3]);
                cornea_centre_z.emplace_back(5, 3);
                cornea_centre_z.back().setCoefficients(coeffs[4]);
                eye_centre_z.emplace_back(5, 3);
                eye_centre_z.back().setCoefficients(coeffs[5]);
                paths.push_back(entry.path().string());
            }
        }
    }

    // Pick random element as a test
    int index = rand() % cornea_centre_x.size();
    // Convert to string with leading zeros
    std::string index_str = std::to_string(index);
    while (index_str.length() < 5) {
        index_str = "0" + index_str;
    }

    std::cout << "Using " << index << "th model" << std::endl;

    std::vector<std::vector<float>> image_features =
        et::Utils::readFloatRowsCsv(paths[index] + "/image_features.csv");

    std::vector<std::vector<float>> eye_features =
        et::Utils::readFloatRowsCsv(paths[index] + "/eye_features.csv");

    image_features.resize(15);
    eye_features.resize(15);

    float best_error = 99999999;
    int best_id = 0;
    float total_total_error = 0;
    int total_counter = 0;
    for (int i = 0; i < cornea_centre_x.size(); i++) {
        if (i == index)
            continue;

        float total_error = 0;
        for (int j = 0; j < 15; j++) {
            cv::Point3f cornea_centre{}, eye_centre{};
            static std::vector<float> input_data{5};

            cv::Point2f pupil_pix_position = cv::Point2f(image_features[j][1], image_features[j][2]);
            cv::RotatedRect ellipse =
                cv::RotatedRect(cv::Point2f(image_features[j][3], image_features[j][4]),
                                cv::Size2f(image_features[j][5], image_features[j][6]), image_features[j][7]);

            // Uses different sets of data for different estimated parameters.
            input_data[0] = pupil_pix_position.x;
            input_data[1] = ellipse.center.x;
            input_data[2] = ellipse.size.width;
            input_data[3] = ellipse.size.height;
            input_data[4] = ellipse.angle;
            eye_centre.x = eye_centre_x[i].getEstimation(input_data);
            cornea_centre.x = cornea_centre_x[i].getEstimation(input_data);

            input_data[0] = pupil_pix_position.y;
            input_data[1] = ellipse.center.y;
            eye_centre.y = eye_centre_y[i].getEstimation(input_data);
            cornea_centre.y = cornea_centre_y[i].getEstimation(input_data);

            input_data[0] = ellipse.center.x;
            eye_centre.z = eye_centre_z[i].getEstimation(input_data);
            cornea_centre.z = cornea_centre_z[i].getEstimation(input_data);

            total_error += std::abs(cornea_centre.x - eye_features[j][1]);
            total_error += std::abs(cornea_centre.y - eye_features[j][2]);
            total_error += std::abs(cornea_centre.z - eye_features[j][3]);
            total_error += std::abs(eye_centre.x - eye_features[j][4]);
            total_error += std::abs(eye_centre.y - eye_features[j][5]);
            total_error += std::abs(eye_centre.z - eye_features[j][6]);
        }
        std::cout << "Model " << i << " error: " << total_error << std::endl;
        if (total_error < best_error) {
            best_error = total_error;
            best_id = i;
        }
        if (total_error < 1000) {
            total_total_error += total_error;
            total_counter++;
        }
    }

    std::cout << "Mean error: " << total_total_error / total_counter << std::endl;

    std::cout << "Best model: " << best_id << "(" << best_error << ")" << std::endl;
    for (int j = 0; j < 15; j++) {
        cv::Point3f cornea_centre{}, eye_centre{};
        static std::vector<float> input_data{5};

        cv::Point2f pupil_pix_position = cv::Point2f(image_features[j][1], image_features[j][2]);
        cv::RotatedRect ellipse =
            cv::RotatedRect(cv::Point2f(image_features[j][3], image_features[j][4]),
                            cv::Size2f(image_features[j][5], image_features[j][6]), image_features[j][7]);

        // Uses different sets of data for different estimated parameters.
        input_data[0] = pupil_pix_position.x;
        input_data[1] = ellipse.center.x;
        input_data[2] = ellipse.size.width;
        input_data[3] = ellipse.size.height;
        input_data[4] = ellipse.angle;
        eye_centre.x = eye_centre_x[best_id].getEstimation(input_data);
        cornea_centre.x = cornea_centre_x[best_id].getEstimation(input_data);

        input_data[0] = pupil_pix_position.y;
        input_data[1] = ellipse.center.y;
        eye_centre.y = eye_centre_y[best_id].getEstimation(input_data);
        cornea_centre.y = cornea_centre_y[best_id].getEstimation(input_data);

        input_data[0] = ellipse.center.x;
        eye_centre.z = eye_centre_z[best_id].getEstimation(input_data);
        cornea_centre.z = cornea_centre_z[best_id].getEstimation(input_data);

        std::cout << "Estimated cornea centre: " << cornea_centre.x << " " << cornea_centre.y << " " << cornea_centre.z
                  << std::endl;
        std::cout << "Estimated eye centre: " << eye_centre.x << " " << eye_centre.y << " " << eye_centre.z
                  << std::endl;

        std::cout << "Expected cornea centre: " << eye_features[j][1] << " " << eye_features[j][2] << " "
                  << eye_features[j][3] << std::endl;
        std::cout << "Expected eye centre: " << eye_features[j][4] << " " << eye_features[j][5] << " "
                  << eye_features[j][6] << std::endl;
    }
    return 0;
}