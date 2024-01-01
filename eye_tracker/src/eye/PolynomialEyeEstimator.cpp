#include <random>
#include "eye_tracker/eye/PolynomialEyeEstimator.hpp"
#include "eye_tracker/Utils.hpp"

namespace et
{
    PolynomialEyeEstimator::PolynomialEyeEstimator(int camera_id) : EyeEstimator(camera_id)
    {
        eye_centre_pos_x_fit = std::make_shared<PolynomialFit>(6, 3);
        eye_centre_pos_y_fit = std::make_shared<PolynomialFit>(6, 3);
        eye_centre_pos_z_fit = std::make_shared<PolynomialFit>(6, 4);
        theta_fit = std::make_shared<PolynomialFit>(6, 4);
        phi_fit = std::make_shared<PolynomialFit>(6, 4);
        pupil_x_fit = std::make_shared<PolynomialFit>(5, 3);
        pupil_y_fit = std::make_shared<PolynomialFit>(5, 3);
        ellipse_x_fit = std::make_shared<PolynomialFit>(5, 3);
        ellipse_y_fit = std::make_shared<PolynomialFit>(5, 3);
        ellipse_width_fit = std::make_shared<PolynomialFit>(5, 3);
        ellipse_height_fit = std::make_shared<PolynomialFit>(5, 3);
        ellipse_angle_fit = std::make_shared<PolynomialFit>(5, 3);
        setModel("default");
    }

    PolynomialEyeEstimator::PolynomialEyeEstimator(int camera_id, std::string model_id) : EyeEstimator(camera_id)
    {
        eye_centre_pos_x_fit = std::make_shared<PolynomialFit>(6, 3);
        eye_centre_pos_y_fit = std::make_shared<PolynomialFit>(6, 3);
        eye_centre_pos_z_fit = std::make_shared<PolynomialFit>(6, 4);
        theta_fit = std::make_shared<PolynomialFit>(6, 4);
        phi_fit = std::make_shared<PolynomialFit>(6, 4);
        pupil_x_fit = std::make_shared<PolynomialFit>(5, 3);
        pupil_y_fit = std::make_shared<PolynomialFit>(5, 3);
        ellipse_x_fit = std::make_shared<PolynomialFit>(5, 3);
        ellipse_y_fit = std::make_shared<PolynomialFit>(5, 3);
        ellipse_width_fit = std::make_shared<PolynomialFit>(5, 3);
        ellipse_height_fit = std::make_shared<PolynomialFit>(5, 3);
        ellipse_angle_fit = std::make_shared<PolynomialFit>(5, 3);
        setModel(model_id);
    }

    void PolynomialEyeEstimator::setModel(std::string user_id)
    {
        std::clog << "Loading model " << user_id << std::endl;
        Coefficients *coefficients;
        if (user_id == "") {
            coefficients = &Settings::parameters.user_polynomial_params[camera_id_]->coefficients;
        } else {
            coefficients = &Settings::parameters.polynomial_params[camera_id_][user_id].coefficients;
        }

        eye_centre_pos_x_fit->setCoefficients(coefficients->eye_centre_pos_x);
        eye_centre_pos_y_fit->setCoefficients(coefficients->eye_centre_pos_y);
        eye_centre_pos_z_fit->setCoefficients(coefficients->eye_centre_pos_z);
        theta_fit->setCoefficients(coefficients->theta);
        phi_fit->setCoefficients(coefficients->phi);

        pupil_x_fit->setCoefficients(coefficients->pupil_x);
        pupil_y_fit->setCoefficients(coefficients->pupil_y);
        ellipse_x_fit->setCoefficients(coefficients->ellipse_x);
        ellipse_y_fit->setCoefficients(coefficients->ellipse_y);
        ellipse_width_fit->setCoefficients(coefficients->ellipse_width);
        ellipse_height_fit->setCoefficients(coefficients->ellipse_height);
        ellipse_angle_fit->setCoefficients(coefficients->ellipse_angle);

        features_params_ = Settings::parameters.user_params[camera_id_];
    }

    bool PolynomialEyeEstimator::fitModel(std::vector<cv::Point2d> &pupils, std::vector<cv::RotatedRect> &ellipses,
                                          std::vector<cv::Point3d> &eye_centres, std::vector<cv::Vec2d> &angles)
    {
        bool result = true;
        std::vector<double> pupil_x{};
        std::vector<double> pupil_y{};
        std::vector<double> ellipse_x{};
        std::vector<double> ellipse_y{};
        std::vector<double> ellipse_width{};
        std::vector<double> ellipse_height{};
        std::vector<double> ellipse_angle{};
        std::vector<double> eye_centre_pos_x{};
        std::vector<double> eye_centre_pos_y{};
        std::vector<double> eye_centre_pos_z{};
        std::vector<double> thetas{};
        std::vector<double> phis{};
        for (int i = 0; i < pupils.size(); i++) {
            pupil_x.push_back(pupils[i].x);
            pupil_y.push_back(pupils[i].y);
            ellipse_x.push_back(ellipses[i].center.x);
            ellipse_y.push_back(ellipses[i].center.y);
            ellipse_width.push_back(ellipses[i].size.width);
            ellipse_height.push_back(ellipses[i].size.height);
            ellipse_angle.push_back(ellipses[i].angle);
            eye_centre_pos_x.push_back(eye_centres[i].x);
            eye_centre_pos_y.push_back(eye_centres[i].y);
            eye_centre_pos_z.push_back(eye_centres[i].z);
            thetas.push_back(angles[i][0]);
            phis.push_back(angles[i][1]);
        }


        std::vector<std::vector<double>> inputs[] = {{eye_centre_pos_x, eye_centre_pos_y, eye_centre_pos_z, thetas,    phis},
                                                     {eye_centre_pos_x, eye_centre_pos_y, eye_centre_pos_z, thetas,    phis},
                                                     {eye_centre_pos_x, eye_centre_pos_y, eye_centre_pos_z, thetas,    phis},
                                                     {eye_centre_pos_x, eye_centre_pos_y, eye_centre_pos_z, thetas,    phis},
                                                     {eye_centre_pos_x, eye_centre_pos_y, eye_centre_pos_z, thetas,    phis},
                                                     {eye_centre_pos_x, eye_centre_pos_y, eye_centre_pos_z, thetas,    phis},
                                                     {eye_centre_pos_x, eye_centre_pos_y, eye_centre_pos_z, thetas,    phis},
                                                     {pupil_x,          pupil_y,          ellipse_x,        ellipse_y, ellipse_width, ellipse_height/*, ellipse_angle*/},
                                                     {pupil_x,          pupil_y,          ellipse_x,        ellipse_y, ellipse_width, ellipse_height/*, ellipse_angle*/},
                                                     {pupil_x,          pupil_y,          ellipse_x,        ellipse_y, ellipse_width, ellipse_height/*, ellipse_angle*/},
                                                     {pupil_x,          pupil_y,          ellipse_x,        ellipse_y, ellipse_width, ellipse_height/*, ellipse_angle*/},
                                                     {pupil_x,          pupil_y,          ellipse_x,        ellipse_y, ellipse_width, ellipse_height/*, ellipse_angle*/},};

        std::vector<double> *outputs[] = {&pupil_x, &pupil_y, &ellipse_x, &ellipse_y, &ellipse_width, &ellipse_height,
                                          &ellipse_angle, &eye_centre_pos_x, &eye_centre_pos_y, &eye_centre_pos_z,
                                          &thetas, &phis};

        std::shared_ptr<PolynomialFit> fitters[] = {pupil_x_fit, pupil_y_fit, ellipse_x_fit, ellipse_y_fit,
                                                    ellipse_width_fit, ellipse_height_fit, ellipse_angle_fit,
                                                    eye_centre_pos_x_fit, eye_centre_pos_y_fit, eye_centre_pos_z_fit,
                                                    theta_fit, phi_fit};

        std::string names[] = {"pupil_x", "pupil_y", "ellipse_x", "ellipse_y", "ellipse_width", "ellipse_height",
                               "ellipse_angle", "eye_centre_pos_x", "eye_centre_pos_y", "eye_centre_pos_z", "theta",
                               "phi"};

        int setup_count = sizeof inputs / sizeof inputs[0];
        int cross_folds = 5;
        int data_points_num = pupils.size();


        for (int setup = 0; setup < setup_count; setup++) {
            bool success = true;
            std::clog << "Fitting " << names[setup] << std::endl;
            std::vector<int> indices(data_points_num);
            Utils::getCrossValidationIndices(indices, data_points_num, cross_folds);
            for (int i = 0; i <= cross_folds; i++) {
                std::vector<std::vector<double>> train_input_data{};
                std::vector<std::vector<double>> test_input_data{};
                std::vector<double> train_output_data{};
                std::vector<double> test_output_data{};
                for (int j = 0; j < indices.size(); j++) {
                    if (indices[j] == i || i == cross_folds) {
                        std::vector<double> row{};
                        for (int k = 0; k < fitters[setup]->getNVariables(); k++) {
                            row.push_back(inputs[setup][k][j]);
                        }
                        test_input_data.push_back(row);
                        test_output_data.push_back((*outputs[setup])[j]);
                    }
                    if (indices[j] != i || i == cross_folds) {
                        std::vector<double> row{};
                        for (int k = 0; k < fitters[setup]->getNVariables(); k++) {
                            row.push_back(inputs[setup][k][j]);
                        }
                        train_input_data.push_back(row);
                        train_output_data.push_back((*outputs[setup])[j]);
                    }
                }
                success = fitters[setup]->fit(train_input_data, &train_output_data);
                if (i == cross_folds) {
                    result &= success;
                }
                if (!success) {
                    std::clog << "[" << (i == cross_folds ? "full" : std::to_string(i)) << "] Fitting failed."
                              << std::endl;
                    continue;
                }

                std::vector<double> error{};
                for (int j = 0; j < test_input_data.size(); j++) {
                    std::vector<double> input{};
                    double estimation = fitters[setup]->getEstimation(test_input_data[j]);
                    error.push_back(std::abs(estimation - test_output_data[j]));
                }
                double mean_error = std::accumulate(error.begin(), error.end(), 0.0) / error.size();
                double std_error = 0.0;
                for (int j = 0; j < error.size(); j++) {
                    std_error += (error[j] - mean_error) * (error[j] - mean_error);
                }
                std_error = std::sqrt(std_error / error.size());

                std::clog << "[" << (i == cross_folds ? "full" : std::to_string(i)) << "] Fitting error: " << mean_error
                          << " ± " << std_error << std::endl;
            }
        }

        if (!result) {
            return false;
        }

        return true;
    }

    void PolynomialEyeEstimator::invertEye(cv::Point3d &nodal_point, cv::Point3d &eye_centre, EyeInfo &eye_info)
    {
        std::vector<double> input_data(6);
        input_data[0] = eye_centre.x;
        input_data[1] = eye_centre.y;
        input_data[2] = eye_centre.z;
        input_data[3] = nodal_point.x;
        input_data[4] = nodal_point.y;
        input_data[5] = nodal_point.z;
        eye_info.pupil.x = pupil_x_fit->getEstimation(input_data);
        eye_info.pupil.y = pupil_y_fit->getEstimation(input_data);
        eye_info.ellipse.center.x = ellipse_x_fit->getEstimation(input_data);
        eye_info.ellipse.center.y = ellipse_y_fit->getEstimation(input_data);
        eye_info.ellipse.size.width = ellipse_width_fit->getEstimation(input_data);
        eye_info.ellipse.size.height = ellipse_height_fit->getEstimation(input_data);
        eye_info.ellipse.angle = ellipse_angle_fit->getEstimation(input_data);
    }

    void PolynomialEyeEstimator::coeffsToMat(cv::Mat &mat)
    {
        std::vector<double> coeffs{};
        std::vector<double> eye_centre_pos_x_coeffs = eye_centre_pos_x_fit->getCoefficients();
        std::vector<double> eye_centre_pos_y_coeffs = eye_centre_pos_y_fit->getCoefficients();
        std::vector<double> eye_centre_pos_z_coeffs = eye_centre_pos_z_fit->getCoefficients();
        std::vector<double> theta_coeffs = theta_fit->getCoefficients();
        std::vector<double> phi_coeffs = phi_fit->getCoefficients();

        coeffs.insert(coeffs.end(), eye_centre_pos_x_coeffs.begin(), eye_centre_pos_x_coeffs.end());
        coeffs.insert(coeffs.end(), eye_centre_pos_y_coeffs.begin(), eye_centre_pos_y_coeffs.end());
        coeffs.insert(coeffs.end(), eye_centre_pos_z_coeffs.begin(), eye_centre_pos_z_coeffs.end());
        coeffs.insert(coeffs.end(), theta_coeffs.begin(), theta_coeffs.end());
        coeffs.insert(coeffs.end(), phi_coeffs.begin(), phi_coeffs.end());

        mat = cv::Mat(coeffs.size(), 1, CV_64F);
        for (int i = 0; i < coeffs.size(); i++) {
            mat.at<double>(i) = coeffs[i];
        }
    }

    Coefficients PolynomialEyeEstimator::getCoefficients() const
    {
        return Coefficients{eye_centre_pos_x_fit->getCoefficients(), eye_centre_pos_y_fit->getCoefficients(),
                            eye_centre_pos_z_fit->getCoefficients(), theta_fit->getCoefficients(),
                            phi_fit->getCoefficients(), pupil_x_fit->getCoefficients(), pupil_y_fit->getCoefficients(),
                            ellipse_x_fit->getCoefficients(), ellipse_y_fit->getCoefficients(),
                            ellipse_width_fit->getCoefficients(), ellipse_height_fit->getCoefficients(),
                            ellipse_angle_fit->getCoefficients()};
    }

    bool PolynomialEyeEstimator::detectEye(EyeInfo &eye_info, cv::Point3d &eye_centre, cv::Vec2d &angles)
    {
        std::vector<double> input_data(6);

        // Uses different sets of data for different estimated parameters.
        input_data[0] = eye_info.pupil.x;
        input_data[1] = eye_info.pupil.y;
        input_data[2] = eye_info.ellipse.center.x;
        input_data[3] = eye_info.ellipse.center.y;
        input_data[4] = eye_info.ellipse.size.width;
        input_data[5] = eye_info.ellipse.size.height;
        eye_centre.x = eye_centre_pos_x_fit->getEstimation(input_data);
        eye_centre.y = eye_centre_pos_y_fit->getEstimation(input_data);
        eye_centre.z = eye_centre_pos_z_fit->getEstimation(input_data);
        angles[0] = theta_fit->getEstimation(input_data);
        angles[1] = phi_fit->getEstimation(input_data);

        eye_centre -= features_params_->eye_centre_offset;
        angles -= features_params_->angles_offset;

        return true;
    }
} // et