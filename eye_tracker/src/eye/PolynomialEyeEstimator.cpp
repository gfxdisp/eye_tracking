#include <random>
#include "eye_tracker/eye/PolynomialEyeEstimator.hpp"
#include "eye_tracker/Utils.hpp"

namespace et
{
    PolynomialEyeEstimator::PolynomialEyeEstimator(int camera_id) : EyeEstimator(camera_id)
    {
        eye_centre_pos_x_fit = std::make_shared<PolynomialFit>(8, 4);
        eye_centre_pos_y_fit = std::make_shared<PolynomialFit>(8, 4);
        eye_centre_pos_z_fit = std::make_shared<PolynomialFit>(8, 4);
        theta_fit = std::make_shared<PolynomialFit>(8, 4);
        phi_fit = std::make_shared<PolynomialFit>(8, 4);

        auto coefficients = &Settings::parameters.polynomial_params[camera_id_].coefficients;
        if (eye_centre_pos_x_fit->getCoefficients().size() == coefficients->eye_centre_pos_x.size()) {
            eye_centre_pos_x_fit->setCoefficients(coefficients->eye_centre_pos_x);
            eye_centre_pos_y_fit->setCoefficients(coefficients->eye_centre_pos_y);
            eye_centre_pos_z_fit->setCoefficients(coefficients->eye_centre_pos_z);
            theta_fit->setCoefficients(coefficients->theta);
            phi_fit->setCoefficients(coefficients->phi);
        } else {
            std::cerr << "[" << camera_id << "] PolynomialEyeEstimator: coefficients size mismatch, please run polynomial_estimator first." << std::endl;
        }

        features_params_ = Settings::parameters.user_params[camera_id_];
    }

    bool PolynomialEyeEstimator::fitModel(std::vector<cv::Point2d>& pupils, std::vector<cv::RotatedRect>& ellipses,
                                          std::vector<cv::Point3d>& eye_centres, std::vector<cv::Vec2d>& angles)
    {
        bool result = true;
        std::vector<double> pupil_x{};
        std::vector<double> pupil_y{};
        std::vector<double> ellipse_x{};
        std::vector<double> ellipse_y{};
        std::vector<double> ellipse_width{};
        std::vector<double> ellipse_height{};
        std::vector<double> inv_ellipse_width{};
        std::vector<double> inv_ellipse_height{};
        std::vector<double> ellipse_angle{};
        std::vector<double> eye_centre_pos_x{};
        std::vector<double> eye_centre_pos_y{};
        std::vector<double> eye_centre_pos_z{};
        std::vector<double> thetas{};
        std::vector<double> phis{};
        for (int i = 0; i < pupils.size(); i++)
        {
            pupil_x.push_back(pupils[i].x);
            pupil_y.push_back(pupils[i].y);
            ellipse_x.push_back(ellipses[i].center.x);
            ellipse_y.push_back(ellipses[i].center.y);
            ellipse_width.push_back(ellipses[i].size.width);
            ellipse_height.push_back(ellipses[i].size.height);
            inv_ellipse_width.push_back(1.0 / ellipses[i].size.width);
            inv_ellipse_height.push_back(1.0 / ellipses[i].size.height);
            ellipse_angle.push_back(ellipses[i].angle);
            eye_centre_pos_x.push_back(eye_centres[i].x);
            eye_centre_pos_y.push_back(eye_centres[i].y);
            eye_centre_pos_z.push_back(eye_centres[i].z);
            thetas.push_back(angles[i][0]);
            phis.push_back(angles[i][1]);
        }


        std::vector<std::vector<double>> inputs[] = {
            {pupil_x, pupil_y, ellipse_x, ellipse_y, ellipse_width, ellipse_height, inv_ellipse_width, inv_ellipse_height/*, ellipse_angle*/},
            {pupil_x, pupil_y, ellipse_x, ellipse_y, ellipse_width, ellipse_height, inv_ellipse_width, inv_ellipse_height/*, ellipse_angle*/},
            {pupil_x, pupil_y, ellipse_x, ellipse_y, ellipse_width, ellipse_height, inv_ellipse_width, inv_ellipse_height/*, ellipse_angle*/},
            {pupil_x, pupil_y, ellipse_x, ellipse_y, ellipse_width, ellipse_height, inv_ellipse_width, inv_ellipse_height/*, ellipse_angle*/},
            {pupil_x, pupil_y, ellipse_x, ellipse_y, ellipse_width, ellipse_height, inv_ellipse_width, inv_ellipse_height/*, ellipse_angle*/},
        };

        std::vector<double>* outputs[] = {&eye_centre_pos_x, &eye_centre_pos_y, &eye_centre_pos_z, &thetas, &phis};

        std::shared_ptr<PolynomialFit> fitters[] = {
            eye_centre_pos_x_fit, eye_centre_pos_y_fit, eye_centre_pos_z_fit, theta_fit, phi_fit
        };

        std::string names[] = {"eye_centre_pos_x", "eye_centre_pos_y", "eye_centre_pos_z", "theta", "phi"};

        int setup_count = std::size(inputs);
        int data_points_num = pupils.size();

        for (int setup = 0; setup < setup_count; setup++)
        {
            int cross_folds = 5;
            bool success = true;
            std::clog << "Fitting " << names[setup] << std::endl;
            std::vector<int> indices(data_points_num);
            Utils::getCrossValidationIndices(indices, data_points_num, cross_folds);
            std::vector<double> error{};
            for (int i = 0; i <= cross_folds; i++)
            {
//                i = cross_folds;
                std::vector<std::vector<double>> train_input_data{};
                std::vector<std::vector<double>> test_input_data{};
                std::vector<double> train_output_data{};
                std::vector<double> test_output_data{};
                for (int j = 0; j < indices.size(); j++)
                {
                    if (indices[j] == i || i == cross_folds)
                    {
                        std::vector<double> row{};
                        for (int k = 0; k < fitters[setup]->getNVariables(); k++)
                        {
                            row.push_back(inputs[setup][k][j]);
                        }
                        test_input_data.push_back(row);
                        test_output_data.push_back((*outputs[setup])[j]);
                    }
                    if (indices[j] != i || i == cross_folds)
                    {
                        std::vector<double> row{};
                        for (int k = 0; k < fitters[setup]->getNVariables(); k++)
                        {
                            row.push_back(inputs[setup][k][j]);
                        }
                        train_input_data.push_back(row);
                        train_output_data.push_back((*outputs[setup])[j]);
                    }
                }
                success = fitters[setup]->fit(train_input_data, &train_output_data);
                if (!success)
                {
                    std::clog << "[" << (i == cross_folds ? "full" : "cross-val") << "] Fitting failed." << std::endl;
                    if (i == cross_folds)
                    {
                        result = false;
                        break;
                    }
                    i = cross_folds - 1;
                    continue;
                }

                for (int j = 0; j < test_input_data.size(); j++)
                {
                    double estimation = fitters[setup]->getEstimation(test_input_data[j]);
                    error.push_back(std::abs(estimation - test_output_data[j]));
                }

                if (i == cross_folds || i == cross_folds - 1)
                {
                    double mean_error = std::accumulate(error.begin(), error.end(), 0.0) / error.size();
                    double std_error = 0.0;
                    for (int j = 0; j < error.size(); j++)
                    {
                        std_error += (error[j] - mean_error) * (error[j] - mean_error);
                    }
                    std_error = std::sqrt(std_error / error.size());

                    std::clog << "[" << (i == cross_folds ? "full" : "cross-val") << "] Fitting error: " << mean_error
                        << " ± " << std_error << std::endl;
                    error.clear();
                }
            }
        }

        if (!result)
        {
            return false;
        }

        return true;
    }

    Coefficients PolynomialEyeEstimator::getCoefficients() const
    {
        return Coefficients{
            eye_centre_pos_x_fit->getCoefficients(), eye_centre_pos_y_fit->getCoefficients(),
            eye_centre_pos_z_fit->getCoefficients(), theta_fit->getCoefficients(), phi_fit->getCoefficients()
        };
    }

    bool PolynomialEyeEstimator::detectEye(EyeInfo& eye_info, cv::Point3d& eye_centre, cv::Vec2d& angles)
    {
        std::vector<double> input_data(8);

        input_data[0] = eye_info.pupil.x;
        input_data[1] = eye_info.pupil.y;
        input_data[2] = eye_info.ellipse.center.x;
        input_data[3] = eye_info.ellipse.center.y;
        input_data[4] = eye_info.ellipse.size.width;
        input_data[5] = eye_info.ellipse.size.height;
        input_data[6] = 1.0 / eye_info.ellipse.size.width;
        input_data[7] = 1.0 / eye_info.ellipse.size.height;
        eye_centre.x = eye_centre_pos_x_fit->getEstimation(input_data);
        eye_centre.y = eye_centre_pos_y_fit->getEstimation(input_data);
        eye_centre.z = eye_centre_pos_z_fit->getEstimation(input_data);
        angles[0] = theta_fit->getEstimation(input_data);
        angles[1] = phi_fit->getEstimation(input_data);
//
//        eye_centre = cv::Point3d(128.428925, 151.086770, 870.013819);
//        eye_centre = cv::Point3d(88,  138,   1000);
//        eye_centre = cv::Point3d(133.804128, 150.495856, 847.675868);

        return true;
    }
} // et
