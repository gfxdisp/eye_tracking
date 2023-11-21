#include "eye_tracker/optimizers/AggregatedPolynomialOptimizer.hpp"
#include "eye_tracker/Utils.hpp"

namespace et
{
    int AggregatedPolynomialOptimizer::getDims() const
    {
        return polynomial_eye_estimator_.eye_centre_pos_x_fit->getCoefficients().size() +
               polynomial_eye_estimator_.nodal_point_x_fit->getCoefficients().size() +
               polynomial_eye_estimator_.eye_centre_pos_y_fit->getCoefficients().size() +
               polynomial_eye_estimator_.nodal_point_y_fit->getCoefficients().size() +
               polynomial_eye_estimator_.eye_centre_pos_z_fit->getCoefficients().size() +
               polynomial_eye_estimator_.nodal_point_z_fit->getCoefficients().size();
    }

    double AggregatedPolynomialOptimizer::calc(const double *x) const
    {
        static std::shared_ptr<PolynomialEyeEstimator> polynomial_eye_estimator = std::make_shared<PolynomialEyeEstimator>(0);
        static std::shared_ptr<PolynomialFit> fitters[] = {polynomial_eye_estimator->eye_centre_pos_x_fit,
                                    polynomial_eye_estimator->nodal_point_x_fit,
                                    polynomial_eye_estimator->eye_centre_pos_y_fit,
                                    polynomial_eye_estimator->nodal_point_y_fit,
                                    polynomial_eye_estimator->eye_centre_pos_z_fit,
                                    polynomial_eye_estimator->nodal_point_z_fit};

        int counter = 0;
        for (int i = 0; i < 6; i++)
        {
            int coeff_num = fitters[i]->getCoefficients().size();
            std::vector<double> coefficients{};
            for (int j = 0; j < coeff_num; j++)
            {
                coefficients.push_back(x[counter++]);
            }
            fitters[i]->setCoefficients(coefficients);
        }

        double total_error = 0;
        for (int i = 0; i < pupils.size(); i++)
        {
            EyeInfo eye_info = {.pupil = pupils[i], .ellipse = ellipses[i]};

            cv::Point3d nodal_point{}, eye_centre{}, visual_axis{};

            polynomial_eye_estimator->detectEye(eye_info, nodal_point, eye_centre, visual_axis);

            double angle = Utils::getAngleBetweenVectors(visual_axis, visual_axes[i]) * 180.0 / CV_PI;
            double distance = cv::norm(eye_centre - eye_centres[i]);
            total_error += std::abs(angle) / ACCEPTABLE_ANGLE_ERROR + distance / ACCEPTABLE_DISTANCE_ERROR;
        }

        std::clog << "Current error: " << total_error / pupils.size() << std::endl;
        return total_error / (pupils.size() * 2.0);
    }
} // et