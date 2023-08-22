#include "PolynomialFit.hpp"

#include <opencv2/opencv.hpp>

namespace et {
bool PolynomialFit::fit(std::vector<std::vector<float> *> &variables, std::vector<float> *outputs) {

    auto max_abs_func = [](const auto &a, const auto &b) {
        return std::abs(a) < std::abs(b);
    };

    // Finds the maximum data point for each variable.
    std::vector<float> maximum{};
    for (const auto &variable : variables) {
        maximum.push_back(std::abs(*std::max_element(variable->begin(), variable->end(), max_abs_func)));
    }

    auto n_data_points = (int) variables[0]->size();
    auto n_coeffs = (int) monomial_sets_.size();
    cv::Mat A{cv::Size(n_coeffs, n_data_points), CV_32FC1};
    cv::Mat scaled_output{cv::Size(1, n_data_points), CV_32FC1};

    // Computes all exponents scaled by max values.
    for (int i = 0; i < n_coeffs; i++) {
        for (int j = 0; j < n_data_points; j++) {
            double val{1};
            for (int k = 0; k < n_variables_; k++) {
                val *= std::pow((*variables[k])[j] / maximum[k], monomial_sets_[i][k]);
            }
            A.at<float>(j, i) = (float) val;
        }
    }

    cv::Mat w, u, vt;
    cv::SVD::compute(A, w, u, vt, cv::SVD::FULL_UV);
    cv::Mat qqs{cv::Size(n_coeffs, n_data_points), CV_32FC1};
    qqs.setTo(0);
    for (int i = 0; i < n_coeffs; i++) {
        if (w.at<float>(i, 0) > 1e-5) {
            qqs.at<float>(i, i) = 1.0f / w.at<float>(i, 0);
        }
    }

    if (outputs->empty()) {
        for (int i = 0; i < n_coeffs; i++) {
            coefficients_.push_back(0);
        }
        return false;
    } else {
        float maximum_output = std::abs(*std::max_element(outputs->begin(), outputs->end(), max_abs_func));

        for (int i = 0; i < n_data_points; i++) {
            scaled_output.at<float>(i, 0) = (*outputs)[i] / maximum_output;
        }

        cv::Mat scaled_coeffs = vt.t() * qqs.t() * u.t() * scaled_output;
        // Rescales the coefficients back.
        for (int i = 0; i < n_coeffs; i++) {
            double val{1};
            for (int j = 0; j < n_variables_; j++) {
                val *= std::pow(1.0 / maximum[j], monomial_sets_[i][j]);
            }
            coefficients_[i] = (float) (val * scaled_coeffs.at<float>(i, 0) * maximum_output);
        }
        return true;
    }
}

float PolynomialFit::getEstimation(const std::vector<float> &input) {
    double total{0};
    for (int i = 0; i < monomial_sets_.size(); i++) {
        double val{coefficients_[i]};
        for (int j = 0; j < n_variables_; j++) {
            val *= std::pow(input[j], monomial_sets_[i][j]);
        }
        total += val;
    }
    return (float) total;
}

std::vector<std::vector<int>> PolynomialFit::generateMonomials(int order, int dimension) {
    std::vector<std::vector<int>> output{};
    if (dimension == 1) {
        for (int i = 0; i <= order; i++) {
            output.push_back({i});
        }
        return output;
    }
    for (int i = 0; i <= order; i++) {
        auto all_lower_monomials = generateMonomials(order - i, dimension - 1);
        for (auto &monomial_set : all_lower_monomials) {
            monomial_set.insert(monomial_set.begin(), i);
        }
        output.insert(output.end(), all_lower_monomials.begin(), all_lower_monomials.end());
    }
    return output;
}

void PolynomialFit::setCoefficients(std::vector<float> &coefficients) {
    coefficients_ = coefficients;
}

std::vector<float> PolynomialFit::getCoefficients() {
    return coefficients_;
}

PolynomialFit::PolynomialFit(int n_variables, int polynomial_degree) {
    n_variables_ = n_variables;
    polynomial_degree_ = polynomial_degree;
    monomial_sets_ = generateMonomials(polynomial_degree_, n_variables_);
    coefficients_ = std::vector<float>(monomial_sets_.size(), 0);
}
} // namespace et