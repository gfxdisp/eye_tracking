#ifndef HDRMFS_EYE_TRACKER_POLYNOMIALFIT_HPP
#define HDRMFS_EYE_TRACKER_POLYNOMIALFIT_HPP

#include <string>
#include <vector>

namespace et {

class PolynomialFit {
public:
    PolynomialFit(int n_variables, int polynomial_degree);
    void fit(std::vector<std::vector<float>*> &variables, std::vector<float> *outputs);
    void setCoefficients(std::vector<float>& coefficients);
    std::vector<float> getCoefficients();
    float getEstimation(const std::vector<float>& input);
private:
    std::vector<float> coefficients_{};
    std::vector<std::vector<int>> monomial_sets_{};
    int polynomial_degree_{};
    int n_variables_{};
    std::vector<std::vector<int>> generateMonomials(int order, int dimension);
};

} // namespace et

#endif //HDRMFS_EYE_TRACKER_POLYNOMIALFIT_HPP
