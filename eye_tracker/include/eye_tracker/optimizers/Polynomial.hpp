#ifndef HDRMFS_EYE_TRACKER_POLYNOMIAL_HPP
#define HDRMFS_EYE_TRACKER_POLYNOMIAL_HPP

#include <vector>

namespace et
{
    class Polynomial
    {
    public:
        Polynomial(int n_variables, int polynomial_degree);

        bool fit(const std::vector<std::vector<double>> &variables, std::vector<double> *outputs);

        void setCoefficients(const std::vector<double>& coefficients);

        std::vector<double> getCoefficients() const;

        double getEstimation(const std::vector<double> &input) const;

        int getVariablesCount() const;

    private:
        std::vector<double> coefficients_{};

        std::vector<std::vector<int8_t>> monomial_sets_{};

        int polynomial_degree_{};

        int n_variables_{};

        static std::vector<std::vector<int8_t>> generateMonomials(int order, int dimension);
    };

} // namespace et

#endif //HDRMFS_EYE_TRACKER_POLYNOMIAL_HPP
