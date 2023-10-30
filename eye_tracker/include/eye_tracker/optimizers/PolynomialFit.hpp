#ifndef HDRMFS_EYE_TRACKER_POLYNOMIAL_FIT_HPP
#define HDRMFS_EYE_TRACKER_POLYNOMIAL_FIT_HPP

#include <string>
#include <vector>

namespace et
{
/**
 * Finds the fitting between a specified input and output.
 */
    class PolynomialFit
    {
    public:
        /**
         * Generates all possible monomials of the polynomial to which the data
         * will be fit.
         * @param n_variables Total number of polynomial variables.
         * @param polynomial_degree Polynomial degree.
         */
        PolynomialFit(int n_variables, int polynomial_degree);

        /**
         * Fits the input data to the output.
         * @param variables Vector of variables where each variable is specified
         * as a vector of data points.
         * @param outputs Vector of output data points.
         * @return True if the fitting was successful.
         */
        bool fit(std::vector<std::vector<float>> &variables, std::vector<float> *outputs);

        /**
         * Sets polynomial coefficients directly, without fitting.
         * @param coefficients Vector of coefficients.
         */
        void setCoefficients(std::vector<float> coefficients);

        /**
         * Retrieves a list of coefficients calculated using fit() or
         * setCoefficients() methods.
         * @return Vector of coefficients.
         */
        std::vector<float> getCoefficients() const;

        /**
         * Estimates a polynomial value based on the input.
         * @param input Set of input variables.
         * @return Estimated polynomial value.
         */
        float getEstimation(const std::vector<float> &input);

        int getNVariables() const;

    private:
        // List of coefficients to polynomial monomials in the order specified
        // in the monomial_sets_ variable.
        std::vector<float> coefficients_{};
        // List of all possible monomials for the specific polynomial. Every vector
        // element contains a vector specifying an exponent of every input variable.
        std::vector<std::vector<int8_t>> monomial_sets_{};
        // Polynomial degree.
        int polynomial_degree_{};
        // Total number of polynomial variables
        int n_variables_{};

        /**
         * Creates a list of all possible monomials for a specific polynomial.
         * @param order Max exponent of the monomial.
         * @param dimension Number of variables per monomial.
         * @return List of all monomials.
         */
        std::vector<std::vector<int8_t>> generateMonomials(int order, int dimension);
    };

} // namespace et

#endif //HDRMFS_EYE_TRACKER_POLYNOMIAL_FIT_HPP
