#ifndef HDRMFS_EYE_TRACKER_POLYNOMIALEYEESTIMATOR_HPP
#define HDRMFS_EYE_TRACKER_POLYNOMIALEYEESTIMATOR_HPP

#include "EyeEstimator.hpp"
#include "eye_tracker/optimizers/PolynomialFit.hpp"

namespace et {

    class PolynomialEyeEstimator : public EyeEstimator {
    public:
        PolynomialEyeEstimator(int camera_id);

        bool fitModel(std::vector<cv::Point2d>& pupils, std::vector<cv::RotatedRect>& ellipses,
                      std::vector<cv::Point3d>& eye_centres, std::vector<cv::Vec2d>& angles);

        bool detectEye(EyeInfo& eye_info, cv::Point3d& eye_centre, cv::Point3d& nodal_point, cv::Vec2d& angles) override;

        Coefficients getCoefficients() const;

        std::shared_ptr<PolynomialFit> eye_centre_pos_x_fit{};
        std::shared_ptr<PolynomialFit> eye_centre_pos_y_fit{};
        std::shared_ptr<PolynomialFit> eye_centre_pos_z_fit{};
        std::shared_ptr<PolynomialFit> theta_fit{};
        std::shared_ptr<PolynomialFit> phi_fit{};
    };

} // et

#endif //HDRMFS_EYE_TRACKER_POLYNOMIALEYEESTIMATOR_HPP
