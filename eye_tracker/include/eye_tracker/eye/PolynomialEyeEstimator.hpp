#ifndef HDRMFS_EYE_TRACKER_POLYNOMIALEYEESTIMATOR_HPP
#define HDRMFS_EYE_TRACKER_POLYNOMIALEYEESTIMATOR_HPP

#include "EyeEstimator.hpp"
#include "eye_tracker/optimizers/PolynomialFit.hpp"

namespace et
{

    class PolynomialEyeEstimator : public EyeEstimator
    {
    public:
        PolynomialEyeEstimator(int camera_id);

        PolynomialEyeEstimator(int camera_id, int model_num);

        void setModel(int model_num);

        bool fitModel(std::vector<cv::Point2d> &pupils, std::vector<cv::RotatedRect> &ellipses,
                      std::vector<cv::Point3d> &eye_centres, std::vector<cv::Point3d> &nodal_points, std::vector<cv::Vec3d> &visual_axes);

        bool detectEye(EyeInfo &eye_info, cv::Point3d &nodal_point, cv::Point3d &eye_centre,
                       cv::Point3d &visual_axis) override;

        void invertEye(cv::Point3d &nodal_point, cv::Point3d &eye_centre, EyeInfo &eye_info);

        Coefficients getCoefficients() const;

        std::shared_ptr<PolynomialFit> eye_centre_pos_x_fit{};
        std::shared_ptr<PolynomialFit> eye_centre_pos_y_fit{};
        std::shared_ptr<PolynomialFit> eye_centre_pos_z_fit{};
        std::shared_ptr<PolynomialFit> nodal_point_x_fit{};
        std::shared_ptr<PolynomialFit> nodal_point_y_fit{};
        std::shared_ptr<PolynomialFit> nodal_point_z_fit{};

        std::shared_ptr<PolynomialFit> pupil_x_fit{};
        std::shared_ptr<PolynomialFit> pupil_y_fit{};
        std::shared_ptr<PolynomialFit> ellipse_x_fit{};
        std::shared_ptr<PolynomialFit> ellipse_y_fit{};
        std::shared_ptr<PolynomialFit> ellipse_width_fit{};
        std::shared_ptr<PolynomialFit> ellipse_height_fit{};
        std::shared_ptr<PolynomialFit> ellipse_angle_fit{};

    private:
        void coeffsToMat(cv::Mat &mat);
    };

} // et

#endif //HDRMFS_EYE_TRACKER_POLYNOMIALEYEESTIMATOR_HPP
