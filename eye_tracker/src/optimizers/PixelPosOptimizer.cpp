#include "eye_tracker/optimizers/PixelPosOptimizer.hpp"

namespace et
{
    void PixelPosOptimizer::setParameters(std::shared_ptr<ModelEyeEstimator> model_eye_estimator,
                                          std::vector<cv::Point3d> front_corners, std::vector<cv::Point3d> back_corners,
                                          VisualAnglesOptimizer *eye_centre_optimizer,
                                          cv::Ptr<cv::DownhillSolver> eye_centre_solver,
                                          EyeAnglesOptimizer *eye_angles_optimizer,
                                          cv::Ptr<cv::DownhillSolver> eye_angles_solver, cv::Point3d marker_pos)
    {
        model_eye_estimator_ = model_eye_estimator;
        front_corners_ = front_corners;
        back_corners_ = back_corners;
        eye_centre_optimizer_ = eye_centre_optimizer;
        eye_centre_solver_ = eye_centre_solver;
        eye_angles_optimizer_ = eye_angles_optimizer;
        eye_angles_solver_ = eye_angles_solver;
        marker_pos_ = marker_pos;
    }

    int PixelPosOptimizer::getDims() const
    {
        return 6;
    }

    double PixelPosOptimizer::calc(const double *x) const
    {
        EyeInfo eye_info;
        eye_info.pupil = cv::Point2d{x[0], x[1]};
        eye_info.glints = {cv::Point2d{x[2], x[3]}, cv::Point2d{x[4], x[5]}};

        cv::Point3d nodal_point, eye_centre, visual_axis;
        bool success = model_eye_estimator_->detectEye(eye_info, nodal_point, eye_centre, visual_axis);
        if (!success)
        {
            return 1000000;
        }

        cv::Vec3d calculated_optical_axis = nodal_point - eye_centre;
        calculated_optical_axis = calculated_optical_axis / cv::norm(calculated_optical_axis);

        cv::Vec3d calculated_visual_axis = marker_pos_ - nodal_point;
        calculated_visual_axis = calculated_visual_axis / cv::norm(calculated_visual_axis);
        eye_angles_optimizer_->setParameters(calculated_visual_axis, calculated_optical_axis);
        cv::Mat x0 = (cv::Mat_<double>(1, 2) << 0.0, 0.0);
        cv::Mat step = cv::Mat::ones(x0.rows, x0.cols, CV_64F) * 0.1;
        eye_angles_solver_->setInitStep(step);
        eye_angles_solver_->minimize(x0);

        double alpha = x0.at<double>(0);
        double beta = x0.at<double>(1);

//        eye_centre_optimizer_->setParameters(front_corners_, back_corners_, alpha, beta, 5.3);
//        cv::Vec3d starting_point = eye_centre_optimizer_->getCrossPoint();
//
//        x0 = (cv::Mat_<double>(1, 3) << starting_point[0], starting_point[1], starting_point[2]);
//        eye_centre_optimizer_->calc(x0.ptr<double>());
//        step = cv::Mat::ones(x0.rows, x0.cols, CV_64F) * 0.1;
//        eye_centre_solver_->setInitStep(step);
//        eye_centre_solver_->minimize(x0);
//
//        cv::Point3d grid_eye_centre = cv::Point3d(x0.at<double>(0, 0), x0.at<double>(0, 1), x0.at<double>(0, 2));
//
//        double error = (grid_eye_centre - eye_centre).dot(grid_eye_centre - eye_centre);
        double error = 0;
        return error;
    }
} // et