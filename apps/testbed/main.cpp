#include <eye_tracker/Utils.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <eye_tracker/optimizers/GlintPositionOptimizer.hpp>
#include <eye_tracker/optimizers/PupilPositionOptimizer.hpp>

int main()
{
    cv::Vec3d nodal_point = cv::Point3d(100, 200, 850);
    cv::Vec3d optical_axis = cv::Point3d(0.1, 0.2, -0.8);
    optical_axis = optical_axis / cv::norm(optical_axis);
    double cornea_radius = 7.8;
    double pupil_cornea_dist = 4.2;
    double refraction_index = 1.3375;
    cv::Vec3d pupil_position = nodal_point + pupil_cornea_dist * optical_axis;
    cv::Vec3d camera_position = cv::Point3d(56, 181, 550);

    auto pupil_position_optimizer = new et::PupilPositionOptimizer();
    auto pupil_minimizer_function = cv::Ptr<cv::DownhillSolver::Function>{pupil_position_optimizer};
    auto pupil_solver = cv::DownhillSolver::create();
    pupil_solver->setFunction(pupil_minimizer_function);
    cv::Mat step = (cv::Mat_<double>(1, 3) << 0.5, 0.5, 0.5);
    pupil_solver->setInitStep(step);

    pupil_position_optimizer->setParameters(nodal_point, pupil_position, camera_position,
                                            cornea_radius);

    cv::Mat x = (cv::Mat_<double>(1, 3) << 1.0, 1.0, 1.0);
    pupil_solver->minimize(x);
    cv::Vec3d ray_direction;
    cv::multiply(-static_cast<cv::Vec3d>(pupil_position - camera_position),
                 cv::Vec3d(x.at<double>(0), x.at<double>(1), x.at<double>(2)), ray_direction);
    cv::normalize(ray_direction, ray_direction);

    double t{};
    et::Utils::getRaySphereIntersection(camera_position, ray_direction, nodal_point, cornea_radius,
                                        t);

    std::cout << "Pupil position: " << pupil_position << std::endl;
    pupil_position = static_cast<cv::Vec3d>(camera_position) + t * ray_direction;


    std::cout << "Pupil position: " << pupil_position << std::endl;

    return 0;
}
