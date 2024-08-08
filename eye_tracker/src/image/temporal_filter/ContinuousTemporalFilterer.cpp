#include "eye_tracker/image/temporal_filter/ContinuousTemporalFilterer.hpp"
#include "eye_tracker/Settings.hpp"

namespace et
{
    bool ContinuousTemporalFilterer::ransac = true;

    ContinuousTemporalFilterer::ContinuousTemporalFilterer(int camera_id) : TemporalFilterer(camera_id)
    {
        region_of_interest_ = &et::Settings::parameters.camera_params[camera_id].region_of_interest;
        pupil_kalman_ = createPixelKalmanFilter(*region_of_interest_,
                                                et::Settings::parameters.camera_params[camera_id].framerate);

        glints_kalman_ = createPixelKalmanFilter(*region_of_interest_,
                                                 et::Settings::parameters.camera_params[camera_id].framerate);
        pupil_radius_kalman_ = createRadiusKalmanFilter(
                et::Settings::parameters.detection_params[camera_id].min_pupil_radius,
                et::Settings::parameters.detection_params[camera_id].max_pupil_radius,
                et::Settings::parameters.camera_params[camera_id].framerate);
        glint_ellipse_kalman_ = createEllipseKalmanFilter(*region_of_interest_,
                                                          et::Settings::parameters.camera_params[camera_id].framerate);

        bayes_minimizer_ = new GlintCircleOptimizer();
        bayes_minimizer_func_ = cv::Ptr<cv::DownhillSolver::Function>{bayes_minimizer_};
        bayes_solver_ = cv::DownhillSolver::create();
        bayes_solver_->setFunction(bayes_minimizer_func_);
        cv::Mat step = (cv::Mat_<double>(1, 3) << 100, 100, 100);
        bayes_solver_->setInitStep(step);
    }

    void ContinuousTemporalFilterer::filterPupil(cv::Point2d &pupil, double &radius)
    {
        pupil_kalman_.correct((cv::Mat_<double>(2, 1) << pupil.x, pupil.y));
        pupil_radius_kalman_.correct((cv::Mat_<double>(1, 1) << radius));
        pupil = toPoint(pupil_kalman_.predict());
        radius = toValue(pupil_radius_kalman_.predict());
    }

    void ContinuousTemporalFilterer::filterGlints(std::vector<cv::Point2f> &glints)
    {
        // Limits the number of glints to 20 which are the closest to the previously
        // estimated circle.
        std::sort(glints.begin(), glints.end(), [this](auto const &a, auto const &b)
        {
            double distance_a = euclideanDistance(a, circle_centre_);
            double distance_b = euclideanDistance(b, circle_centre_);
            return distance_a < distance_b;
        });

        glints.resize(std::min((int) glints.size(), 20));

        cv::Point2d im_centre{*region_of_interest_ / 2};
        std::string bitmask(3, 1);
        std::vector<cv::Point2d> circle_points{};
        circle_points.resize(3);
        int best_counter = 0;
        cv::Point2d best_circle_centre{};
        double best_circle_radius{};
        bitmask.resize(glints.size() - 3, 0);
        cv::Point2d ellipse_centre{};
        double ellipse_radius;
        // Loop on every possible triplet of glints.
        do
        {
            int counter = 0;
            for (int i = 0; counter < 3; i++)
            {
                if (bitmask[i])
                {
                    circle_points[counter] = glints[i];
                    counter++;
                }
            }
            bayes_minimizer_->setParameters(circle_points, circle_centre_, circle_radius_);

            cv::Mat x = (cv::Mat_<double>(1, 3) << im_centre.x, im_centre.y, circle_radius_);
            // Find the most likely position of the circle based on the previous
            // circle position and triplet of glints that should lie on it.
            bayes_solver_->minimize(x);
            ellipse_centre.x = x.at<double>(0, 0);
            ellipse_centre.y = x.at<double>(0, 1);
            ellipse_radius = std::abs(x.at<double>(0, 2));

            counter = 0;
            // Counts all glints that lie close to the circle
            for (auto &glint: glints)
            {
                double value{0.0};
                value += (ellipse_centre.x - glint.x) * (ellipse_centre.x - glint.x);
                value += (ellipse_centre.y - glint.y) * (ellipse_centre.y - glint.y);
                if (std::abs(std::sqrt(value) - ellipse_radius) <= 3.0 || !ransac)
                {
                    counter++;
                }
            }
            // Finds the circle which contains the most glints.
            if (counter > best_counter)
            {
                best_counter = counter;
                best_circle_centre = ellipse_centre;
                best_circle_radius = ellipse_radius;
            }
        } while (std::prev_permutation(bitmask.begin(), bitmask.end()));

        circle_centre_ = best_circle_centre;
        circle_radius_ = best_circle_radius;

        if (best_counter < 3)
        {
            glints = {};
            return;
        }

        // Remove all glints that are too far from the estimated circle.

        glints.erase(std::remove_if(glints.begin(), glints.end(), [this](auto const& p) {
            double value{0.0};
            value += (circle_centre_.x - p.x) * (circle_centre_.x - p.x);
            value += (circle_centre_.y - p.y) * (circle_centre_.y - p.y);
            return std::abs(std::sqrt(value) - circle_radius_) > 3.0;
        }), glints.end());
    }

    void ContinuousTemporalFilterer::filterEllipse(cv::RotatedRect &ellipse)
    {
        glint_ellipse_kalman_.correct((cv::Mat_<double>(5, 1)
                << ellipse.center.x, ellipse.center.y, ellipse.size.width, ellipse.size.height, ellipse.angle));

        cv::Mat predicted_ellipse = glint_ellipse_kalman_.predict();
        ellipse.center.x = predicted_ellipse.at<double>(0, 0);
        ellipse.center.y = predicted_ellipse.at<double>(0, 1);
        ellipse.size.width = predicted_ellipse.at<double>(0, 2);
        ellipse.size.height = predicted_ellipse.at<double>(0, 3);
        ellipse.angle = predicted_ellipse.at<double>(0, 4);
    }

    cv::KalmanFilter ContinuousTemporalFilterer::createPixelKalmanFilter(const cv::Size2i &resolution, double framerate)
    {
        double velocity_decay = 0.9f;
        cv::Mat transition_matrix{(cv::Mat_<double>(4, 4) << 1, 0, 1.0 / framerate, 0, 0, 1, 0, 1.0 /
                                                                                                framerate, 0, velocity_decay, 0, 0, 0, 0, 0, velocity_decay)};
        cv::Mat measurement_matrix{(cv::Mat_<double>(2, 4) << 1, 0, 0, 0, 0, 1, 0, 0)};
        cv::Mat process_noise_cov{cv::Mat::eye(4, 4, CV_64F) * 2};
        cv::Mat measurement_noise_cov{cv::Mat::eye(2, 2, CV_64F) * 1};
        cv::Mat error_cov_post{cv::Mat::eye(4, 4, CV_64F)};
        cv::Mat state_post{(cv::Mat_<double>(4, 1) << resolution.width / 2, resolution.height / 2, 0, 0)};

        cv::KalmanFilter KF(4, 2, CV_64F);
        KF.transitionMatrix = transition_matrix;
        KF.measurementMatrix = measurement_matrix;
        KF.processNoiseCov = process_noise_cov;
        KF.measurementNoiseCov = measurement_noise_cov;
        KF.errorCovPost = error_cov_post;
        KF.statePost = state_post;
        // Without this line, OpenCV complains about incorrect matrix dimensions.
        KF.predict();
        return KF;
    }

    cv::KalmanFilter
    ContinuousTemporalFilterer::createRadiusKalmanFilter(const double &min_radius, const double &max_radius,
                                                         double framerate)
    {
        double velocity_decay = 0.9f;
        cv::Mat transition_matrix{(cv::Mat_<double>(2, 2) << 1, 1.0 / framerate, 0, velocity_decay)};
        cv::Mat measurement_matrix{(cv::Mat_<double>(1, 2) << 1, 0)};
        cv::Mat process_noise_cov{cv::Mat::eye(2, 2, CV_64F) * 2};
        cv::Mat measurement_noise_cov{(cv::Mat_<double>(1, 1) << 1)};
        cv::Mat error_cov_post{cv::Mat::eye(2, 2, CV_64F)};
        cv::Mat state_post{(cv::Mat_<double>(2, 1) << (max_radius - min_radius) / 2, 0)};

        cv::KalmanFilter KF(2, 1, CV_64F);
        KF.transitionMatrix = transition_matrix;
        KF.measurementMatrix = measurement_matrix;
        KF.processNoiseCov = process_noise_cov;
        KF.measurementNoiseCov = measurement_noise_cov;
        KF.errorCovPost = error_cov_post;
        KF.statePost = state_post;
        // Without this line, OpenCV complains about incorrect matrix dimensions.
        KF.predict();
        return KF;
    }

    cv::KalmanFilter
    ContinuousTemporalFilterer::createEllipseKalmanFilter(const cv::Size2i &resolution, double framerate)
    {
        double velocity_decay = 0.9f;
        cv::Mat transition_matrix{
                (cv::Mat_<double>(10, 10) << 1, 0, 0, 0, 0, 1.0 / framerate, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1.0 /
                                                                                                           framerate, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                        1.0 / framerate, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1.0 / framerate, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                        1.0 /
                        framerate, 0, 0, 0, 0, 0, velocity_decay, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, velocity_decay, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, velocity_decay, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, velocity_decay, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, velocity_decay)};
        cv::Mat measurement_matrix{(cv::Mat_<double>(5, 10)
                << 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0)};
        cv::Mat process_noise_cov{cv::Mat::eye(10, 10, CV_64F) * 2};
        cv::Mat measurement_noise_cov{cv::Mat::eye(5, 5, CV_64F) * 5};
        cv::Mat error_cov_post{cv::Mat::eye(10, 10, CV_64F)};
        cv::Mat state_post{(cv::Mat_<double>(10, 1) << resolution.width / 2, resolution.height / 2, resolution.width / 4, resolution.height / 4, 0, 0, 0, 0, 0,
                0)};

        cv::KalmanFilter KF(10, 5, CV_64F);
        KF.transitionMatrix = transition_matrix;
        KF.measurementMatrix = measurement_matrix;
        KF.processNoiseCov = process_noise_cov;
        KF.measurementNoiseCov = measurement_noise_cov;
        KF.errorCovPost = error_cov_post;
        KF.statePost = state_post;
        KF.predict();
        return KF;
    }

    ContinuousTemporalFilterer::~ContinuousTemporalFilterer()
    {
        if (!bayes_solver_->empty())
        {
            bayes_solver_.release();
        }
        if (!bayes_minimizer_func_.empty())
        {
            bayes_minimizer_func_.release();
        }

    }
} // et