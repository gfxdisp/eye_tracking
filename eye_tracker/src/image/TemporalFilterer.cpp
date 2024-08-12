#include <eye_tracker/Settings.hpp>
#include <eye_tracker/image/TemporalFilterer.hpp>

namespace et {
    bool TemporalFilterer::ransac = true;

    TemporalFilterer::TemporalFilterer(const int camera_id) : camera_id_(camera_id) {
        const auto& detection_params = et::Settings::parameters.detection_params[camera_id];
        const auto& camera_params = et::Settings::parameters.camera_params[camera_id];
        const auto& region_of_interest = camera_params.region_of_interest;
        pupil_kalman_ = createPixelKalmanFilter(region_of_interest, camera_params.framerate);

        glints_kalman_ = createPixelKalmanFilter(region_of_interest, camera_params.framerate);
        pupil_radius_kalman_ = createRadiusKalmanFilter(detection_params.min_pupil_radius, detection_params.max_pupil_radius, camera_params.framerate);

        bayes_minimizer_ = std::make_shared<GlintCircleOptimizer>();
        bayes_minimizer_func_ = cv::Ptr<cv::DownhillSolver::Function>{bayes_minimizer_};
        bayes_solver_ = cv::DownhillSolver::create();
        bayes_solver_->setFunction(bayes_minimizer_func_);
        const cv::Mat step = (cv::Mat_<double>(1, 3) << 100, 100, 100);
        bayes_solver_->setInitStep(step);
    }

    void TemporalFilterer::filterPupil(cv::Point2d& pupil, double& radius) {
        pupil_kalman_.correct((cv::Mat_<double>(2, 1) << pupil.x, pupil.y));
        pupil_radius_kalman_.correct(cv::Mat_<double>(1, 1) << radius);
        cv::Mat pupil_kalman = pupil_kalman_.predict();
        pupil = cv::Point2d(pupil_kalman.at<double>(0, 0), pupil_kalman.at<double>(1, 0));
        radius = pupil_radius_kalman_.predict().at<double>(0, 0);
    }

    void TemporalFilterer::filterGlints(std::vector<cv::Point2f>& glints) {
        std::sort(glints.begin(), glints.end(), [this](auto const& a, auto const& b) {
            const double distance_a = euclideanDistance(a, circle_centre_);
            const double distance_b = euclideanDistance(b, circle_centre_);
            return distance_a < distance_b;
        });

        glints.resize(std::min(static_cast<int>(glints.size()), 20));

        const static cv::Point2d im_centre{et::Settings::parameters.camera_params[camera_id_].region_of_interest / 2};
        std::string bitmask(3, 1);
        std::vector<cv::Point2d> circle_points{};
        circle_points.resize(3);
        int best_counter = 0;
        cv::Point2d best_circle_centre{};
        double best_circle_radius{};
        bitmask.resize(glints.size() - 3, 0);
        cv::Point2d ellipse_centre{};
        // Loop on every possible triplet of glints.
        do {
            int counter = 0;
            for (int i = 0; counter < 3; i++) {
                if (bitmask[i]) {
                    circle_points[counter] = glints[i];
                    counter++;
                }
            }
            bayes_minimizer_->setParameters(circle_points, circle_centre_, circle_radius_);

            cv::Mat x = (cv::Mat_<double>(1, 3) << im_centre.x, im_centre.y, circle_radius_);
            bayes_solver_->minimize(x);
            ellipse_centre.x = x.at<double>(0, 0);
            ellipse_centre.y = x.at<double>(0, 1);
            double ellipse_radius = std::abs(x.at<double>(0, 2));

            counter = 0;
            for (const auto& glint: glints) {
                double value{0.0};
                value += (ellipse_centre.x - glint.x) * (ellipse_centre.x - glint.x);
                value += (ellipse_centre.y - glint.y) * (ellipse_centre.y - glint.y);
                if (std::abs(std::sqrt(value) - ellipse_radius) <= 3.0 || !ransac) {
                    counter++;
                }
            }

            if (counter > best_counter) {
                best_counter = counter;
                best_circle_centre = ellipse_centre;
                best_circle_radius = ellipse_radius;
            }
        } while (std::prev_permutation(bitmask.begin(), bitmask.end()));

        circle_centre_ = best_circle_centre;
        circle_radius_ = best_circle_radius;

        if (best_counter < 3) {
            glints = {};
            return;
        }

        // Remove all glints that are too far from the estimated circle.

        std::erase_if(glints, [this](auto const& p) {
            double value{0.0};
            value += (circle_centre_.x - p.x) * (circle_centre_.x - p.x);
            value += (circle_centre_.y - p.y) * (circle_centre_.y - p.y);
            return std::abs(std::sqrt(value) - circle_radius_) > 3.0;
        });
    }

    cv::KalmanFilter TemporalFilterer::createPixelKalmanFilter(const cv::Size2i& resolution, double framerate) {
        double velocity_decay = 0.9f;
        cv::Mat transition_matrix{(cv::Mat_<double>(4, 4) << 1, 0, 1.0 / framerate, 0, 0, 1, 0, 1.0 / framerate, 0, velocity_decay, 0, 0, 0, 0, 0, velocity_decay)};
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

    cv::KalmanFilter TemporalFilterer::createRadiusKalmanFilter(const double& min_radius, const double& max_radius, double framerate) {
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

    TemporalFilterer::~TemporalFilterer() {
        if (!bayes_solver_->empty()) {
            bayes_solver_.release();
        }
        if (!bayes_minimizer_func_.empty()) {
            bayes_minimizer_func_.release();
        }
    }
} // et
