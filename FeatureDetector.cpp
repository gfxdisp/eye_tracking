#include "FeatureDetector.hpp"

#include <opencv2/cudaarithm.hpp>

#include <iostream>

using KFMat = cv::Mat_<double>;

namespace et
{
	FeatureDetector::FeatureDetector(std::string template_path) : template_path_(std::move(template_path))
	{
	    spots_.upload(cv::imread(template_path_, cv::IMREAD_GRAYSCALE));
	    spots_matcher_ = cv::cuda::createTemplateMatching(CV_8UC1, cv::TM_CCOEFF_NORMED);
	}

	void FeatureDetector::initialize()
	{
		cv::Mat dilate_element_ = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(dilation_size_, dilation_size_));
        cv::Mat erode_element_ = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(erosion_size_, erosion_size_));
		cv::Mat morphology_element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(13, 13));
        open_filter_  = cv::cuda::createMorphologyFilter(cv::MORPH_OPEN,  CV_8UC1, morphology_element);
        close_filter_ = cv::cuda::createMorphologyFilter(cv::MORPH_CLOSE, CV_8UC1, morphology_element);

       	dilate_filter_ = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, CV_8UC1, dilate_element_);
        erode_filter_ = cv::cuda::createMorphologyFilter(cv::MORPH_ERODE, CV_8UC1, erode_element_);

        
	}

	void FeatureDetector::initializeKalmanFilters(cv::Size2i resolution, float framerate)
	{
		pupil_kalman_ = makeKalmanFilter(resolution, framerate);
        for (int i = 0; i < LED_COUNT; i++)
        {
        	led_kalmans_[i] = makeKalmanFilter(resolution, framerate);
        }
	}

	bool FeatureDetector::findImageFeatures(cv::Mat image)
	{
		gpu_image_.upload(image);
		cv::cuda::threshold(gpu_image_, pupil_thresholded_image_, pupil_threshold, 255, cv::THRESH_BINARY_INV);
		cv::cuda::threshold(gpu_image_, glints_thresholded_image_, glints_threshold_, 255, cv::THRESH_BINARY);
    	return findPupil() & findGlints();
	}

	cv::Point2f FeatureDetector::getPupil()
	{
		return pupil_location_;
	}

	void FeatureDetector::getPupil(cv::Point2f &pupil)
	{
		mtx_features_.lock();
		pupil = pupil_location_;
		mtx_features_.unlock();
	}

	float FeatureDetector::getPupilRadius()
	{
		return pupil_radius_;
	}

	void FeatureDetector::getPupilRadius(float &pupil_radius)
	{
		mtx_features_.lock();
		pupil_radius = pupil_radius_;
		mtx_features_.unlock();
	}

	cv::Point2f *FeatureDetector::getLeds()
	{
		return leds_locations_;
	}

	void FeatureDetector::getLeds(cv::Point2f *leds_locations)
	{
		mtx_features_.lock();
		for (int i = 0; i < LED_COUNT; i++)
		{
			leds_locations[i] = leds_locations_[i];
		}
		mtx_features_.unlock();
	}

	bool FeatureDetector::findPupil()
	{
		pupil_thresholded_image_.download(cpu_image_);

        cv::findContours(cpu_image_, contours_, hierarchy_, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::Point2f best_centre{};
        float best_radius{};
        float best_rating{0};

        static cv::Size image_size{pupil_thresholded_image_.size()};
        static cv::Point2f image_centre{cv::Point2f(image_size.width / 2, image_size.height / 2)};
        static float max_distance{std::max(image_size.width, image_size.height) / 2.0f};

        for (const std::vector<cv::Point> &contour: contours_) {
            cv::Point2f centre;
            float radius;

            cv::minEnclosingCircle(contour, centre, radius);
            if (radius < min_pupil_radius_ or radius > max_pupil_radius_) continue;

            float distance = euclideanDistance(centre, image_centre);
            if (distance > max_distance) continue;

            const float contour_area = cv::contourArea(contour);
            if (contour_area <= 0) continue;
            const float circle_area = 3.14159 * std::pow(radius, 2);
            float rating = contour_area / circle_area * (1.0f - distance / max_distance);
            if (rating >= best_rating)
            {
            	best_centre = centre;
            	best_rating = rating;
            	best_radius = radius;
            }
        }

        if (best_rating == 0)
        {
        	return false;
        }

        pupil_kalman_.correct((KFMat(2, 1) << best_centre.x, best_centre.y));
		mtx_features_.lock();
        pupil_location_ = toPoint(pupil_kalman_.predict());
        pupil_radius_ = best_radius;
		mtx_features_.unlock();
        return  true;
	}

	bool FeatureDetector::findGlints()
	{
		// dilate_filter_->apply(glints_thresholded_image_, glints_thresholded_image_);
        glints_thresholded_image_.download(cpu_image_);

		cv::Point2f best_centre[LED_COUNT]{};
        float best_radius{};
        float best_rating{0};

		cv::findContours(cpu_image_, contours_, hierarchy_, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        static cv::Size image_size{glints_thresholded_image_.size()};	
        static cv::Point2f image_centre{cv::Point2f(image_size.width / 2, image_size.height / 2)};
        static float max_distance{std::max(image_size.width, image_size.height) / 2.0f};

        std::vector<GlintCandidate> glint_candidates{};
        glint_candidates.reserve(contours_.size());

		for (const std::vector<cv::Point> &contour: contours_) 
		{
			cv::Point2f centre;
            float radius;
            cv::minEnclosingCircle(contour, centre, radius);
            if (radius > max_glint_radius_) continue;

            float distance = euclideanDistance(centre, image_centre);
            if (distance > max_distance) continue;

            const float contour_area = cv::contourArea(contour);
            if (contour_area <= 0) continue;
            const float circle_area = 3.14159 * std::pow(radius, 2);
            float rating = contour_area / circle_area * (1.0f - distance / max_distance) * radius / max_glint_radius_;
        	GlintCandidate glint_candidate{};
        	glint_candidate.location = centre;
        	glint_candidate.rating = rating;
        	glint_candidates.push_back(glint_candidate);
		}

		for (int i = 0; i < glint_candidates.size(); i++)
		{
			for (int j = i + 1; j < glint_candidates.size(); j++)
			{
				if (abs(glint_candidates[i].location.y - glint_candidates[j].location.y) > max_vertical_distance) continue;
				if (abs(glint_candidates[i].location.x - glint_candidates[j].location.x) > max_horizontal_distance) continue;
				if (abs(glint_candidates[i].location.x - glint_candidates[j].location.x) < min_horizontal_distance) continue;
				float rating = glint_candidates[i].rating + glint_candidates[j].rating;
				if (rating > best_rating)
				{
					best_rating = rating;
					best_centre[0] = glint_candidates[i].location;
					best_centre[1] = glint_candidates[j].location;
					if (best_centre[0].y > best_centre[1].y)
					{
						std::swap(best_centre[0], best_centre[1]);
					}
				}
			}
		}

		if (best_rating == 0)
		{
			return false;
		}

		for (int i = 0; i < LED_COUNT; i++)
		{
			led_kalmans_[i].correct((KFMat(2, 1) << best_centre[i].x, best_centre[i].y));
			mtx_features_.lock();
	    	leds_locations_[i] = toPoint(led_kalmans_[i].predict());	
			mtx_features_.unlock();
		}
		return true;
	}

    cv::KalmanFilter FeatureDetector::makeKalmanFilter(cv::Size2i resolution, float framerate) {
        constexpr static double VELOCITY_DECAY = 0.9;
        const static cv::Mat TRANSITION_MATRIX = (KFMat(4, 4) << 1, 0, 1.0f / framerate, 0,
                0, 1, 0, 1.0f / framerate,
                0, 0, VELOCITY_DECAY, 0,
                0, 0, 0, VELOCITY_DECAY);
        const static cv::Mat MEASUREMENT_MATRIX = (KFMat(2, 4) << 1, 0, 0, 0,
                0, 1, 0, 0);
        const static cv::Mat PROCESS_NOISE_COV = cv::Mat::eye(4, 4, CV_64F) * 100;
        const static cv::Mat MEASUREMENT_NOISE_COV = cv::Mat::eye(2, 2, CV_64F) * 50;
        const static cv::Mat ERROR_COV_POST = cv::Mat::eye(4, 4, CV_64F) * 0.1;
        const static cv::Mat STATE_POST = (KFMat(4, 1) << resolution.width / 2.0, resolution.height / 2.0, 0, 0);

        cv::KalmanFilter KF(4, 2);
        // clone() is needed as, otherwise, the matrices will be used by reference, and all the filters will be the same
        KF.transitionMatrix = TRANSITION_MATRIX.clone();
        KF.measurementMatrix = MEASUREMENT_MATRIX.clone();
        KF.processNoiseCov = PROCESS_NOISE_COV.clone();
        KF.measurementNoiseCov = MEASUREMENT_NOISE_COV.clone();
        KF.errorCovPost = ERROR_COV_POST.clone();
        KF.statePost = STATE_POST.clone();
        KF.predict(); // Without this line, OpenCV complains about incorrect matrix dimensions
        return KF;
    }

	cv::Mat FeatureDetector::getThresholdedPupilImage()
	{
		cv::Mat image_{};
		pupil_thresholded_image_.download(image_);
		return image_;
	}

	cv::Mat FeatureDetector::getThresholdedGlintsImage()
	{
		cv::Mat image_{};
		glints_thresholded_image_.download(image_);
		return image_;
	}

}