#ifndef EYE_TRACKER_H
#define EYE_TRACKER_H

#include "FeatureDetector.hpp"

#include "ImageProvider.hpp"

#include <opencv2/opencv.hpp>

#include <mutex>
#include <optional>
#include <vector>

namespace et
{
	struct EyePosition
	{
		std::optional<cv::Vec3d> cornea_curvature{};
		std::optional<cv::Vec3d> pupil{};
		std::optional<cv::Vec3d> eye_centre{};
		inline operator bool() const 
		{
			return eye_centre and pupil and cornea_curvature;
		}
	};

	struct EyeProperties
	{
		float cornea_curvature_radius{7.8f};
		float pupil_cornea_distance{4.2f};
		float refraction_index{1.3375f};
		float pupil_eye_centre_distance{11.6f};
	};

	struct SetupLayout
	{
		double camera_lambda;
		cv::Vec3d camera_nodal_point_position;
		cv::Vec3d led_positions[FeatureDetector::LED_COUNT];
		double camera_eye_distance;
		double camera_eye_projection_factor;
	};

	class EyeTracker
	{
	public:
		EyeTracker(SetupLayout &setup_layout, ImageProvider *image_provider);

		void calculateEyePosition(cv::Point2f pupil_pixel_position, cv::Point2f glints_pixel_positions[]);

		void calculateJoined(cv::Point2f pupil_pixel_position, cv::Point2f glints_pixel_positions[]);

		EyePosition getEyePosition();

		void getCorneaCurvaturePosition(cv::Vec3d &eye_centre);

		void getGazeDirection(cv::Vec3d &gaze_direction);

		cv::Point2d getCorneaCurvaturePixelPosition();

		void setNewSetupLayout(SetupLayout &setup_layout);

		void initializeKalmanFilter(float framerate);

	private:
		SetupLayout setup_layout_{};
		SetupLayout new_setup_layout_{};
		bool new_setup_layout_needed_{false};

		EyeProperties eye_properties_{};
		ImageProvider *image_provider_{};
		cv::KalmanFilter kalman_{};
		EyePosition eye_position_{};
        std::mutex mtx_eye_position_{};
        std::mutex mtx_setup_to_change_{};

		inline cv::Vec3d project(cv::Point2f point) const 
		{ 
			return project(ICStoWCS(point));
		}

		inline cv::Vec3d ICStoWCS(cv::Point2d point) const 
		{ 
			return CCStoWCS(ICStoCCS(point));
		}

		inline cv::Point2d WCStoICS(cv::Vec3d point) const { 
			return CCStoICS(WCStoCCS(point)); 
		}
		
		cv::Vec3d project(cv::Vec3d point) const;

		cv::Point2d unproject(cv::Vec3d point) const;

		cv::Vec3d ICStoCCS(cv::Point2d point) const;
		
		cv::Vec3d CCStoWCS(cv::Vec3d point) const;

		cv::Vec3d WCStoCCS(cv::Vec3d point) const;

		cv::Point2d CCStoICS(cv::Vec3d point) const;
		
		std::vector<cv::Vec3d> lineSphereIntersections(cv::Vec3d sphere_centre, float radius, cv::Vec3d line_point, cv::Vec3d line_direction);
		
		cv::KalmanFilter makeKalmanFilter(float framerate) const;


	};
}

#endif