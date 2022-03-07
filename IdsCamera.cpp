#include "IdsCamera.hpp"

#include <iostream>
#include <cassert>

namespace et
{
	IdsCamera::IdsCamera(int camera_index) : camera_index_(camera_index)
	{
		assert(camera_index_ >= 0);
	}

	void IdsCamera::initialize()
	{
		initializeCamera();
		initializeImage();
	}

	void IdsCamera::initializeCamera()
	{
		int result{};

		result = is_GetNumberOfCameras(&n_cameras_);
		assert(result == IS_SUCCESS);
		assert(n_cameras_ > 0 && camera_index_ < n_cameras_);
		
		camera_list_ = new UEYE_CAMERA_LIST[n_cameras_];
		result = is_GetCameraList(camera_list_);
		assert(result == IS_SUCCESS);

		camera_handle_ = camera_list_->uci[camera_index_].dwCameraID;

		result = is_InitCamera(&camera_handle_, nullptr);
		assert(result == IS_SUCCESS);

		result = is_SetColorMode(camera_handle_, IS_CM_MONO8);
		assert(result == IS_SUCCESS);

		result = is_GetSensorInfo(camera_handle_, &sensor_info_);
		assert(result == IS_SUCCESS);

		// double enable_auto_gain = 1.0;
		// result = is_SetAutoParameter(camera_handle_, IS_SET_ENABLE_AUTO_GAIN, &enable_auto_gain, nullptr);
		// assert(result == IS_SUCCESS);
	}

	void IdsCamera::initializeImage()
	{
		int result{};

		IS_RECT area_of_interest{};
		area_of_interest.s32X = 280;
		area_of_interest.s32Y = 280;
		area_of_interest.s32Width = 560;
		area_of_interest.s32Height = 464;
		result = is_AOI(camera_handle_, IS_AOI_IMAGE_SET_AOI, &area_of_interest, sizeof(area_of_interest));
		assert(result == IS_SUCCESS);

		image_resolution_.width = area_of_interest.s32Width;
		image_resolution_.height = area_of_interest.s32Height;

		result = is_AllocImageMem(camera_handle_, image_resolution_.width, image_resolution_.height, sizeof(char) * CHAR_BIT, &image_handle_, &image_id_);
		assert(result == IS_SUCCESS);

		result = is_SetImageMem(camera_handle_, image_handle_, image_id_);
		assert(result == IS_SUCCESS);

		image_.create(image_resolution_.height, image_resolution_.width, CV_8UC1);
	}

	cv::Mat IdsCamera::grabImage()
	{
		int result{};

		result = is_FreezeVideo(camera_handle_, IS_DONT_WAIT);
		assert(result == IS_SUCCESS);

		result = is_CopyImageMem(camera_handle_, image_handle_, image_id_, (char*)image_.data);
		assert(result == IS_SUCCESS);
		return image_;
	}

	cv::Size2i IdsCamera::getResolution()
	{
		return image_resolution_;
	}

	void IdsCamera::close()
	{
		int result{};

		result = is_FreeImageMem(camera_handle_, image_handle_, image_id_);
		assert(result == IS_SUCCESS);

		result = is_ExitCamera(camera_handle_);
		assert(result == IS_SUCCESS);
	}

	void IdsCamera::setExposure(double exposure)
	{
		int result{};

		result = is_Exposure(camera_handle_, IS_EXPOSURE_CMD_SET_EXPOSURE, (void*)&exposure, sizeof(exposure));
		assert(result == IS_SUCCESS);
	}

	void IdsCamera::setGamma(float gamma)
	{
		int result{};
		
		int scaled_gamma = gamma * 100;
		result = is_Gamma(camera_handle_, IS_GAMMA_CMD_SET, (void*)&scaled_gamma, sizeof(scaled_gamma));
		assert(result == IS_SUCCESS);
	}

	void IdsCamera::setFramerate(double framerate)
	{
		int result{};
		
		result = is_SetFrameRate(camera_handle_, framerate, &framerate);
		assert(result == IS_SUCCESS);
	}
}