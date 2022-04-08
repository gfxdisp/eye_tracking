#include "IdsCamera.hpp"

#include <cassert>
#include <iostream>

namespace et {
IdsCamera::IdsCamera(int camera_index) : camera_index_(camera_index) {
    assert(camera_index_ >= 0);
    image_resolution_ = cv::Size2i(560, 464);
}

void IdsCamera::initialize() {
    initializeCamera();
    initializeImage();
    image_gatherer_ = std::thread{&IdsCamera::imageGatheringThread, this};
}

void IdsCamera::initializeCamera() {
    int result;

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

void IdsCamera::initializeImage() {
    int result;

    IS_RECT area_of_interest{};
    area_of_interest.s32X = offset_.x;
    area_of_interest.s32Y = offset_.y;
    area_of_interest.s32Width = image_resolution_.width;
    area_of_interest.s32Height = image_resolution_.height;
    result = is_AOI(camera_handle_, IS_AOI_IMAGE_SET_AOI, &area_of_interest, sizeof(area_of_interest));
    assert(result == IS_SUCCESS);

    result = is_AllocImageMem(camera_handle_, area_of_interest.s32Width, area_of_interest.s32Height,
                              sizeof(char) * CHAR_BIT, &image_handle_, &image_id_);
    assert(result == IS_SUCCESS);

    result = is_SetImageMem(camera_handle_, image_handle_, image_id_);
    assert(result == IS_SUCCESS);

    image_resolution_.width = area_of_interest.s32Width;
    image_resolution_.height = area_of_interest.s32Height;

    image_.create(image_resolution_.height, image_resolution_.width, CV_8UC1);
    for (auto & i : image_queue_) {
        i.create(image_resolution_.height, image_resolution_.width, CV_8UC1);
    }
}

void IdsCamera::imageGatheringThread() {
    int result;
    int new_image_index{image_index_};

    while (thread_running_) {
        new_image_index = (new_image_index + 1) % IMAGE_IN_QUEUE_COUNT;

        result = is_FreezeVideo(camera_handle_, IS_WAIT);
        assert(result == IS_SUCCESS);

        result = is_CopyImageMem(camera_handle_, image_handle_, image_id_, (char *)image_queue_[new_image_index].data);
        assert(result == IS_SUCCESS);

        image_index_ = new_image_index;
    }
}

cv::Mat IdsCamera::grabImage() {
    image_ = image_queue_[image_index_];
    return image_;
}

cv::Size2i IdsCamera::getImageResolution() {
    return image_resolution_;
}

cv::Size2i IdsCamera::getDeviceResolution() {
    return {1280, 1024};
}


void IdsCamera::close() {
    int result;

    thread_running_ = false;
    image_gatherer_.join();

    result = is_FreeImageMem(camera_handle_, image_handle_, image_id_);
    assert(result == IS_SUCCESS);

    result = is_ExitCamera(camera_handle_);
    assert(result == IS_SUCCESS);
}

void IdsCamera::setExposure(double exposure) {
    int result;

    result = is_Exposure(camera_handle_, IS_EXPOSURE_CMD_SET_EXPOSURE, (void *)&exposure, sizeof(exposure));
    assert(result == IS_SUCCESS);
}

void IdsCamera::setGamma(float gamma) {
    int result;

    int scaled_gamma = static_cast<int>(gamma * 100);
    result = is_Gamma(camera_handle_, IS_GAMMA_CMD_SET, (void *)&scaled_gamma, sizeof(scaled_gamma));
    assert(result == IS_SUCCESS);
}

void IdsCamera::setFramerate(double framerate) {
    int result;

    result = is_SetFrameRate(camera_handle_, framerate, &framerate);
    assert(result == IS_SUCCESS);
}
cv::Point2d IdsCamera::getOffset() {
    return offset_;
}
}// namespace et