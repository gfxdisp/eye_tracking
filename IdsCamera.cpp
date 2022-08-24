#include "IdsCamera.hpp"
#include "Settings.hpp"

#include <cassert>
#include <iostream>

namespace et {

void IdsCamera::initialize() {
    initializeCamera();
    initializeImage();
    image_gatherer_ = std::thread{&IdsCamera::imageGatheringThread, this};
}

void IdsCamera::initializeCamera() {
    int result;

    result = is_GetNumberOfCameras(&n_cameras_);
    assert(result == IS_SUCCESS);
    assert(n_cameras_ > 0);

    camera_list_ = new UEYE_CAMERA_LIST[n_cameras_];
    result = is_GetCameraList(camera_list_);
    assert(result == IS_SUCCESS);

    int camera_index{-1};

    for (int i = 0; i < n_cameras_; i++) {
        if (std::string(camera_list_->uci[i].Model)
            == et::Settings::parameters.camera_params.name) {
            camera_index = i;
            break;
        }
    }
    assert(camera_index >= 0);

    camera_handle_ = camera_list_->uci[camera_index].dwCameraID;

    result = is_InitCamera(&camera_handle_, nullptr);
    assert(result == IS_SUCCESS);

    result = is_SetColorMode(camera_handle_, IS_CM_MONO16);
    assert(result == IS_SUCCESS);

    result = is_GetSensorInfo(camera_handle_, &sensor_info_);
    assert(result == IS_SUCCESS);

    double enable_auto_gain = 0.0;
    result = is_SetAutoParameter(camera_handle_, IS_SET_ENABLE_AUTO_GAIN, &enable_auto_gain, nullptr);
    assert(result == IS_SUCCESS);
}

void IdsCamera::initializeImage() {
    int result;

    IS_RECT area_of_interest{};
    area_of_interest.s32X =
        Settings::parameters.camera_params.capture_offset.width;
    area_of_interest.s32Y =
        Settings::parameters.camera_params.capture_offset.height;
    area_of_interest.s32Width =
        Settings::parameters.camera_params.region_of_interest.width;
    area_of_interest.s32Height =
        Settings::parameters.camera_params.region_of_interest.height;

    result = is_AOI(camera_handle_, IS_AOI_IMAGE_SET_AOI, &area_of_interest,
                    sizeof(area_of_interest));
    assert(result == IS_SUCCESS);

    result = is_AllocImageMem(
        camera_handle_, area_of_interest.s32Width, area_of_interest.s32Height,
        sizeof(char) * CHAR_BIT * 2, &image_handle_, &image_id_);
    assert(result == IS_SUCCESS);

    result = is_SetImageMem(camera_handle_, image_handle_, image_id_);
    assert(result == IS_SUCCESS);

    image_.create(area_of_interest.s32Height, area_of_interest.s32Width,
                  CV_16UC1);
    for (auto &i : pupil_image_queue_) {
        i.create(area_of_interest.s32Height, area_of_interest.s32Width,
                 CV_16UC1);
    }

    for (auto &i : glint_image_queue_) {
        i.create(area_of_interest.s32Height, area_of_interest.s32Width,
                 CV_16UC1);
    }
}

void IdsCamera::imageGatheringThread() {
    int result;
    int new_image_index{image_index_};
    while (thread_running_) {
        new_image_index = (new_image_index + 1) % IMAGE_IN_QUEUE_COUNT;

        setExposure(et::Settings::parameters.camera_params.pupil_exposure);
        result = is_FreezeVideo(camera_handle_, IS_WAIT);
        assert(result == IS_SUCCESS);
        result = is_CopyImageMem(camera_handle_, image_handle_, image_id_,
                                 (char *) pupil_image_queue_[new_image_index].data);
        assert(result == IS_SUCCESS);

        setExposure(et::Settings::parameters.camera_params.glint_exposure);
        result = is_FreezeVideo(camera_handle_, IS_WAIT);
        assert(result == IS_SUCCESS);
        result = is_CopyImageMem(camera_handle_, image_handle_, image_id_,
                                 (char *) glint_image_queue_[new_image_index].data);
        assert(result == IS_SUCCESS);

        image_index_ = new_image_index;
    }
}

cv::Mat IdsCamera::grabPupilImage() {
    image_ = pupil_image_queue_[image_index_];
    image_.convertTo(image_, CV_8UC1, 1.0 / 256.0);
    return image_;
}

cv::Mat IdsCamera::grabGlintImage() {
    image_ = glint_image_queue_[image_index_];
    image_.convertTo(image_, CV_8UC1, 1.0 / 256.0);
    return image_;
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

    result = is_Exposure(camera_handle_, IS_EXPOSURE_CMD_SET_EXPOSURE,
                         (void *) &exposure, sizeof(exposure));
    assert(result == IS_SUCCESS);
}

void IdsCamera::setGamma(float gamma) {
    int result;

    int scaled_gamma = static_cast<int>(gamma * 100);
    result = is_Gamma(camera_handle_, IS_GAMMA_CMD_SET, (void *) &scaled_gamma,
                      sizeof(scaled_gamma));
    assert(result == IS_SUCCESS);
}

void IdsCamera::setFramerate(double framerate) {
    int result;

    result = is_SetFrameRate(camera_handle_, framerate, &framerate);
    assert(result == IS_SUCCESS);
}
}// namespace et