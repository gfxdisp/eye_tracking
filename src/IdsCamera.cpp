#include "IdsCamera.hpp"
#include "Settings.hpp"

#include <cassert>
#include <chrono>
#include <iostream>
#include <thread>

namespace et {

void IdsCamera::initialize(bool separate_exposures) {
    initializeCamera();
    separate_exposures_ = separate_exposures;
    if (separate_exposures_) {
        image_gatherer_ =
            std::thread{&IdsCamera::imageGatheringTwoExposuresThread, this};
    } else {
        image_gatherer_ =
            std::thread{&IdsCamera::imageGatheringOneExposureThread, this};
    }
}

void IdsCamera::initializeCamera() {
    int result;
    int all_camera_count;

    result = is_GetNumberOfCameras(&all_camera_count);
    assert(result == IS_SUCCESS);
    assert(all_camera_count > 0);

    UEYE_CAMERA_LIST* camera_list = (UEYE_CAMERA_LIST*) new BYTE[sizeof (DWORD) + all_camera_count * sizeof (UEYE_CAMERA_LIST)];
    result = is_GetCameraList(camera_list);
    assert(result == IS_SUCCESS);

    std::vector<int> camera_indices{};

    for (int i = 0; i < all_camera_count; i++) {
        std::clog << "Connected camera: " << std::string(camera_list->uci[i].Model) << ", "
                  << std::string(camera_list->uci[i].SerNo) << "\n";
        for (int j = 0; j < 2; j++) {
            if (std::string(camera_list->uci[i].SerNo)
                == Settings::parameters.camera_params[j].serial_number) {
                camera_indices.push_back(i);
                camera_ids_.push_back(j);
            }
        }
    }
    assert(!camera_indices.empty());
    used_camera_count_ = static_cast<int>(camera_indices.size());

    for (int i = 0; i < camera_ids_.size(); i++) {
        int camera_id = camera_ids_[i];
        camera_handles_[camera_id] =
            camera_list->uci[camera_indices[i]].dwCameraID;

        result = is_InitCamera(&camera_handles_[camera_id], nullptr);
        assert(result == IS_SUCCESS);

        result = is_SetColorMode(camera_handles_[camera_id], IS_CM_MONO8);
        assert(result == IS_SUCCESS);

        double enable_auto_gain = 0.0;
        result = is_SetAutoParameter(camera_handles_[camera_id],
                                     IS_SET_ENABLE_AUTO_GAIN, &enable_auto_gain,
                                     nullptr);
        assert(result == IS_SUCCESS);

        IS_RECT area_of_interest{};
        area_of_interest.s32X =
            Settings::parameters.camera_params[camera_id].capture_offset.width;
        area_of_interest.s32Y =
            Settings::parameters.camera_params[camera_id].capture_offset.height;
        area_of_interest.s32Width =
            Settings::parameters.camera_params[camera_id].region_of_interest.width;
        area_of_interest.s32Height =
            Settings::parameters.camera_params[camera_id].region_of_interest.height;

        result = is_AOI(camera_handles_[camera_id], IS_AOI_IMAGE_SET_AOI,
                        &area_of_interest, sizeof(area_of_interest));
        assert(result == IS_SUCCESS);


        uint32_t pixel_clock = (uint32_t) et::Settings::parameters.camera_params[camera_id].pixel_clock;
        result = is_PixelClock(camera_handles_[camera_id], IS_PIXELCLOCK_CMD_SET, &pixel_clock, sizeof(pixel_clock));
        assert(result == IS_SUCCESS);

        setGamma(et::Settings::parameters.camera_params[camera_id].gamma, camera_id);
        setFramerate(et::Settings::parameters.camera_params[camera_id].framerate, camera_id);
        setExposure(et::Settings::parameters.camera_params[camera_id].pupil_exposure, camera_id);

        result = is_AllocImageMem(camera_handles_[camera_id], area_of_interest.s32Width,
                                  area_of_interest.s32Height,
                                  sizeof(char) * CHAR_BIT * 1,
                                  &image_handles_[camera_id], &image_ids_[camera_id]);
        assert(result == IS_SUCCESS);

        result = is_SetImageMem(camera_handles_[camera_id], image_handles_[camera_id],
                                image_ids_[camera_id]);
        assert(result == IS_SUCCESS);

        image_.create(area_of_interest.s32Height, area_of_interest.s32Width,
                      CV_8UC1);
        for (auto &j : pupil_image_queues_) {
            j[camera_id].create(area_of_interest.s32Height, area_of_interest.s32Width,
                        CV_8UC1);
        }

        if (separate_exposures_) {
            for (auto &j : glint_image_queues_) {
                j[camera_id].create(area_of_interest.s32Height,
                            area_of_interest.s32Width, CV_8UC1);
            }
        }
    }

    delete[] camera_list;
}

void IdsCamera::imageGatheringTwoExposuresThread() {
    int result;
    int new_image_index{image_index_};
    while (thread_running_) {
        new_image_index = (new_image_index + 1) % IMAGE_IN_QUEUE_COUNT;

        for (int camera_id : camera_ids_) {
            setExposure(
                et::Settings::parameters.camera_params[camera_id].pupil_exposure, camera_id);
            result = is_FreezeVideo(camera_handles_[camera_id], IS_WAIT);
            result = is_CopyImageMem(
                camera_handles_[camera_id], image_handles_[camera_id], image_ids_[camera_id],
                (char *) pupil_image_queues_[new_image_index][camera_id].data);

            setExposure(
                et::Settings::parameters.camera_params[camera_id].glint_exposure, camera_id);
            result = is_FreezeVideo(camera_handles_[camera_id], IS_WAIT);
            result = is_CopyImageMem(
                camera_handles_[camera_id], image_handles_[camera_id], image_ids_[camera_id],
                (char *) glint_image_queues_[new_image_index][camera_id].data);
        }

        image_index_ = new_image_index;
    }
}

void IdsCamera::imageGatheringOneExposureThread() {
    int result;
    int new_image_index{image_index_};

    while (thread_running_) {
        new_image_index = (new_image_index + 1) % IMAGE_IN_QUEUE_COUNT;

        for (int camera_id : camera_ids_) {
            setExposure(et::Settings::parameters.camera_params[camera_id].pupil_exposure,
                    camera_id);
            result = is_FreezeVideo(camera_handles_[camera_id], IS_WAIT);
            result = is_CopyImageMem(
                camera_handles_[camera_id], image_handles_[camera_id], image_ids_[camera_id],
                (char *) pupil_image_queues_[new_image_index][camera_id].data);
        }
        image_index_ = new_image_index;
    }
}

cv::Mat IdsCamera::grabPupilImage(int camera_id) {
    image_ = pupil_image_queues_[image_index_][camera_id];
    // image_.convertTo(image_, CV_8UC1, 1.0 / 256.0);
    return image_;
}

cv::Mat IdsCamera::grabGlintImage(int camera_id) {
    image_ = glint_image_queues_[image_index_][camera_id];
    // image_.convertTo(image_, CV_8UC1, 1.0 / 256.0);
    return image_;
}

void IdsCamera::close() {
    int result;

    thread_running_ = false;
    image_gatherer_.join();

    for (int camera_id : camera_ids_) {
        result = is_FreeImageMem(camera_handles_[camera_id],
                                 image_handles_[camera_id],
                                 image_ids_[camera_id]);
        assert(result == IS_SUCCESS);

        result = is_ExitCamera(camera_handles_[camera_id]);
        assert(result == IS_SUCCESS);
    }
}

void IdsCamera::setExposure(double exposure, int camera_id) {
    int result;

    result =
        is_Exposure(camera_handles_[camera_id], IS_EXPOSURE_CMD_SET_EXPOSURE,
                    (void *) &exposure, sizeof(exposure));

    // std::this_thread::sleep_for(
    //      std::chrono::microseconds((int) (0.5f / framerate_ * 1e6)));
    assert(result == IS_SUCCESS);
}

void IdsCamera::setGamma(float gamma, int camera_id) {
    int result;

    int scaled_gamma = static_cast<int>(gamma * 100);
    result = is_Gamma(camera_handles_[camera_id], IS_GAMMA_CMD_SET,
                      (void *) &scaled_gamma, sizeof(scaled_gamma));
    assert(result == IS_SUCCESS);
}

void IdsCamera::setFramerate(double framerate, int camera_id) {
    int result;

    result =
        is_SetFrameRate(camera_handles_[camera_id], framerate, &framerate_);
    assert(result == IS_SUCCESS);
    if (framerate_ != framerate) {
        std::cout << "Requested " << framerate << " FPS, set to " << framerate_
                  << " FPS.\n";
    }
}

std::vector<int> IdsCamera::getCameraIds() {
    return camera_ids_;
}

} // namespace et