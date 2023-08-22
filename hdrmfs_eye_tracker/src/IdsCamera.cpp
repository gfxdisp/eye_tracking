#include "IdsCamera.hpp"
#include "Settings.hpp"

#include <cassert>
#include <chrono>
#include <iostream>
#include <thread>

namespace et {

void IdsCamera::initialize() {
    initializeCamera();
    image_gatherer_ = std::thread{&IdsCamera::imageGatheringThread, this};
}

void IdsCamera::initializeCamera() {
    int all_camera_count;

    is_GetNumberOfCameras(&all_camera_count);

    auto camera_list = reinterpret_cast<UEYE_CAMERA_LIST *>(
        new BYTE[sizeof(DWORD) + all_camera_count * sizeof(UEYE_CAMERA_LIST)]);
    is_GetCameraList(camera_list);

    std::vector<int> camera_indices{};

    // Lists all cameras and adds those that are observing left and right eyes.
    for (int i = 0; i < all_camera_count; i++) {
        std::clog << "Connected camera: "
                  << std::string(camera_list->uci[i].Model) << ", "
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

    for (int i = 0; i < camera_ids_.size(); i++) {
        int camera_id = camera_ids_[i];
        camera_handles_[camera_id] =
            camera_list->uci[camera_indices[i]].dwCameraID;

        is_InitCamera(&camera_handles_[camera_id], nullptr);

        is_SetColorMode(camera_handles_[camera_id], IS_CM_SENSOR_RAW8);

        double enable_auto_gain = 0.0;
        is_SetAutoParameter(camera_handles_[camera_id], IS_SET_ENABLE_AUTO_GAIN,
                            &enable_auto_gain, nullptr);

        IS_RECT area_of_interest{};
        area_of_interest.s32X =
            Settings::parameters.camera_params[camera_id].capture_offset.width;
        area_of_interest.s32Y =
            Settings::parameters.camera_params[camera_id].capture_offset.height;
        area_of_interest.s32Width =
            Settings::parameters.camera_params[camera_id]
                .region_of_interest.width;
        area_of_interest.s32Height =
            Settings::parameters.camera_params[camera_id]
                .region_of_interest.height;

        is_AOI(camera_handles_[camera_id], IS_AOI_IMAGE_SET_AOI,
               &area_of_interest, sizeof(area_of_interest));

        auto pixel_clock =
            (uint32_t) et::Settings::parameters.camera_params[camera_id]
                .pixel_clock;
        is_PixelClock(camera_handles_[camera_id], IS_PIXELCLOCK_CMD_SET,
                      &pixel_clock, sizeof(pixel_clock));

        setGamma(et::Settings::parameters.camera_params[camera_id].gamma,
                 camera_id);
        setFramerate(
            et::Settings::parameters.camera_params[camera_id].framerate,
            camera_id);
        setExposure(et::Settings::parameters.user_params[camera_id]->exposure,
                    camera_id);

        is_AllocImageMem(camera_handles_[camera_id], area_of_interest.s32Width,
                         area_of_interest.s32Height, CHAR_BIT * 1,
                         &image_handles_[camera_id], &image_ids_[camera_id]);

        is_SetImageMem(camera_handles_[camera_id], image_handles_[camera_id],
                       image_ids_[camera_id]);

        pupil_image_.create(area_of_interest.s32Height, area_of_interest.s32Width,
                      CV_8UC1);
        for (auto &j : image_queues_) {
            j[camera_id].create(area_of_interest.s32Height,
                                area_of_interest.s32Width, CV_8UC1);
        }
    }

    delete[] camera_list;
}

void IdsCamera::imageGatheringThread() {
    int new_image_index{image_index_};

    // Saves images to the looping buffer.
    while (thread_running_) {
        new_image_index = (new_image_index + 1) % IMAGE_IN_QUEUE_COUNT;

        for (int camera_id : camera_ids_) {
            setExposure(
                et::Settings::parameters.user_params[camera_id]->exposure,
                camera_id);
            is_FreezeVideo(camera_handles_[camera_id], IS_WAIT);
            is_CopyImageMem(
                camera_handles_[camera_id], image_handles_[camera_id],
                image_ids_[camera_id],
                (char *) image_queues_[new_image_index][camera_id].data);
        }
        image_index_ = new_image_index;
    }
}

ImageToProcess IdsCamera::grabImage(int camera_id) {
    pupil_image_ = image_queues_[image_index_][camera_id];
    glints_image_ = pupil_image_;
    return {pupil_image_, glints_image_};
}

void IdsCamera::close() {
    thread_running_ = false;
    image_gatherer_.join();

    for (int camera_id : camera_ids_) {
        is_FreeImageMem(camera_handles_[camera_id], image_handles_[camera_id],
                        image_ids_[camera_id]);
        is_ExitCamera(camera_handles_[camera_id]);
    }
}

void IdsCamera::setExposure(double exposure, int camera_id) {
    is_Exposure(camera_handles_[camera_id], IS_EXPOSURE_CMD_SET_EXPOSURE,
                (void *) &exposure, sizeof(exposure));
}

void IdsCamera::setGamma(float gamma, int camera_id) {
    int scaled_gamma = static_cast<int>(gamma * 100);
    is_Gamma(camera_handles_[camera_id], IS_GAMMA_CMD_SET,
             (void *) &scaled_gamma, sizeof(scaled_gamma));
}

void IdsCamera::setFramerate(double framerate, int camera_id) {
    is_SetFrameRate(camera_handles_[camera_id], framerate, &framerate_);
    if (framerate_ != framerate) {
        std::cout << "Requested " << framerate << " FPS, set to " << framerate_
                  << " FPS.\n";
    }
}

} // namespace et