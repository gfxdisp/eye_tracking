#include "eye_tracker/input/IdsCamera.hpp"

#include <cassert>
#include <chrono>
#include <iostream>
#include <thread>


namespace et
{

    IdsCamera::IdsCamera(int camera_id)
    {
        auto path = (Settings::settings_folder_ / ("fake_" + std::to_string(camera_id) + ".png"));
        fake_image_ = cv::imread(path.string(), cv::IMREAD_GRAYSCALE);

        camera_params_ = &Settings::parameters.camera_params[camera_id];
        user_params_ = Settings::parameters.user_params[camera_id];

        int all_camera_count;

        is_GetNumberOfCameras(&all_camera_count);

        auto camera_list = reinterpret_cast<UEYE_CAMERA_LIST *>(
                new BYTE[sizeof(DWORD) + all_camera_count * sizeof(UEYE_CAMERA_LIST)]);
        is_GetCameraList(camera_list);

        int camera_index = -1;

        // Lists all cameras and adds those that are observing left and right eyes.
        for (int i = 0; i < all_camera_count; i++)
        {
            std::clog << "Connected camera: " << std::string(camera_list->uci[i].Model) << ", "
                      << std::string(camera_list->uci[i].SerNo) << "\n";
            if (std::string(camera_list->uci[i].SerNo) == camera_params_->serial_number)
            {
                camera_index = i;
                break;
            }
        }

        if (camera_index == -1)
        {
            fake_camera_ = true;
        }
        else
        {
            fake_camera_ = false;
            camera_handle_ = camera_list->uci[camera_index].dwCameraID;

            is_InitCamera(&camera_handle_, nullptr);

            is_SetColorMode(camera_handle_, IS_CM_SENSOR_RAW8);

            double enable_auto_gain = 0.0;
            is_SetAutoParameter(camera_handle_, IS_SET_ENABLE_AUTO_GAIN, &enable_auto_gain, nullptr);

            IS_RECT area_of_interest{};
            area_of_interest.s32X = camera_params_->capture_offset.width;
            area_of_interest.s32Y = camera_params_->capture_offset.height;
            area_of_interest.s32Width = camera_params_->region_of_interest.width;
            area_of_interest.s32Height = camera_params_->region_of_interest.height;

            is_AOI(camera_handle_, IS_AOI_IMAGE_SET_AOI, &area_of_interest, sizeof(area_of_interest));

            auto pixel_clock = (uint32_t) camera_params_->pixel_clock;
            is_PixelClock(camera_handle_, IS_PIXELCLOCK_CMD_SET, &pixel_clock, sizeof(pixel_clock));

            setGamma(camera_params_->gamma);
            setFramerate(camera_params_->framerate);
            setExposure(user_params_->exposure);

            is_AllocImageMem(camera_handle_, area_of_interest.s32Width, area_of_interest.s32Height, CHAR_BIT * 1,
                             &image_handle_, &image_id_);

            is_SetImageMem(camera_handle_, image_handle_, image_id_);

            pupil_image_.create(area_of_interest.s32Height, area_of_interest.s32Width, CV_8UC1);
            for (auto &j: image_queue_)
            {
                j.create(area_of_interest.s32Height, area_of_interest.s32Width, CV_8UC1);
            }
        }


        delete[] camera_list;
        image_gatherer_ = std::thread{&IdsCamera::imageGatheringThread, this};
    }

    void IdsCamera::imageGatheringThread()
    {
        int new_image_index{image_index_};

        // Saves images to the looping buffer.
        while (thread_running_)
        {
            new_image_index = (new_image_index + 1) % IMAGE_IN_QUEUE_COUNT;

            if (fake_camera_)
            {
                fake_image_.copyTo(image_queue_[new_image_index]);
                usleep(10 * 1000);
            }
            else
            {
                setExposure(user_params_->exposure);
                is_FreezeVideo(camera_handle_, IS_WAIT);
                is_CopyImageMem(camera_handle_, image_handle_, image_id_, (char *) image_queue_[new_image_index].data);
            }
            image_index_ = new_image_index;
        }
    }

    EyeImage IdsCamera::grabImage()
    {
        pupil_image_ = image_queue_[image_index_];
        glints_image_ = pupil_image_;
        return {pupil_image_, glints_image_, image_counter_++};
    }

    void IdsCamera::close()
    {
        thread_running_ = false;
        image_gatherer_.join();
        is_FreeImageMem(camera_handle_, image_handle_, image_id_);
        is_ExitCamera(camera_handle_);
    }

    void IdsCamera::setExposure(double exposure)
    {
        is_Exposure(camera_handle_, IS_EXPOSURE_CMD_SET_EXPOSURE, (void *) &exposure, sizeof(exposure));
    }

    void IdsCamera::setGamma(double gamma)
    {
        int scaled_gamma = static_cast<int>(gamma * 100);
        is_Gamma(camera_handle_, IS_GAMMA_CMD_SET, (void *) &scaled_gamma, sizeof(scaled_gamma));
    }

    void IdsCamera::setFramerate(double framerate)
    {
        is_SetFrameRate(camera_handle_, framerate, &framerate_);
        if (framerate_ != framerate)
        {
            std::cout << "Requested " << framerate << " FPS, set to " << framerate_ << " FPS.\n";
        }
    }

} // namespace et