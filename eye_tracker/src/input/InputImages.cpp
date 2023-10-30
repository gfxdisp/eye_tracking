#include "eye_tracker/input/InputImages.hpp"

#include <iostream>

namespace et
{
    InputImages::InputImages(const std::string &images_folder_path)
    {
        cv::glob(images_folder_path + "/images/*_lights_off.jpg", pupil_paths_, false);
        cv::glob(images_folder_path + "/images/*_lights_on.jpg", glints_paths_, false);
    }

    EyeImage InputImages::grabImage()
    {
        if (image_count_ >= pupil_paths_.size() || image_count_ >= glints_paths_.size())
        {
            // Once all images have been processed, empty image is returned.
            return {};
        }

        // Extract number from the filename.
        std::string pupil_path = pupil_paths_[image_count_];
        std::string glints_path = glints_paths_[image_count_];

        // Get just the filenames without directories
        std::string pupil_filename = pupil_path.substr(pupil_path.find_last_of('/') + 1);
        std::string glints_filename = glints_path.substr(glints_path.find_last_of('/') + 1);

        // Number is between "image_" and "_lights"
        int pupil_num = std::stoi(pupil_filename.substr(pupil_filename.find("image_") + 6,
                                                        pupil_filename.find("_lights") - pupil_filename.find("image_") -
                                                        6));
        int glints_num = std::stoi(glints_filename.substr(glints_filename.find("image_") + 6,
                                                          glints_filename.find("_lights") -
                                                          glints_filename.find("image_") - 6));

        if (pupil_num != glints_num)
        {
            std::cerr << "Error: pupil and glints images do not match" << std::endl;
            return {};
        }

        pupil_image_ = cv::imread(pupil_paths_[image_count_]);
        cv::cvtColor(pupil_image_, pupil_image_, cv::COLOR_BGR2GRAY);
        glints_image_ = cv::imread(glints_paths_[image_count_]);
        cv::cvtColor(glints_image_, glints_image_, cv::COLOR_BGR2GRAY);

        image_count_++;
        return {pupil_image_, glints_image_, pupil_num};
    }

    void InputImages::close()
    {
    }

} // namespace et