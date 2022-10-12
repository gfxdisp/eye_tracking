#ifndef HDRMFS_EYE_TRACKER_INPUT_IMAGES_HPP
#define HDRMFS_EYE_TRACKER_INPUT_IMAGES_HPP

#include "ImageProvider.hpp"

#include <string>
#include <vector>

namespace et {
/**
 * Load all images from a given folder and uses them as a video feed.
 */
class InputImages : public ImageProvider {
public:
    /**
     * Loads a folder of images to serve as a video feed.
     * @param images_folder_path Path to a folder containing the images.
     */
    explicit InputImages(const std::string &images_folder_path);
    /**
     * Does nothing.
     */
    void initialize() override;
    /**
     * Does nothing.
     */
    void close() override;
    /**
     * Retrieves the next image in the folder and converts it to grayscale.
     * @param camera_id An id of the camera for which the value is returned.
     * @return Matrix with an image from the folder or an empty image if all
     * images have already been processed.
     */
    cv::Mat grabImage(int camera_id) override;

private:
    // List of all names of all images in the folder.
    std::vector<std::string> filenames_{};
    // Number of already processed images.
    int image_count_{0};
};

} // namespace et

#endif //HDRMFS_EYE_TRACKER_INPUT_IMAGES_HPP
