#include "ImageProvider.hpp"

namespace et {

void ImageProvider::initialize(bool separate_exposures) {
    separate_exposures = separate_exposures_;
}

void ImageProvider::close() {
}

cv::Mat ImageProvider::grabImage(int camera_id) {
    return grabPupilImage(camera_id);
}

bool ImageProvider::isSeparateExposures() {
    return separate_exposures_;
}
} // namespace et