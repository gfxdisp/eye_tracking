#include "ImageProvider.hpp"

namespace et {
double ImageProvider::getPixelPitch() {
    return pixel_pitch_;
}

[[maybe_unused]] void ImageProvider::setExposure(double exposure) {
}

void ImageProvider::setGamma(float gamma) {
}

void ImageProvider::setFramerate(double framerate) {
}
}// namespace et