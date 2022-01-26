#ifndef EYE_TRACKER_FEATUREDETECTOR_H
#define EYE_TRACKER_FEATUREDETECTOR_H

#include <opencv2/core/cuda.hpp>
#include <opencv2/videoio.hpp>
#include "eye_tracking.hpp"

namespace EyeTracking {

    class FeatureDetector {
    public:
        explicit FeatureDetector(ImageProperties imageProperties, char const *templateFilename);

        void extractFrameAndWrite(cv::Mat &input, cv::VideoWriter &outputVideo, cv::cuda::GpuMat &croppedFrame,
                                  cv::cuda::GpuMat &fullFrame);

        void extractFrame(cv::Mat &input, cv::cuda::GpuMat &croppedFrame, cv::cuda::GpuMat &fullFrame);

        void createBorder(cv::Mat &input, cv::Size2i resolution);

        bool findReflection(cv::cuda::GpuMat &frame, cv::Point &location);

        void setMapToVertical(cv::Point point);

        void setMapToDistance(cv::Point point1, cv::Point point2);

    private:
        ImageProperties imageProperties;
        cv::cuda::GpuMat weightMapDistance, weightMapVertical;
        cv::cuda::GpuMat correlationMap;
        cv::Ptr<cv::cuda::TemplateMatching> spotsMatcher;
        cv::cuda::GpuMat spots;
        cv::cuda::Stream streamSpots;
        cv::cuda::GpuMat croppedWeightMap;
        cv::cuda::GpuMat multipliedMap;
    };
}

#endif //EYE_TRACKER_FEATUREDETECTOR_H
