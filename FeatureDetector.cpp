#include <opencv2/cudaimgproc.hpp>
#include <utility>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/cudaarithm.hpp>
#include "FeatureDetector.h"

EyeTracking::FeatureDetector::FeatureDetector(ImageProperties imageProperties, char const *templateFilename)
        : imageProperties(std::move(imageProperties)) {
    spots.upload(cv::imread(templateFilename, cv::IMREAD_GRAYSCALE));
    spotsMatcher = cv::cuda::createTemplateMatching(CV_8UC1, cv::TM_CCOEFF_NORMED);

    int mapWidth = (imageProperties.ROI.width - spots.cols + 1) * 2;
    int mapHeight = (imageProperties.ROI.height - spots.rows + 1) * 2;
    cv::Mat weightMapDistanceCPU = cv::Mat(mapHeight, mapWidth, CV_32F);
    cv::Mat weightMapVerticalCPU = cv::Mat(mapHeight, mapWidth, CV_32F);
    float maxDistance = sqrt((mapWidth / 2) * (mapWidth / 2) + (mapHeight / 2) * (mapHeight / 2));
    float maxVertical = pow(mapWidth / 2, 2);
    for (int i = 0; i < mapHeight; i++) {
        for (int j = 0; j < mapWidth; j++) {
            float distance = sqrt((mapWidth / 2 - j) * (mapWidth / 2 - j) + (mapHeight / 2 - i) * (mapHeight / 2 - i));
            float vertical = pow(mapWidth / 2 - j, 2);
            weightMapDistanceCPU.at<float>(i, j) = 1.0f * (1 - distance / maxDistance);
            weightMapVerticalCPU.at<float>(i, j) =
                    0.33f * (1 - distance / maxDistance) + 0.67f * (1 - vertical / maxVertical);
        }
    }

    weightMapDistance.upload(weightMapDistanceCPU);
    weightMapVertical.upload(weightMapVerticalCPU);
    croppedWeightMap = cv::cuda::GpuMat(mapHeight / 2, mapWidth / 2, CV_32F);
    croppedWeightMap.setTo(1);
}

void EyeTracking::FeatureDetector::extractFrameAndWrite(cv::Mat &input, cv::VideoWriter &outputVideo,
                                                        cv::cuda::GpuMat &croppedFrame, cv::cuda::GpuMat &fullFrame) {
    fullFrame.upload(input);
    if (input.type() == CV_8UC3) cv::cuda::cvtColor(fullFrame, fullFrame, cv::COLOR_BGR2GRAY);
    fullFrame.download(input);
    outputVideo.write(input);
    croppedFrame = fullFrame(imageProperties.ROI);
}

void EyeTracking::FeatureDetector::extractFrame(cv::Mat &input, cv::cuda::GpuMat &croppedFrame,
                                                cv::cuda::GpuMat &fullFrame) {
    fullFrame.upload(input);
    if (input.type() == CV_8UC3) cv::cuda::cvtColor(fullFrame, fullFrame, cv::COLOR_BGR2GRAY);
    croppedFrame = fullFrame(imageProperties.ROI);
}

void EyeTracking::FeatureDetector::createBorder(cv::Mat &input, cv::Size2i resolution) {
    static bool first = true;
    static int border_v = 0, border_h = 0;
    if (first) {
        first = false;
        float a = resolution.height;
        float b = resolution.width;
        float c = input.rows;
        float d = input.cols;
        if (a / b >= c / d) {
            border_v = (int) ((((a / b) * d) - c) / 2);
        } else {
            border_h = (int) ((((a / b) * c) - d) / 2);
        }
    }
    cv::copyMakeBorder(input, input, border_v, border_v, border_h, border_h, cv::BORDER_CONSTANT, 0);
    cv::resize(input, input, resolution);
}

bool EyeTracking::FeatureDetector::findReflection(cv::cuda::GpuMat &frame, cv::Point &location) {
    spotsMatcher->match(frame, spots, correlationMap, streamSpots);

    cv::cuda::add(correlationMap, croppedWeightMap, multipliedMap);
    double maxVal;
    cv::cuda::minMaxLoc(multipliedMap, nullptr, &maxVal, nullptr, &location);

    bool reflectionFound = maxVal > imageProperties.templateMatchingThreshold;

    if (location.y > 0 and location.x > 0 and reflectionFound) {
        cv::Rect glint = cv::Rect(location.x, location.y, spots.cols, spots.rows);
        frame(glint).setTo(0);
        location += imageProperties.ROI.tl();
        location.x += spots.cols / 2;
        location.y += spots.rows / 2;
    }

    return reflectionFound;
}

void EyeTracking::FeatureDetector::setMapToVertical(cv::Point point) {
    point.x = imageProperties.ROI.width - point.x + imageProperties.ROI.tl().x;
    point.y = imageProperties.ROI.height - point.y + imageProperties.ROI.tl().y;

    point.x = std::clamp(point.x, 0, imageProperties.ROI.width - spots.cols + 1);
    point.y = std::clamp(point.y, 0, imageProperties.ROI.height - spots.rows + 1);


    cv::Rect croppedRect = cv::Rect(point.x, point.y, imageProperties.ROI.width - spots.cols + 1,
                                    imageProperties.ROI.height - spots.rows + 1);
    croppedWeightMap = weightMapVertical(croppedRect);
}

void EyeTracking::FeatureDetector::setMapToDistance(cv::Point point1, cv::Point point2) {
    cv::Point point;

    point.x = (point1.x + point2.x) / 2 - imageProperties.ROI.tl().x;
    point.y = (point1.y + point2.y) / 2 - imageProperties.ROI.tl().y;

    point.x = imageProperties.ROI.width - point.x;
    point.y = imageProperties.ROI.height - point.y;

    point.x = std::clamp(point.x, 0, imageProperties.ROI.width - spots.cols + 1);
    point.y = std::clamp(point.y, 0, imageProperties.ROI.height - spots.rows + 1);

    cv::Rect croppedRect = cv::Rect(point.x, point.y, imageProperties.ROI.width - spots.cols + 1,
                                    imageProperties.ROI.height - spots.rows + 1);
    croppedWeightMap = weightMapDistance(croppedRect);
}


