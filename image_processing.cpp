#include "image_processing.hpp"
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/imgproc.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xoperation.hpp>
namespace EyeTracker::ImageProcessing {
    using namespace cv;

    void correct(const cuda::GpuMat& image, cuda::GpuMat& result, float alpha, float beta, float gamma) {
        const static xt::xtensor_fixed<float, xt::xshape<256>> RANGE = xt::arange<float>(0, 255, 1)/255;
        xt::xtensor_fixed<uint8_t, xt::xshape<256>> xLUT = xt::cast<uint8_t>(xt::clip(xt::pow(RANGE, gamma) * alpha * 255.0 + beta, 0, 255));
        Mat mLUT(1, 256, CV_8UC1, xLUT.data());
        Ptr<cuda::LookUpTable> cudaLUT = cuda::createLookUpTable(mLUT);
        cudaLUT->transform(image, result);
    }

    std::vector<PointWithRating> findCircles(const cuda::GpuMat& frame, uint8_t thresh, float min_radius, float max_radius, float max_rating) {
        const static Mat morphologyElement = getStructuringElement(MORPH_ELLIPSE, Size(13, 13));
        const static Ptr<cuda::Filter> morphOpen  = cuda::createMorphologyFilter(MORPH_OPEN,  CV_8UC1, morphologyElement);
        const static Ptr<cuda::Filter> morphClose = cuda::createMorphologyFilter(MORPH_CLOSE, CV_8UC1, morphologyElement);
        cuda::GpuMat thresholded;
        cuda::threshold(frame, thresholded, thresh, 255, THRESH_BINARY_INV);
        morphOpen->apply(thresholded, thresholded);
        morphClose->apply(thresholded, thresholded);
        const Mat thresh_cpu(thresholded);
        std::vector<std::vector<Point>> contours;
        std::vector<Vec4i> hierarchy;
        std::vector<PointWithRating> result;
        findContours(thresh_cpu, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        for (const std::vector<Point>& contour : contours) {
            Point2f centre;
            float radius;
            minEnclosingCircle(contour, centre, radius);
            if (radius < min_radius or radius > max_radius) continue;
            const float contour_area = contourArea(contour);
            if (contour_area <= 0) continue;
            const float circle_area = xt::numeric_constants<float>::PI * std::pow(radius, 2);
            const float rating = xt::square((circle_area-contour_area)/circle_area);
            if (rating <= max_rating) result.push_back({centre, rating});
        }
        return result;
    }
}
