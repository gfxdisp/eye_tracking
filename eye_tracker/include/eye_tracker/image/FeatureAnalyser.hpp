#ifndef HDRMFS_EYE_TRACKER_FEATURE_ANALYSER_HPP
#define HDRMFS_EYE_TRACKER_FEATURE_ANALYSER_HPP

#include "eye_tracker/image/preprocess/ImagePreprocessor.hpp"
#include "eye_tracker/image/temporal_filter/TemporalFilterer.hpp"
#include "eye_tracker/image/position/FeatureEstimator.hpp"

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/opencv.hpp>

#include <mutex>
#include <vector>


namespace et {

/**
 * Detects various eye features on the image directly in the image space.
 */
    class FeatureAnalyser {
    public:


        FeatureAnalyser(int camera_id);

/**
         * Uploads an image to GPU, thresholds it for glints and pupil detection,
         * and saves to CPU.
         * @param image A struct with two images: one for detecting pupil and one for detecting glints.
         */
        void preprocessImage(const EyeImage& image);

        /**
         * Detects a pupil in the image preprocessed using preprocessImage().
         * @return True if the pupil was found. False otherwise.
         */
        bool findPupil();

        /**
         * Detects an ellipse formed from glints in the image preprocessed using
         * preprocessImage().
         * @return True if the glint ellipse was found. False otherwise.
         */
        bool findEllipsePoints();

        /**
         * Retrieves a vector between a specific glint and pupil position
         * based on parameters that were previously calculated in findPupil()
         * and findGlints(). If the ellipse fitting was used through findEllipsePoints(),
         * the vector is calculated between pupil position and ellipse centre.
         * @param pupil_glint_vector Variable that will contain the vector.
         */
        void getPupilGlintVector(cv::Vec2d& pupil_glint_vector);

        /**
         * Retrieves pupil position in image space that was previously
         * calculated in findPupil().
         * @param pupil Variable that will contain the pupil position.
         */
        void getPupilUndistorted(cv::Point2d& pupil);

        /**
         * Retrieves a vector between a specific glint and pupil position
         * based on parameters that were previously calculated in findPupil()
         * and findGlints(). If the ellipse fitting was used through findEllipsePoints(),
         * the vector is calculated between pupil position and ellipse centre.
         * Value is averaged across a number of frames set using
         * the setGazeBufferSize() method.
         * @param pupil_glint_vector Variable that will contain the vector.
         */
        void getPupilGlintVectorFiltered(cv::Vec2d& pupil_glint_vector);

        /**
         * Retrieves pupil position in image space that was previously
         * calculated in findPupil(). Value is averaged across a number of frames
         * set using the setGazeBufferSize() method.
         * @param pupil Variable that will contain the pupil position.
         */
        void getPupilBuffered(cv::Point2d& pupil);

        /**
         * Retrieves pupil position in image space that was previously
         * calculated in findPupil().
         * @return Pupil position.
         */
        cv::Point2d getPupilUndistorted();

        /**
         * Retrieves distorted pupil position in image space that was previously
         * calculated in findPupil().
         * @return Pupil position.
         */
        cv::Point2d getPupilDistorted();

        /**
         * Retrieves a radius of a pupil.
         * @return Pupil radius.
         */
        [[nodiscard]] int getPupilRadiusUndistorted() const;

        /**
         * Retrieves a radius of a distorted pupil.
         * @return Pupil radius.
         */
        [[nodiscard]] int getPupilRadiusDistorted() const;

        /**
         * Retrieves a pointer to the vector of glints detected using findGlints().
         * @return A vector of glints.
         */
        std::vector<cv::Point2d>* getGlints();

        /**
         * Retrieves a pointer to the vector of distorted glints detected using findGlints().
         * @return A vector of glints.
         */
        std::vector<cv::Point2d>* getDistortedGlints();

        std::vector<bool>* getGlintsValidity();

        /**
         * Retrieves an ellipse detected using findEllipsePoints().
         * @return A detected ellipse.
         */
        cv::RotatedRect getEllipseUndistorted();

        void getEllipseUndistorted(cv::RotatedRect& ellipse);

        void getFrameNum(int& frame_num);

        /**
         * Retrieves a distorted ellipse detected using findEllipsePoints().
         * @return A detected ellipse.
         */
        cv::RotatedRect getEllipseDistorted();

        /**
         * Retrieves a thresholded image generated using preprocessImage()
         * used for pupil detection.
         * @return A thresholded image.
         */
        cv::Mat getThresholdedPupilImage();

        /**
         * Retrieves a thresholded image generated using preprocessImage()
         * used for glint detection.
         * @return A thresholded image.
         */
        cv::Mat getThresholdedGlintsImage();

        /**
         * Used to set the number of frames, across which getPupilBuffered() and
         * getPupilGlintVectorFiltered() are calculated.
         * @param value Number of frames.
         */
        void setGazeBufferSize(uint8_t value);

        /**
         * Updates the pupil and pupil-glint vectors based on the newest data
         * from findPupil(), findGlints() and findEllipsePoints() averaged across a number
         * of frames set using setGazeBufferSize().
         */
        void updateGazeBuffer();

        virtual cv::Point2d undistort(cv::Point2d point) = 0;

        virtual cv::Point2d distort(cv::Point2d point) = 0;

    protected:
        int camera_id_{};

        std::shared_ptr<ImagePreprocessor> image_preprocessor_{};
        std::shared_ptr<TemporalFilterer> temporal_filterer_{};
        std::shared_ptr<FeatureEstimator> position_estimator_{};

        // Synchronization variable between feature detection and socket server.
        std::mutex mtx_features_{};

        // Pupil location estimated using findPupil().
        cv::Point2d pupil_location_distorted_{};
        // Undisorted pupil location estimated using findPupil().
        cv::Point2d pupil_location_undistorted_{};
        // Undistorted location of the representative glint estimated using findGlints() or findEllipse().
        cv::Point2d glint_represent_undistorted_{};

        // Radius of the pupil in pixels.
        double pupil_radius_distorted_{0};
        // Radius of the undistorted pupil in pixels.
        double pupil_radius_undistorted_{0};
        // Glint locations estimated using findGlints() or findEllipse().
        std::vector<cv::Point2d> glint_locations_distorted_{};
        // Undistorted glint locations estimated using findGlints() or findEllipse().
        std::vector<cv::Point2d> glint_locations_undistorted_{};

        std::vector<bool> glint_validity_{};

        // Ellipse parameters estimated using findEllipse().
        cv::RotatedRect glint_ellipse_distorted_{};
        // Undisorted ellipse parameters estimated using findEllipse().
        cv::RotatedRect glint_ellipse_undistorted_{};

        // Size of the buffer used to average pupil position and pupil-glint vector
        // across multiple frames.
        int buffer_size_{4};
        // Index in the buffer of the most recent pupil and glint positions.
        int buffer_idx_{0};
        // Number of total pupil and glint positions.
        int buffer_summed_count_{0};
        // Buffer of the most recent pupil positions.
        std::vector<cv::Point2d> pupil_location_buffer_{};
        // Buffer of the most recent glint positions.
        std::vector<cv::Point2d> glint_location_buffer_{};
        // Summed values from the pupil_location_buffer_ vector.
        cv::Point2d pupil_location_summed_{};
        // Summed values from the glint_location_buffer_ vector.
        cv::Point2d glint_location_summed_{};
        // Mean pupil position averaged across whole pupil_location_buffer_ vector.
        cv::Point2d pupil_location_buffered_{};
        // Mean pupil position averaged across whole glint_location_buffer_ vector.
        cv::Point2d glint_location_filtered_{};

        int frame_num_{};

        // pupil_thresholded_image_gpu_ downloaded to CPU.
        cv::Mat pupil_thresholded_image_;
        // glints_thresholded_image_gpu_ downloaded to CPU.
        cv::Mat glints_thresholded_image_;
    };
} // namespace et

#endif //HDRMFS_EYE_TRACKER_FEATURE_ANALYSER_HPP