#ifndef EYE_TRACKER_METAMODEL_HPP
#define EYE_TRACKER_METAMODEL_HPP

#include "eye_tracker/image/FeatureAnalyser.hpp"
#include "eye_tracker/optimizers/VisualAnglesOptimizer.hpp"
#include "eye_tracker/optimizers/PolynomialFit.hpp"
#include "eye_tracker/optimizers/OpticalAxisOptimizer.hpp"
#include "eye_tracker/optimizers/EyeAnglesOptimizer.hpp"

#include <memory>

namespace et
{

    class MetaModel
    {
    public:
        MetaModel(int camera_id);

        cv::Point3d
        findMetaModel(std::shared_ptr<ImageProvider> image_provider, std::shared_ptr<FeatureAnalyser> feature_analyser,
                      std::string path_to_csv, std::string user_id);

    private:
        VisualAnglesOptimizer *visual_angles_optimizer_{};
        cv::Ptr<cv::DownhillSolver::Function> visual_angles_minimizer_function_{};
        cv::Ptr<cv::DownhillSolver> visual_angles_solver_{};

        OpticalAxisOptimizer *optical_axis_optimizer_{};
        cv::Ptr<cv::DownhillSolver::Function> optical_axis_minimizer_function_{};
        cv::Ptr<cv::DownhillSolver> optical_axis_solver_{};

        EyeAnglesOptimizer *eye_angles_optimizer_{};
        cv::Ptr<cv::DownhillSolver::Function> eye_angles_minimizer_function_{};
        cv::Ptr<cv::DownhillSolver> eye_angles_solver_{};

        EyeCentreOptimizer *eye_centre_optimizer_{};
        cv::Ptr<cv::DownhillSolver::Function> eye_centre_minimizer_function_{};
        cv::Ptr<cv::DownhillSolver> eye_centre_solver_{};


        int camera_id_{};
    };

} // et

#endif //EYE_TRACKER_METAMODEL_HPP
