#include "lice/map_distance_field.h"
#include "types.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include "lice/math_utils.h"


struct SubmapBundleAdjustmentOptions
{
    double voxel_size = 0.3;
    double loop_loss_scale_pos = 1.0;
    double loop_loss_scale_rot = 1.0 * M_PI / 180.0;
    double odom_pos_std = 0.2;
    double odom_rot_std = 0.3 * M_PI / 180.0;
    double loop_pos_std = 0.2;
    double loop_rot_std = 1.0 * M_PI / 180.0;
    bool ram_efficient = false;
};


class SubmapBundleAdjustment
{
    public:
        SubmapBundleAdjustment(const std::string& data_folder, const SubmapBundleAdjustmentOptions& options); 

        void refineLoopTransforms();

        void poseGraphOptimization();

        void bundleAdjustmentOptimization();

    private:
        void loadSubmapPoses(const std::string& filename);
        void loadLoopClosures(const std::string& filename);
        void writeOptimizedPoses(const std::string& filename) const;

        double readResolutions(const std::string& filename);

        std::string loop_folder_;
        std::string pcd_folder_;
        SubmapBundleAdjustmentOptions options_;

        std::vector<Vec7> submap_poses_;
        std::vector<Mat4> odom_transforms_;
        std::vector<std::pair<int, int> > loop_closures_;
        std::vector<Mat4> loop_transforms_;

};















struct RelativePoseCostFunctor
{
    RelativePoseCostFunctor(const Mat4& measured_rel_pose, const Mat6& cov)
        : measured_rel_pose_(measured_rel_pose)
    {
        // LLt decomposition for weight matrix
        Eigen::LLT<Mat6> llt(cov.inverse());
        weight_ = llt.matrixL();
    }

    template <typename T>
    bool operator()(const T* const pose_i, const T* const pose_j, T* residuals) const
    {
        // Convert poses to transformation matrices
        Eigen::Matrix<T, 4, 4> Ti = posQuatToTransform(pose_i);
        Eigen::Matrix<T, 4, 4> Tj = posQuatToTransform(pose_j);
        Eigen::Matrix<T, 4, 4> Tij_measured = measured_rel_pose_.template cast<T>();
        // Compute the predicted relative transformation
        Eigen::Matrix<T, 4, 4> Tij_predicted = Ti.inverse() * Tj;
        // Compute the error transformation
        Eigen::Matrix<T, 4, 4> Te = Tij_measured.inverse() * Tij_predicted;
        // Convert the error transformation to a 6D vector (3 for rotation, 3 for translation)
        Eigen::Matrix<T, 6, 1> error = transformToPoseVector(Te);
        // Scale the residuals by the weight
        Eigen::Map<Eigen::Matrix<T, 6, 1>> residual_map(residuals);
        residual_map = weight_.template cast<T>() * error;
        return true;
    }

    Mat4 measured_rel_pose_;
    Mat6 weight_;
};

struct RelativePositionCostFunctor
{
    RelativePositionCostFunctor(const Vec3& measured_rel_pos, const Mat3& cov)
        : measured_rel_pos_(measured_rel_pos)
    {
        // LLt decomposition for weight matrix
        Eigen::LLT<Mat3> llt(cov.inverse());
        weight_ = llt.matrixL();
    }

    template <typename T>
    bool operator()(const T* const pose_i, const T* const pose_j, T* residuals) const
    {
        // Convert poses to transformation matrices
        Eigen::Matrix<T, 4, 4> Ti = posQuatToTransform(pose_i);
        Eigen::Matrix<T, 4, 4> Tj = posQuatToTransform(pose_j);
        // Compute the predicted relative transformation
        Eigen::Matrix<T, 4, 4> Tij_predicted = Ti.inverse() * Tj;
        // Compute the error transformation
        Eigen::Map<Eigen::Matrix<T, 3, 1>> residual_map(residuals);
        residual_map = weight_.template cast<T>() * (Tij_predicted.template block<3,1>(0,3) - measured_rel_pos_.template cast<T>());
        return true;
    }

    Vec3 measured_rel_pos_;
    Mat3 weight_;
};

struct RelativeRotationCostFunctor
{
    RelativeRotationCostFunctor(const Mat3& measured_rel_rot, const Mat3& cov)
        : measured_rel_rot_inv_(measured_rel_rot.inverse())
    {
        // LLt decomposition for weight matrix
        Eigen::LLT<Mat3> llt(cov.inverse());
        weight_ = llt.matrixL();
    }

    template <typename T>
    bool operator()(const T* const pose_i, const T* const pose_j, T* residuals) const
    {
        // Convert poses to transformation matrices
        Eigen::Matrix<T, 4, 4> Ti = posQuatToTransform(pose_i);
        Eigen::Matrix<T, 4, 4> Tj = posQuatToTransform(pose_j);
        // Compute the predicted relative transformation
        Eigen::Matrix<T, 4, 4> Tij_predicted = Ti.inverse() * Tj;
        // Compute the error transformation
        Eigen::Map<Eigen::Matrix<T, 3, 1>> residual_map(residuals);
        Eigen::Matrix<T, 3, 3> rot_diff = measured_rel_rot_inv_.template cast<T>() * Tij_predicted.template block<3,3>(0,0);
        residual_map = weight_.template cast<T>() * rotMatToAngleAxis(rot_diff);
        return true;
    }

    Mat3 measured_rel_rot_inv_;
    Mat3 weight_;
};