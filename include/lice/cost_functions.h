#pragma once

#include "lice/types.h"
#include "lice/utils.h"
#include "lice/math_utils.h"
#include <ceres/rotation.h>

#include "lice/state.h"



class ZeroPrior : public ceres::CostFunction
{
    private:
        int nb_dim_;
        double weight_;

    public:
        ZeroPrior(const int nb_dim, const double weight): nb_dim_(nb_dim), weight_(weight)
        {
            set_num_residuals(nb_dim);
            mutable_parameter_block_sizes()->push_back(nb_dim);
        }

        bool Evaluate(const double* const* parameters, double* residuals, double** jacobians) const
        {
            Eigen::Map<const VecX> state(parameters[0], nb_dim_);
            Eigen::Map<VecX> res(residuals, nb_dim_);
            res = weight_ * state;

            if(jacobians != NULL)
            {
                if(jacobians[0] != NULL)
                {
                    Eigen::Map<MatX> jac(jacobians[0], nb_dim_, nb_dim_);
                    jac.setIdentity();
                    jac *= weight_;
                }
            }

            return true;
        }
};

class GaussianPrior : public ceres::CostFunction
{
    private:
        int nb_dim_;
        const VecX mean_;
        MatX weight_;

    public:
        GaussianPrior(int nb_dim, const VecX& mean, const MatX& cov): nb_dim_(nb_dim), mean_(mean)
        {
            set_num_residuals(nb_dim);
            mutable_parameter_block_sizes()->push_back(nb_dim);

            MatX cov_inv = cov.inverse();
            Eigen::LLT<MatX> lltOfA(cov);
            weight_ = lltOfA.matrixL().transpose();
        }

        bool Evaluate(const double* const* parameters, double* residuals, double** jacobians) const
        {
            Eigen::Map<const VecX> state(parameters[0], nb_dim_);
            Eigen::Map<VecX> res(residuals, nb_dim_);
            res = weight_*(state-mean_);

            if(jacobians != NULL)
            {
                if(jacobians[0] != NULL)
                {
                    Eigen::Map<MatX> jac(jacobians[0], nb_dim_, nb_dim_);
                    jac = weight_;
                }
            }

            return true;
        }
};



class LidarNoCalCostFunction: public ceres::SizedCostFunction<1, 3,3,3,3,1>
{
    private:
        const State& state_;
        const DataAssociation data_association_;
        Vec3 source_pt_;
        std::vector<Vec3> target_pts_;
        const std::vector<int> block_sizes_;
        std::vector<double> feature_times_;
        const double weight_ = 1.0;


    public:
        LidarNoCalCostFunction(
                const State& state
                , const DataAssociation& data_association
                , const std::vector<std::shared_ptr<std::vector<Pointd> > >& features
                , const std::vector<std::shared_ptr<std::vector<Pointd> > >& sparse_features
                , const double weight
                , const int64_t& offset_time
                , const Vec7& extrinsic)
                : state_(state)
                , data_association_(data_association)
                , block_sizes_({3,3,3,3,1})
                , weight_(weight)
        {
            feature_times_.resize(1+data_association_.target_ids.size());
            feature_times_[0] = nanosToSeconds(sparse_features[data_association_.pc_id]->at(data_association_.feature_id).t, offset_time);
            for(size_t i = 0; i < data_association_.target_ids.size(); ++i)
            {
                feature_times_[i+1] = nanosToSeconds(features[data_association_.target_ids[i].first]->at(data_association_.target_ids[i].second).t, offset_time);
            }

            Vec4 quat_I_L = extrinsic.head<4>();
            Vec3 pos_I_L = extrinsic.tail<3>();

            source_pt_ = sparse_features[data_association_.pc_id]->at(data_association_.feature_id).vec3();
            ceres::UnitQuaternionRotatePoint(quat_I_L.data(), source_pt_.data(), source_pt_.data());
            source_pt_ += pos_I_L;

            target_pts_.resize(data_association_.target_ids.size());
            for(size_t j = 0; j < data_association_.target_ids.size(); ++j)
            {
                target_pts_[j] = features[data_association_.target_ids[j].first]->at(data_association_.target_ids[j].second).vec3();
                ceres::UnitQuaternionRotatePoint(quat_I_L.data(), target_pts_[j].data(), target_pts_[j].data());
                target_pts_[j] += pos_I_L;
            }
        }

        bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
        {
            Eigen::Map<const Vec3> arg_0(parameters[0]);
            Eigen::Map<const Vec3> arg_1(parameters[1]);
            Eigen::Map<const Vec3> arg_2(parameters[2]);
            Eigen::Map<const Vec3> arg_3(parameters[3]);
            double time_offset = parameters[4][0];

            std::vector<std::pair<Vec3, Vec3> > poses(1+data_association_.target_ids.size());
            std::vector<std::vector<std::pair<MatX, MatX> > > pose_jacobians;
            if(jacobians != NULL)
            {
                pose_jacobians.resize(1+data_association_.target_ids.size());
                for(size_t i = 0; i < feature_times_.size(); ++i)
                {
                    std::tie(poses[i], pose_jacobians[i]) = state_.queryWthJacobian(feature_times_[i], arg_0, arg_1, arg_2, arg_3, time_offset);
                }
            }
            else
            {
                for(size_t i = 0; i < feature_times_.size(); ++i)
                {
                    poses[i] = state_.query(feature_times_[i], arg_0, arg_1, arg_2, arg_3, time_offset);
                }
            }

            Vec3& feature_rot = poses[0].second;
            Vec3& feature_pos = poses[0].first;

            Vec3 feature_W_rot;
            ceres::AngleAxisRotatePoint(feature_rot.data(), source_pt_.data(), feature_W_rot.data());
            Vec3 feature_W = feature_W_rot + feature_pos;

            std::vector<Vec3> target_rots(data_association_.target_ids.size());
            std::vector<Vec3> targets_W_rot(data_association_.target_ids.size());
            std::vector<Vec3> targets_W(data_association_.target_ids.size());
            for(size_t j = 0; j < data_association_.target_ids.size(); ++j)
            {
                Vec3& target_rot = poses[j+1].second;
                Vec3& target_pos = poses[j+1].first;

                Vec3 target_W;
                ceres::AngleAxisRotatePoint(target_rot.data(), target_pts_[j].data(), target_W.data());
                target_rots[j] = target_rot;
                targets_W_rot[j] = target_W;
                targets_W[j] = target_W + target_pos;
            }


            residuals[0] = weight_ * data_association_.computeResidual(feature_W, targets_W);

            if(jacobians != NULL)
            {
                RowX d_res_d_pts = data_association_.computeJacobian(feature_W, targets_W);
                Mat3 d_feature_d_rot = -ugpm::toSkewSymMat(feature_W_rot)*ugpm::jacobianRighthandSO3(-feature_rot);
                std::vector<Mat3> d_target_d_rot(data_association_.target_ids.size());
                for(size_t j = 0; j < data_association_.target_ids.size(); ++j)
                {
                    Vec3& target_rot = poses[j+1].second;
                    d_target_d_rot[j] = -ugpm::toSkewSymMat(targets_W_rot[j])*ugpm::jacobianRighthandSO3(-target_rot);
                }


                for(int j = 0; j < 5; ++j)
                {
                    if(jacobians[j] != NULL)
                    {
                        int block_size = block_sizes_[j];
                        Eigen::Map<Eigen::Matrix<double, 1,Eigen::Dynamic> > j_s(&(jacobians[j][0]),1,block_size);
                        j_s.setZero();
                        if(pose_jacobians[0][j].first.size() > 0)
                        {
                            j_s += d_res_d_pts.segment<3>(0) * pose_jacobians[0][j].first;
                        }
                        if(pose_jacobians[0][j].second.size() > 0)
                        {
                            j_s += d_res_d_pts.segment<3>(0) * d_feature_d_rot * pose_jacobians[0][j].second;
                        }
                        for(size_t k = 0; k < data_association_.target_ids.size(); ++k)
                        {
                            if(pose_jacobians[k+1][j].first.size() > 0)
                            {
                                j_s += d_res_d_pts.segment<3>(3*(k+1)) * pose_jacobians[k+1][j].first;
                            }
                            if(pose_jacobians[k+1][j].second.size() > 0)
                            {
                                j_s += d_res_d_pts.segment<3>(3*(k+1)) * d_target_d_rot[k] * pose_jacobians[k+1][j].second;
                            }
                        }
                        j_s *= weight_;
                    }
                }
            }

            return true;
        }
};

