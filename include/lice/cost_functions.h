#pragma once

#include "lice/types.h"
#include "lice/utils.h"
#include "lice/math_utils.h"
#include <ceres/rotation.h>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

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


// The lidar residuals of one problem, evaluated from the state cache: what they need from the state
// at the evaluation point (its value and jacobians at the state times, around 70 of them) is gathered
// once per solver evaluation by StateCacheCallback, and each residual then computes its own points
// in its Evaluate, so ceres' threads share the work as they did with LidarNoCalCostFunction.
//
// Against LidarNoCalCostFunction, which queried the state for each point (copying 8 cached 3x3
// matrices and interpolating 4 to 5 of them) and took the sine and cosine of the rotation angle
// twice (the rotation, then the right jacobian), a point here costs one sin/cos, the position
// jacobians are read in place (linear in the interpolation weight between two state times,
// g J(alpha) = g J0 + alpha g (J1 - J0)), those w.r.t. gravity and velocity are the scalar times
// the identity they are, and the jacobian rows are accumulated locally. The points are not shared
// between residuals: the neighbours of different associations barely overlap (about 2%).
// The results are those of LidarNoCalCostFunction up to the rounding of the reordered operations.
//
// Needs the state cache (IMU and GYR modes, where StateCacheCallback exists). The point times do
// not change during a solve, so each point's interpolation interval is found once, when it is added.
class LidarResiduals
{
    public:
        explicit LidarResiduals(const State& state)
            : state_(state)
        {}

        // Adds the residual of one association and returns its index. Same inputs as the
        // LidarNoCalCostFunction constructor.
        size_t addResidual(
                const DataAssociation& data_association
                , const std::vector<std::shared_ptr<std::vector<Pointd> > >& features
                , const std::vector<std::shared_ptr<std::vector<Pointd> > >& sparse_features
                , const double weight
                , const int64_t offset_time
                , const Vec7& extrinsic)
        {
            if(data_association.target_ids.size() > kMaxAssociationTargets)
            {
                throw std::invalid_argument("LidarResiduals: an association has more than kMaxAssociationTargets target points");
            }
            const Vec4 quat_I_L = extrinsic.head<4>();
            const Vec3 pos_I_L = extrinsic.tail<3>();
            auto addPoint = [&](const Pointd& pt)
            {
                ResidualPoint rp;
                rp.p_I = pt.vec3();
                ceres::UnitQuaternionRotatePoint(quat_I_L.data(), rp.p_I.data(), rp.p_I.data());
                rp.p_I += pos_I_L;
                state_.interpolationInterval(nanosToSeconds(pt.t, offset_time), rp.state_id, rp.alpha);
                pts_.push_back(rp);
            };

            Residual r;
            r.da = &data_association;
            r.weight = weight;
            r.first_point = (uint32_t)pts_.size();
            r.nb_targets = (uint32_t)data_association.target_ids.size();
            addPoint(sparse_features[data_association.pc_id]->at(data_association.feature_id));
            for(const auto& [chunk, index] : data_association.target_ids)
            {
                addPoint(features[chunk]->at(index));
            }
            residuals_.push_back(r);
            valid_values_ = false;
            valid_jacobians_ = false;
            return residuals_.size() - 1;
        }

        // Called before every evaluation (see StateCacheCallback), once the state cache holds the
        // evaluation point: the state data at the state times, and when with_jacobians, the steps of
        // the bias jacobians between consecutive ones.
        void prepare(const Vec3& vel, const bool with_jacobians, const bool new_evaluation_point)
        {
            if(new_evaluation_point)
            {
                valid_values_ = false;
                valid_jacobians_ = false;
            }
            if(!valid_values_)
            {
                state_.cachedStateKnots(vel, knots_);
                valid_values_ = true;
            }
            if(with_jacobians && !valid_jacobians_)
            {
                d_knots_.resize(knots_.size() > 0 ? knots_.size() - 1 : 0);
                for(size_t k = 0; k + 1 < knots_.size(); ++k)
                {
                    d_knots_[k].acc_bias = knots_[k+1].pos_jac_acc_bias - knots_[k].pos_jac_acc_bias;
                    d_knots_[k].gyr_bias = knots_[k+1].pos_jac_gyr_bias - knots_[k].pos_jac_gyr_bias;
                }
                valid_jacobians_ = true;
            }
        }

        // Residual i and its jacobians w.r.t. the 4 state blocks (acc_bias, gyr_bias, gravity, vel),
        // at the point of the last prepare(). Only reads the shared data: safe from ceres' threads.
        void evaluate(const size_t i, double* residual, double** jacobians) const
        {
            const bool with_jacobians = (jacobians != nullptr);
            if(!valid_values_ || (with_jacobians && !valid_jacobians_))
            {
                throw std::logic_error("LidarResiduals::evaluate: not prepared for this evaluation");
            }
            const Residual& r = residuals_[i];
            const uint32_t nb_points = 1 + r.nb_targets;
            std::array<Vec3, 1 + kMaxAssociationTargets> p_W;
            std::array<Mat3, 1 + kMaxAssociationTargets> dp_W_d_bw;
            for(uint32_t m = 0; m < nb_points; ++m)
            {
                projectPoint(pts_[r.first_point + m], with_jacobians, p_W[m], dp_W_d_bw[m]);
            }
            TargetPoints targets;
            targets.n = r.nb_targets;
            for(uint32_t j = 0; j < r.nb_targets; ++j)
            {
                targets[j] = p_W[j+1];
            }
            residual[0] = r.weight * r.da->computeResidual(p_W[0], targets);
            if(!with_jacobians)
            {
                return;
            }

            // Accumulated locally for the 4 blocks, then written to the ones ceres asks for
            const RowAssocJacobian d_res_d_pts = r.da->computeJacobian(p_W[0], targets);
            Eigen::Matrix<double, 1, 3> j_acc_bias = Eigen::Matrix<double, 1, 3>::Zero();
            Eigen::Matrix<double, 1, 3> j_gyr_bias = Eigen::Matrix<double, 1, 3>::Zero();
            Eigen::Matrix<double, 1, 3> j_gravity = Eigen::Matrix<double, 1, 3>::Zero();
            Eigen::Matrix<double, 1, 3> j_vel = Eigen::Matrix<double, 1, 3>::Zero();
            for(uint32_t m = 0; m < nb_points; ++m)
            {
                const ResidualPoint& rp = pts_[r.first_point + m];
                const State::CachedStateKnot& k0 = knots_[rp.state_id];
                const State::CachedStateKnot& k1 = knots_[rp.state_id + 1];
                const KnotStep& dk = d_knots_[rp.state_id];
                const Eigen::Matrix<double, 1, 3> g = d_res_d_pts.segment<3>(3*m);
                const Eigen::Matrix<double, 1, 3> g_alpha = rp.alpha * g;
                j_acc_bias += g * k0.pos_jac_acc_bias + g_alpha * dk.acc_bias;
                j_gyr_bias += g * k0.pos_jac_gyr_bias + g_alpha * dk.gyr_bias + g * dp_W_d_bw[m];
                j_gravity += (k0.pos_jac_gravity + rp.alpha * (k1.pos_jac_gravity - k0.pos_jac_gravity)) * g;
                j_vel += (k0.pos_jac_vel + rp.alpha * (k1.pos_jac_vel - k0.pos_jac_vel)) * g;
            }
            const Eigen::Matrix<double, 1, 3>* rows[4] = {&j_acc_bias, &j_gyr_bias, &j_gravity, &j_vel};
            for(int b = 0; b < 4; ++b)
            {
                if(jacobians[b] != nullptr)
                {
                    Eigen::Map<Eigen::Matrix<double, 1, 3> > j_out(jacobians[b]);
                    j_out = r.weight * (*rows[b]);
                }
            }
        }

    private:
        struct ResidualPoint
        {
            Vec3 p_I;       // in the IMU frame (calibration applied)
            int state_id;   // interpolation interval and weight
            double alpha;
        };
        struct Residual
        {
            const DataAssociation* da;
            double weight;
            uint32_t first_point;   // the source, then the targets, in pts_
            uint32_t nb_targets;
        };
        // Position jacobians w.r.t. the biases, J(k+1) - J(k)
        struct KnotStep
        {
            Mat3 acc_bias;
            Mat3 gyr_bias;
        };

        // The point in the world frame at its time and, with the jacobians, d p_W / d gyr_bias
        // through the rotation. The rotation and the right jacobian of -rot come from one sin/cos of
        // the angle: the former as ceres::AngleAxisRotatePoint writes it, the latter as
        // ugpm::jacobianRighthandSO3(-rot) does, I + (theta - s)/theta^3 K^2 + (1 - c)/theta^2 K with
        // K = [rot]x (the sign of [-rot]x only flips the odd power), with the same small angle branches
        void projectPoint(const ResidualPoint& rp, const bool with_jacobians, Vec3& p_W, Mat3& dp_W_d_bw) const
        {
            const State::CachedStateKnot& k0 = knots_[rp.state_id];
            const State::CachedStateKnot& k1 = knots_[rp.state_id + 1];
            const Vec3 pos = k0.pos + rp.alpha * (k1.pos - k0.pos);
            const Vec3 rot = k0.rot + rp.alpha * (k1.rot - k0.rot);

            const double theta2 = rot.squaredNorm();
            const double theta = std::sqrt(theta2);
            double s = 0.0;
            double c = 1.0;
            if((theta2 > std::numeric_limits<double>::epsilon()) || (theta > ugpm::kExpNormTolerance))
            {
                sincos(theta, &s, &c);
            }
            Vec3 rotated;
            if(theta2 > std::numeric_limits<double>::epsilon())
            {
                const Vec3 w = rot / theta;
                rotated = rp.p_I*c + w.cross(rp.p_I)*s + w*(w.dot(rp.p_I)*(1.0 - c));
            }
            else
            {
                rotated = rp.p_I + rot.cross(rp.p_I);
            }
            p_W = rotated + pos;
            if(!with_jacobians)
            {
                return;
            }

            Mat3 right_jacobian = Mat3::Identity();
            if(theta > ugpm::kExpNormTolerance)
            {
                Mat3 K;
                K << 0.0, -rot(2), rot(1),
                     rot(2), 0.0, -rot(0),
                     -rot(1), rot(0), 0.0;
                right_jacobian += ((theta - s)/(theta2*theta))*(K*K) + ((1.0 - c)/theta2)*K;
            }
            Mat3 minus_skew_rotated;
            minus_skew_rotated << 0.0, rotated(2), -rotated(1),
                                  -rotated(2), 0.0, rotated(0),
                                  rotated(1), -rotated(0), 0.0;
            dp_W_d_bw = (minus_skew_rotated*right_jacobian)
                    * (k0.rot_jac_gyr_bias + rp.alpha * (k1.rot_jac_gyr_bias - k0.rot_jac_gyr_bias));
        }

        const State& state_;
        std::vector<ResidualPoint> pts_;
        std::vector<Residual> residuals_;

        // At the evaluation point
        std::vector<State::CachedStateKnot> knots_;
        std::vector<KnotStep> d_knots_;
        bool valid_values_ = false;
        bool valid_jacobians_ = false;
};


// One residual of a LidarResiduals (see there). The parameter blocks are declared for ceres, but the
// values are read from the state data StateCacheCallback gathered from the same blocks just before
// the evaluation.
class LidarResidualCostFunction: public ceres::SizedCostFunction<1, 3,3,3,3>
{
    private:
        const LidarResiduals& residuals_;
        const size_t id_;

    public:
        LidarResidualCostFunction(const LidarResiduals& residuals, const size_t id)
            : residuals_(residuals)
            , id_(id)
        {}

        bool Evaluate(double const* const* /*parameters*/, double* residuals, double** jacobians) const override
        {
            residuals_.evaluate(id_, residuals, jacobians);
            return true;
        }
};


class StateCacheCallback: public ceres::EvaluationCallback
{
    private:
        State& state_;
        Vec3& acc_bias_;
        Vec3& gyr_bias_;
        Vec3& gravity_;
        Vec3& vel_;
        LidarResiduals* lidar_residuals_;
    public:
        StateCacheCallback(State& state, Vec3& acc_bias, Vec3& gyr_bias, Vec3& gravity, Vec3& vel, LidarResiduals* lidar_residuals = nullptr): state_(state), acc_bias_(acc_bias), gyr_bias_(gyr_bias), gravity_(gravity), vel_(vel), lidar_residuals_(lidar_residuals)
        {}

        virtual void PrepareForEvaluation(bool evaluate_jacobians, bool new_evaluation_point) override
        {
            if (new_evaluation_point)
            {
                state_.computeCache(acc_bias_, gyr_bias_, gravity_, vel_);
            }
            if(lidar_residuals_ != nullptr)
            {
                lidar_residuals_->prepare(vel_, evaluate_jacobians, new_evaluation_point);
            }
        }
};



class LidarNoCalCostFunction: public ceres::SizedCostFunction<1, 3,3,3,3>
{
    private:
        const State& state_;
        const DataAssociation data_association_;
        Vec3 source_pt_;
        std::vector<Vec3> target_pts_;
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
                , weight_(weight)
        {
            // Evaluate holds the targets in fixed-capacity containers
            if(data_association_.target_ids.size() > kMaxAssociationTargets)
            {
                throw std::invalid_argument("LidarNoCalCostFunction: an association has more than kMaxAssociationTargets target points");
            }
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

            // Fixed capacity rather than std::vector: this runs once per residual per solver evaluation,
            // and every container below used to be a heap allocation (the target count is checked in
            // the constructor)
            const size_t nb_targets = data_association_.target_ids.size();
            std::array<std::pair<Vec3, Vec3>, 1 + kMaxAssociationTargets> poses;
            std::array<std::array<std::pair<Mat3, Mat3>, 4>, 1 + kMaxAssociationTargets> pose_jacobians;
            if(jacobians != NULL)
            {
                for(size_t i = 0; i < feature_times_.size(); ++i)
                {
                    std::tie(poses[i], pose_jacobians[i]) = state_.queryWthJacobian(feature_times_[i], arg_0, arg_1, arg_2, arg_3, true);
                }
            }
            else
            {
                for(size_t i = 0; i < feature_times_.size(); ++i)
                {
                    poses[i] = state_.query(feature_times_[i], arg_0, arg_1, arg_2, arg_3, true);
                }
            }

            Vec3& feature_rot = poses[0].second;
            Vec3& feature_pos = poses[0].first;

            Vec3 feature_W_rot;
            ceres::AngleAxisRotatePoint(feature_rot.data(), source_pt_.data(), feature_W_rot.data());
            Vec3 feature_W = feature_W_rot + feature_pos;

            std::array<Vec3, kMaxAssociationTargets> targets_W_rot;
            TargetPoints targets_W;
            targets_W.n = nb_targets;
            for(size_t j = 0; j < nb_targets; ++j)
            {
                Vec3& target_rot = poses[j+1].second;
                Vec3& target_pos = poses[j+1].first;

                Vec3 target_W;
                ceres::AngleAxisRotatePoint(target_rot.data(), target_pts_[j].data(), target_W.data());
                targets_W_rot[j] = target_W;
                targets_W[j] = target_W + target_pos;
            }


            residuals[0] = weight_ * data_association_.computeResidual(feature_W, targets_W);

            if(jacobians != NULL)
            {
                const RowAssocJacobian d_res_d_pts = data_association_.computeJacobian(feature_W, targets_W);
                Mat3 d_feature_d_rot = -ugpm::toSkewSymMat(feature_W_rot)*ugpm::jacobianRighthandSO3(-feature_rot);
                std::array<Mat3, kMaxAssociationTargets> d_target_d_rot;
                for(size_t j = 0; j < nb_targets; ++j)
                {
                    Vec3& target_rot = poses[j+1].second;
                    d_target_d_rot[j] = -ugpm::toSkewSymMat(targets_W_rot[j])*ugpm::jacobianRighthandSO3(-target_rot);
                }


                for(int j = 0; j < 4; ++j)
                {
                    if(jacobians[j] != NULL)
                    {
                        Eigen::Map<Eigen::Matrix<double, 1,3> > j_s(&(jacobians[j][0]));
                        j_s = d_res_d_pts.segment<3>(0) * pose_jacobians[0][j].first;
                        if(j==1)
                        {
                            j_s += d_res_d_pts.segment<3>(0) * d_feature_d_rot * pose_jacobians[0][j].second;
                        }
                        for(size_t k = 0; k < nb_targets; ++k)
                        {
                            j_s += d_res_d_pts.segment<3>(3*(k+1)) * pose_jacobians[k+1][j].first;
                            if(j==1)
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

