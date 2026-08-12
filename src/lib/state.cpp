#include "lice/state.h"

#include "lice/utils.h"
#include "lice/math_utils.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

State::State(const ugpm::ImuData& imu_data, const double first_t, const double state_freq, const LidarOdometryMode mode)
    : start_t_(first_t)
    , mode_(mode)
{
    state_period_ = 1.0 / state_freq;
    double last_t;
    if(mode_ == LidarOdometryMode::IMU)
    {
        last_t = std::max(imu_data.acc.back().t, imu_data.gyr.back().t);
    }
    else if(mode_ == LidarOdometryMode::GYR)
    {
        last_t = imu_data.gyr.back().t;
    }
    else // NO_IMU
    {
        last_t = first_t + 0.25; // Arbitrary 0.25 seconds duration
    }
    nb_state_ = std::ceil((last_t - first_t) / state_period_);
    for(int i = 0; i < nb_state_; ++i)
    {
        state_time_.push_back(first_t + i * state_period_);
    }

    if(mode_ != LidarOdometryMode::NO_IMU)
    {
        preint_meas_.resize(nb_state_);

        ugpm::PreintOption opt;
        opt.type = ugpm::PreintType::LPM;
        opt.min_freq = 500;

        ugpm::ImuPreintegration preint(imu_data, first_t, state_time_, opt, ugpm::PreintPrior());

        for(int i = 0; i < nb_state_; ++i)
        {
            preint_meas_[i] = preint.get(i);
        }
    }
}

std::vector<std::pair<Vec3, Vec3> > State::queryApprox(
        const std::vector<double>& query_time
        , const Vec3& acc_bias
        , const Vec3& gyr_bias
        , const Vec3& gravity
        , const Vec3& vel
        ) const
{
    std::vector<std::pair<Vec3, Vec3> > query_pose(query_time.size());
    // Get the pose at the state time
    std::vector<std::pair<Vec3, Vec3> > state_pose(nb_state_);
    for(int i = 0; i < nb_state_; ++i)
    {
        Mat3 R;
        Vec3 p;
        if(mode_ == LidarOdometryMode::NO_IMU)
        {
            double dt = state_time_.at(i) - start_t_;
            R = expMap(gyr_bias * dt);
            p = vel * dt;
        }
        else
        {
            const ugpm::PreintMeas& preint = preint_meas_.at(i);

            R = preint.delta_R * ugpm::expMap(preint.d_delta_R_d_bw * gyr_bias);
            if(mode_ == LidarOdometryMode::GYR)
            {
                p = vel * (state_time_.at(i) - start_t_);
            }
            else
            {
                p = preint.delta_p + preint.d_delta_p_d_bf * acc_bias + preint.d_delta_p_d_bw * gyr_bias + vel*preint.dt + gravity*preint.dt_sq_half;
            }
        }
        state_pose.at(i) = {p, ugpm::logMap(R)};
    }

    // Compute the pose at the query time as a linear interpolation of the state poses
    for(size_t i = 0; i < query_time.size(); ++i)
    {
        double t = query_time.at(i);
        int state_id = std::floor((t - state_time_.at(0)) / state_period_);
        if(state_id < 0)
        {
            state_id = 0;
        }
        else if(state_id >= nb_state_-1)
        {
            state_id = nb_state_-2;
        }
        double t0 = state_time_.at(state_id);
        double t1 = state_time_.at(state_id+1);
        double alpha = (t - t0) / (t1 - t0);

        const Vec3& p0 = state_pose.at(state_id).first;
        const Vec3& p1 = state_pose.at(state_id+1).first;
        const Vec3& r0 = state_pose.at(state_id).second;
        const Vec3& r1 = state_pose.at(state_id+1).second;

        query_pose.at(i).first = p0 + alpha * (p1 - p0);
        query_pose.at(i).second = r0 + alpha * (r1 - r0);
    }
    return query_pose;

}



// Overload to query a single time
std::pair<Vec3, Vec3> State::query(
        const double query_time
        , const Vec3& acc_bias
        , const Vec3& gyr_bias
        , const Vec3& gravity
        , const Vec3& vel
        , const bool use_cache
        ) const
{
    std::pair<Vec3, Vec3> query_pose;


    if( mode_ == LidarOdometryMode::NO_IMU)
    {
        double dt = query_time - start_t_;
        Vec3 p = vel * dt;
        query_pose.first = p;
        query_pose.second = gyr_bias * dt;
    }
    else
    {
        double t = query_time;
        int state_id = std::floor((t - state_time_[0]) / state_period_);
        if(state_id < 0)
        {
            state_id = 0;
        }
        else if(state_id >= nb_state_-1)
        {
            state_id = nb_state_-2;
        }
        double t0 = state_time_[state_id];
        double t1 = state_time_[state_id+1];
        double alpha = (t - t0) / (t1 - t0);

        Vec3 p0;
        Vec3 p1;
        if(mode_ == LidarOdometryMode::GYR)
        {
            p0 = vel * (state_time_[state_id] - start_t_);
            p1 = vel * (state_time_[state_id+1] - start_t_);
        }
        else
        {
            if(use_cache && !cached_state_poses_.empty())
            {
                p0 = cached_state_poses_[state_id].first;
                p1 = cached_state_poses_[state_id+1].first;
            }
            else
            {
                p0 = preint_meas_[state_id].delta_p + preint_meas_[state_id].d_delta_p_d_bf * acc_bias + preint_meas_[state_id].d_delta_p_d_bw * gyr_bias + vel*preint_meas_[state_id].dt + gravity*preint_meas_[state_id].dt_sq_half;
                p1 = preint_meas_[state_id+1].delta_p + preint_meas_[state_id+1].d_delta_p_d_bf * acc_bias + preint_meas_[state_id+1].d_delta_p_d_bw * gyr_bias + vel*preint_meas_[state_id+1].dt + gravity*preint_meas_[state_id+1].dt_sq_half;
            }
        }
        
        // Approximation of the rotation interpolation: linear interpolation of the log map of the rotation. Mathematically, this is not correct, but it is good enough for small rotations and it is much faster than computing the exact interpolation on SO(3).
        Vec3 r0;
        Vec3 r1;
        if(use_cache && !cached_state_poses_.empty())
        {
            r0 = cached_state_poses_[state_id].second;
            r1 = cached_state_poses_[state_id+1].second;
        }
        else
        {
            r0 = ugpm::logMap(preint_meas_[state_id].delta_R * ugpm::expMap(preint_meas_[state_id].d_delta_R_d_bw * gyr_bias));
            r1 = ugpm::logMap(preint_meas_[state_id+1].delta_R * ugpm::expMap(preint_meas_[state_id+1].d_delta_R_d_bw * gyr_bias));
        }

        query_pose.first = p0 + alpha * (p1 - p0);
        query_pose.second = r0 + alpha * (r1 - r0);
    }

    return query_pose;
}

// Overload to query a single time
std::tuple<std::pair<Vec3, Vec3>,
        std::array<std::pair<Mat3, Mat3>, 4> > State::queryWthJacobian(
        const double query_time
        , const Vec3& acc_bias
        , const Vec3& gyr_bias
        , const Vec3& gravity
        , const Vec3& vel
        , const bool use_cache
        ) const
{
    std::pair<Vec3, Vec3> query_pose;
    std::array<std::pair<Mat3, Mat3>, 4> query_jacobian;

    if( mode_ == LidarOdometryMode::NO_IMU)
    {
        double dt = query_time - start_t_;
        Mat3 R = expMap(gyr_bias * dt);
        query_pose.first = vel * dt;
        query_pose.second = ugpm::logMap(R);

        query_jacobian[0].first = Mat3::Zero();
        query_jacobian[0].second = Mat3::Zero();
        query_jacobian[1].first = Mat3::Zero();
        query_jacobian[1].second = Mat3::Identity()*dt;
        query_jacobian[2].first = Mat3::Zero();
        query_jacobian[2].second = Mat3::Zero();
        query_jacobian[3].first = Mat3::Identity()*dt;
        query_jacobian[3].second = Mat3::Zero();

    }
    else
    {
        // Get the pose at the state time
        std::array<Mat3, 4> state_jacobian_0;
        std::array<Mat3, 4> state_jacobian_1;


        std::array<Mat3, 3> state_R_shift_dw_0;
        std::array<Mat3, 3> state_R_shift_dw_1;

        double eps = eps_;


        double t = query_time;
        int state_id = std::floor((t - state_time_[0]) / state_period_);
        if(state_id < 0)
        {
            state_id = 0;
        }
        else if(state_id >= nb_state_-1)
        {
            state_id = nb_state_-2;
        }
        double t0 = state_time_[state_id];
        double t1 = state_time_[state_id+1];
        double alpha = (t - t0) / (t1 - t0);

        Vec3 p0;
        Vec3 p1;
        if(mode_ == LidarOdometryMode::GYR)
        {
            p0 = vel * (state_time_[state_id] - start_t_);
            p1 = vel * (state_time_[state_id+1] - start_t_);

            state_jacobian_0[0] = Mat3::Zero();
            state_jacobian_0[1] = Mat3::Zero();
            state_jacobian_0[2] = Mat3::Zero();
            state_jacobian_0[3] = Mat3::Identity()* (state_time_[state_id] - start_t_);

            state_jacobian_1[0] = Mat3::Zero();
            state_jacobian_1[1] = Mat3::Zero();
            state_jacobian_1[2] = Mat3::Zero();
            state_jacobian_1[3] = Mat3::Identity()* (state_time_[state_id+1] - start_t_);
        }
        else
        {
            if(use_cache && !cached_state_poses_.empty())
            {
                p0 = cached_state_poses_[state_id].first;
                p1 = cached_state_poses_[state_id+1].first;
            
                state_jacobian_0 = cached_state_jacobians_[state_id];
                state_jacobian_1 = cached_state_jacobians_[state_id+1];
            }
            else
            {
                p0 = preint_meas_[state_id].delta_p + preint_meas_[state_id].d_delta_p_d_bf * acc_bias + preint_meas_[state_id].d_delta_p_d_bw * gyr_bias + vel*preint_meas_[state_id].dt + gravity*preint_meas_[state_id].dt_sq_half;
                p1 = preint_meas_[state_id+1].delta_p + preint_meas_[state_id+1].d_delta_p_d_bf * acc_bias + preint_meas_[state_id+1].d_delta_p_d_bw * gyr_bias + vel*preint_meas_[state_id+1].dt + gravity*preint_meas_[state_id+1].dt_sq_half;

                state_jacobian_0[0] = preint_meas_[state_id].d_delta_p_d_bf;
                state_jacobian_0[1] = preint_meas_[state_id].d_delta_p_d_bw;
                state_jacobian_0[2] = Mat3::Identity()*preint_meas_[state_id].dt_sq_half;
                state_jacobian_0[3] = Mat3::Identity()*preint_meas_[state_id].dt;

                state_jacobian_1[0] = preint_meas_[state_id+1].d_delta_p_d_bf;
                state_jacobian_1[1] = preint_meas_[state_id+1].d_delta_p_d_bw;
                state_jacobian_1[2] = Mat3::Identity()*preint_meas_[state_id+1].dt_sq_half;
                state_jacobian_1[3] = Mat3::Identity()*preint_meas_[state_id+1].dt;
            }
        }

        // Approximation of the rotation interpolation: linear interpolation of the log map of the rotation. Mathematically, this is not correct, but it is good enough for small rotations and it is much faster than computing the exact interpolation on SO(3).
        Vec3 r0;
        Vec3 r1;
        
        if(use_cache && !cached_state_poses_.empty())
        {
            r0 = cached_state_poses_[state_id].second;
            r1 = cached_state_poses_[state_id+1].second;

            state_jacobian_0 = cached_state_jacobians_[state_id];
            state_jacobian_1 = cached_state_jacobians_[state_id+1];
        }
        else
        {

            r0 = ugpm::logMap(preint_meas_[state_id].delta_R * ugpm::expMap(preint_meas_[state_id].d_delta_R_d_bw * gyr_bias));
            r1 = ugpm::logMap(preint_meas_[state_id+1].delta_R * ugpm::expMap(preint_meas_[state_id+1].d_delta_R_d_bw * gyr_bias));

        }

        query_pose.first = p0 + alpha * (p1 - p0);
        query_pose.second = r0 + alpha * (r1 - r0);

        // Compute the jacobian
        for(int j = 0; j < 4; ++j)
        {
            query_jacobian[j].first = state_jacobian_0[j] + alpha * (state_jacobian_1[j] - state_jacobian_0[j]);

            if(j == 1)
            {
                if(use_cache && !cached_state_poses_.empty())
                {
                    query_jacobian[j].second = cached_state_dr_dw_[state_id] + alpha * (cached_state_dr_dw_[state_id+1] - cached_state_dr_dw_[state_id]);
                }
                else
                {

                    Vec3 dw_shift = gyr_bias;
                    Mat3 dr_dw0;
                    Mat3 dr_dw1;
                    for(int j = 0; j < 3; ++j)
                    {
                        dw_shift[j] += eps;
                        dr_dw0.col(j) = (ugpm::logMap(preint_meas_[state_id].delta_R * ugpm::expMap(preint_meas_[state_id].d_delta_R_d_bw * dw_shift)) - r0) / eps;
                        dr_dw1.col(j) = (ugpm::logMap(preint_meas_[state_id+1].delta_R * ugpm::expMap(preint_meas_[state_id+1].d_delta_R_d_bw * dw_shift)) - r1) / eps;
                        dw_shift[j] -= eps;
                    }
                    query_jacobian[j].second = dr_dw0 + alpha * (dr_dw1 - dr_dw0);
                }
                //}
            }
        }
    }

    return {query_pose, query_jacobian};
}


// Query the linear (first) and angular (second) velocity at the query time
std::pair<Vec3, Vec3> State::queryTwist(
        const double query_time
        , const Vec3& acc_bias
        , const Vec3& gyr_bias
        , const Vec3& gravity
        , const Vec3& vel
        ) const
{
    std::pair<Vec3, Vec3> query_vel;

    if( mode_ == LidarOdometryMode::NO_IMU)
    {
        query_vel.first = vel;
        query_vel.second = gyr_bias;
    }
    else
    {
        // Get the pose at the state time
        std::vector<std::tuple<Vec3, Mat3, Vec3> > state_pose(nb_state_);
        for(int i = 0; i < nb_state_; ++i)
        {
            const ugpm::PreintMeas& preint = preint_meas_.at(i);

            Mat3 R = preint.delta_R * ugpm::expMap(preint.d_delta_R_d_bw * gyr_bias);
            Vec3 v;
            if(mode_ == LidarOdometryMode::GYR)
            {
                v = vel;
            }
            else
            {
                v = vel + preint.delta_v + preint.d_delta_v_d_bf * acc_bias + preint.d_delta_v_d_bw * gyr_bias + gravity*preint.dt;
            }
            state_pose.at(i) = {Vec3::Zero(), R, v};
        }

        // Compute the pose at the query time as a linear interpolation of the state poses
        int state_id = std::floor((query_time - state_time_.at(0)) / state_period_);
        if(state_id < 0)
        {
            state_id = 0;
        }
        else if(state_id >= nb_state_-1)
        {
            state_id = nb_state_-2;
        }
        double t0 = state_time_.at(state_id);
        double t1 = state_time_.at(state_id+1);
        double alpha = (query_time - t0) / (t1 - t0);

        const Mat3& R0 = std::get<1>(state_pose.at(state_id));
        const Mat3& R1 = std::get<1>(state_pose.at(state_id+1));

        Vec3 delta_r = ugpm::logMap(R0.transpose() * R1);
        query_vel.second = delta_r / (t1 - t0);

        const Vec3& v0 = std::get<2>(state_pose.at(state_id));
        const Vec3& v1 = std::get<2>(state_pose.at(state_id+1));
        Vec3 temp_vel = v0 + alpha * (v1 - v0);
        Mat3 temp_R = R0 * ugpm::expMap(delta_r * alpha);
        query_vel.first = temp_R.transpose() * temp_vel;
    }

    return query_vel;
}


void State::computeCache(const Vec3& acc_bias, const Vec3& gyr_bias, const Vec3& gravity, const Vec3& vel)
{
    cached_state_poses_.clear();
    cached_state_jacobians_.clear();
    cached_state_dr_dw_.clear();
    for(int i = 0; i < nb_state_; ++i)
    {
        Vec3 p = preint_meas_[i].delta_p + preint_meas_[i].d_delta_p_d_bf * acc_bias + preint_meas_[i].d_delta_p_d_bw * gyr_bias + vel*preint_meas_[i].dt + gravity*preint_meas_[i].dt_sq_half;
        Vec3 r = ugpm::logMap(preint_meas_[i].delta_R * ugpm::expMap(preint_meas_[i].d_delta_R_d_bw * gyr_bias));
        cached_state_poses_.push_back({p, r});


        std::array<Mat3, 4> state_jacobian;
        state_jacobian[0] = preint_meas_[i].d_delta_p_d_bf;
        state_jacobian[1] = preint_meas_[i].d_delta_p_d_bw;
        state_jacobian[2] = Mat3::Identity()*preint_meas_[i].dt_sq_half;
        state_jacobian[3] = Mat3::Identity()*preint_meas_[i].dt;
        cached_state_jacobians_.push_back(state_jacobian);

        Vec3 dw_shift = gyr_bias;
        Mat3 dr_dw;
        for(int j = 0; j < 3; ++j)
        {
            dw_shift[j] += eps_;
            Vec3 r_shift = ugpm::logMap(preint_meas_[i].delta_R * ugpm::expMap(preint_meas_[i].d_delta_R_d_bw * dw_shift));
            dr_dw.col(j) = (r_shift - r) / eps_;
            dw_shift[j] -= eps_;
        }
        cached_state_dr_dw_.push_back(dr_dw);
    }
}





// Generate a smooth synthetic IMU sequence, with enough excitation on the 3 axes for the jacobians
// to be well defined. The rotations are kept well below pi so that the rotation vectors do not wrap.
static ugpm::ImuData generateTestImuData(const double first_t, const double duration, const double freq)
{
    ugpm::ImuData imu_data;
    imu_data.acc_var = 1e-4;
    imu_data.gyr_var = 1e-6;

    const int nb_samples = (int)std::ceil(duration*freq) + 1;
    for(int i = 0; i < nb_samples; ++i)
    {
        const double t = first_t + i/freq;
        const double s = i/freq;

        ugpm::ImuSample gyr;
        gyr.t = t;
        gyr.data[0] = 0.35*std::sin(1.7*s);
        gyr.data[1] = 0.25*std::cos(1.1*s);
        gyr.data[2] = 0.45*std::sin(0.7*s + 0.4);
        imu_data.gyr.push_back(gyr);

        ugpm::ImuSample acc;
        acc.t = t;
        acc.data[0] = 0.9*std::sin(0.9*s);
        acc.data[1] = 0.7*std::cos(1.3*s + 0.2);
        acc.data[2] = 9.81 + 0.5*std::sin(2.1*s);
        imu_data.acc.push_back(acc);
    }
    return imu_data;
}

// Report the difference between an analytic and a numerical jacobian block
// queryWthJacobian only fills the rotation part of the gyroscope bias block when the state comes
// from the IMU preintegration: the rotation does not depend on the accelerometer bias, the gravity,
// nor the velocity, so those blocks are left uninitialised instead of being zeroed. They must not be
// read. In NO_IMU mode every block is explicitly set.
static bool isRotationBlockSet(const LidarOdometryMode mode, const int block)
{
    if(mode == LidarOdometryMode::NO_IMU)
    {
        return true;
    }
    return (block == 1);
}

static void reportJacobianBlock(const std::string& name, const Mat3& analytic, const Mat3& numerical, double& worst_error)
{
    const double error = (analytic - numerical).cwiseAbs().maxCoeff();
    const double scale = std::max(1.0, numerical.cwiseAbs().maxCoeff());
    worst_error = std::max(worst_error, error/scale);

    std::cout << "      " << std::left << std::setw(26) << name
              << " max|analytic-numerical| = " << std::scientific << std::setprecision(3) << error
              << "   (max|numerical| = " << numerical.cwiseAbs().maxCoeff() << ")" << std::endl;
    if(error/scale > 1e-3)
    {
        std::cout << "        ^^ MISMATCH" << std::endl;
        std::cout << "        analytic:\n" << analytic << std::endl;
        std::cout << "        numerical:\n" << numerical << std::endl;
    }
}

// Check the jacobians returned by State::queryWthJacobian against central finite differences.
// The reference is the pose returned by queryWthJacobian itself, so that the test validates the
// jacobian against the very function it is supposed to differentiate.
static double testJacobiansOneConfig(State& state, const LidarOdometryMode mode, const bool use_cache, const std::vector<double>& query_times,
        const Vec3& acc_bias, const Vec3& gyr_bias, const Vec3& gravity, const Vec3& vel)
{
    const double eps = 1e-6;
    const char* block_names[4] = {"d/d_acc_bias", "d/d_gyr_bias", "d/d_gravity", "d/d_vel"};
    double worst_error = 0.0;

    for(const double t : query_times)
    {
        std::cout << "    query time offset " << std::fixed << std::setprecision(4) << t << " s" << std::endl;

        if(use_cache)
        {
            state.computeCache(acc_bias, gyr_bias, gravity, vel);
        }
        auto [pose, jacobian] = state.queryWthJacobian(t, acc_bias, gyr_bias, gravity, vel, use_cache);
        (void)pose;

        for(int j = 0; j < 4; ++j)
        {
            Mat3 num_pos;
            Mat3 num_rot;
            for(int k = 0; k < 3; ++k)
            {
                Vec3 args[4] = {acc_bias, gyr_bias, gravity, vel};

                args[j][k] += eps;
                if(use_cache)
                {
                    state.computeCache(args[0], args[1], args[2], args[3]);
                }
                auto [pose_plus, jac_plus] = state.queryWthJacobian(t, args[0], args[1], args[2], args[3], use_cache);
                (void)jac_plus;

                args[j][k] -= 2.0*eps;
                if(use_cache)
                {
                    state.computeCache(args[0], args[1], args[2], args[3]);
                }
                auto [pose_minus, jac_minus] = state.queryWthJacobian(t, args[0], args[1], args[2], args[3], use_cache);
                (void)jac_minus;

                num_pos.col(k) = (pose_plus.first - pose_minus.first) / (2.0*eps);
                num_rot.col(k) = (pose_plus.second - pose_minus.second) / (2.0*eps);
            }

            reportJacobianBlock(std::string(block_names[j]) + " (pos)", jacobian[j].first, num_pos, worst_error);
            if(isRotationBlockSet(mode, j))
            {
                reportJacobianBlock(std::string(block_names[j]) + " (rot)", jacobian[j].second, num_rot, worst_error);
            }
            else
            {
                // The block is not filled in, so it cannot be compared. The numerical derivative is
                // still checked to be zero: that independence is what makes leaving it out valid.
                const double num_rot_max = num_rot.cwiseAbs().maxCoeff();
                std::cout << "      " << std::left << std::setw(26) << (std::string(block_names[j]) + " (rot)")
                          << " not set by queryWthJacobian, max|numerical| = "
                          << std::scientific << std::setprecision(3) << num_rot_max << std::endl;
                if(num_rot_max > 1e-9)
                {
                    std::cout << "        ^^ the rotation does depend on this block, it should be filled in" << std::endl;
                }
            }
        }

        // Restore the cache to the nominal parameters, the caller may rely on it
        if(use_cache)
        {
            state.computeCache(acc_bias, gyr_bias, gravity, vel);
        }
    }
    return worst_error;
}

// Compare the cached and the non-cached evaluation of the same quantities
static void testCacheConsistency(State& state, const LidarOdometryMode mode, const std::vector<double>& query_times,
        const Vec3& acc_bias, const Vec3& gyr_bias, const Vec3& gravity, const Vec3& vel)
{
    double worst_pose = 0.0;
    double worst_jacobian = 0.0;

    state.computeCache(acc_bias, gyr_bias, gravity, vel);
    for(const double t : query_times)
    {
        auto [pose_cache, jac_cache] = state.queryWthJacobian(t, acc_bias, gyr_bias, gravity, vel, true);
        auto [pose_plain, jac_plain] = state.queryWthJacobian(t, acc_bias, gyr_bias, gravity, vel, false);

        worst_pose = std::max(worst_pose, (pose_cache.first - pose_plain.first).cwiseAbs().maxCoeff());
        worst_pose = std::max(worst_pose, (pose_cache.second - pose_plain.second).cwiseAbs().maxCoeff());
        for(int j = 0; j < 4; ++j)
        {
            worst_jacobian = std::max(worst_jacobian, (jac_cache[j].first - jac_plain[j].first).cwiseAbs().maxCoeff());
            if(isRotationBlockSet(mode, j))
            {
                worst_jacobian = std::max(worst_jacobian, (jac_cache[j].second - jac_plain[j].second).cwiseAbs().maxCoeff());
            }
        }
    }

    std::cout << "      cached vs non-cached, pose     : max diff = " << std::scientific << std::setprecision(3) << worst_pose << std::endl;
    std::cout << "      cached vs non-cached, jacobian : max diff = " << worst_jacobian << std::endl;
}

// Check that query() and queryWthJacobian() return the same pose. The optimisation evaluates the
// residuals with the former and the jacobians with the latter, so any difference between the two
// means the jacobians do not describe the residuals being minimised.
static double testQueryConsistency(State& state, const bool use_cache, const std::vector<double>& query_times,
        const Vec3& acc_bias, const Vec3& gyr_bias, const Vec3& gravity, const Vec3& vel)
{
    double worst_pos = 0.0;
    double worst_rot = 0.0;

    if(use_cache)
    {
        state.computeCache(acc_bias, gyr_bias, gravity, vel);
    }
    for(const double t : query_times)
    {
        const std::pair<Vec3, Vec3> pose_query = state.query(t, acc_bias, gyr_bias, gravity, vel, use_cache);
        auto [pose_jacobian, jacobian] = state.queryWthJacobian(t, acc_bias, gyr_bias, gravity, vel, use_cache);
        (void)jacobian;

        worst_pos = std::max(worst_pos, (pose_query.first - pose_jacobian.first).cwiseAbs().maxCoeff());
        worst_rot = std::max(worst_rot, (pose_query.second - pose_jacobian.second).cwiseAbs().maxCoeff());
    }

    std::cout << "      " << (use_cache ? "with cache   " : "without cache")
              << " : max diff position = " << std::scientific << std::setprecision(3) << worst_pos
              << ", rotation = " << worst_rot << std::endl;

    const double worst = std::max(worst_pos, worst_rot);
    if(worst > 1e-12)
    {
        std::cout << "        ^^ MISMATCH between query() and queryWthJacobian()" << std::endl;
    }
    return worst;
}

void testState()
{
    std::cout << "================ State jacobian test ================" << std::endl;

    const double first_t = 10.0;
    const double duration = 0.5;
    const ugpm::ImuData imu_data = generateTestImuData(first_t, duration, 200.0);

    const Vec3 acc_bias(0.02, -0.03, 0.01);
    const Vec3 gyr_bias(0.004, -0.002, 0.003);
    const Vec3 gravity(0.05, -0.08, 9.81);
    const Vec3 vel(1.2, -0.4, 0.15);

    const std::vector<std::pair<std::string, LidarOdometryMode> > modes = {
        {"IMU", LidarOdometryMode::IMU},
        {"GYR", LidarOdometryMode::GYR},
        {"NO_IMU", LidarOdometryMode::NO_IMU},
    };

    double worst_error = 0.0;
    double worst_query_diff = 0.0;
    for(const auto& [mode_name, mode] : modes)
    {
        std::cout << "\n--- mode " << mode_name << " ---" << std::endl;
        State state(imu_data, first_t, 200.0, mode);

        // Sample inside the window, avoiding the very edges where the state index is clamped
        std::vector<double> query_times;
        for(int i = 1; i < 5; ++i)
        {
            query_times.push_back(first_t + 0.2*duration*i);
        }

        // computeCache indexes preint_meas_, which is empty without IMU preintegration
        const bool cache_supported = (mode != LidarOdometryMode::NO_IMU);

        std::cout << "  without caching:" << std::endl;
        worst_error = std::max(worst_error, testJacobiansOneConfig(state, mode, false, query_times, acc_bias, gyr_bias, gravity, vel));

        if(cache_supported)
        {
            std::cout << "  with caching:" << std::endl;
            worst_error = std::max(worst_error, testJacobiansOneConfig(state, mode, true, query_times, acc_bias, gyr_bias, gravity, vel));
        }
        else
        {
            std::cout << "  with caching: skipped (computeCache needs the preintegrated measurements)" << std::endl;
        }

        std::cout << "  query() against queryWthJacobian():" << std::endl;
        worst_query_diff = std::max(worst_query_diff, testQueryConsistency(state, false, query_times, acc_bias, gyr_bias, gravity, vel));
        if(cache_supported)
        {
            worst_query_diff = std::max(worst_query_diff, testQueryConsistency(state, true, query_times, acc_bias, gyr_bias, gravity, vel));

            std::cout << "  cache consistency:" << std::endl;
            testCacheConsistency(state, mode, query_times, acc_bias, gyr_bias, gravity, vel);
        }
    }

    std::cout << "\nWorst relative jacobian error over all modes    : " << std::scientific << std::setprecision(3) << worst_error << std::endl;
    std::cout << "Worst query()/queryWthJacobian() difference     : " << worst_query_diff << std::endl;
    std::cout << "================ End of the test ================" << std::endl;
}