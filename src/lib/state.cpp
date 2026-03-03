#include "lice/state.h"

#include "lice/utils.h"
#include "lice/math_utils.h"

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
        
        Mat3 R0;
        Mat3 R1;
        if(use_cache && !cached_state_poses_.empty())
        {
            R0 = cached_state_poses_[state_id].second;
            R1 = cached_state_poses_[state_id+1].second;
        }
        else
        {
             R0 = preint_meas_[state_id].delta_R * ugpm::expMap(preint_meas_[state_id].d_delta_R_d_bw * gyr_bias);
             R1 = preint_meas_[state_id+1].delta_R * ugpm::expMap(preint_meas_[state_id+1].d_delta_R_d_bw * gyr_bias);
        }

        query_pose.first = p0 + alpha * (p1 - p0);
        Vec3 delta_r = ugpm::logMap(R0.transpose() * R1);
        query_pose.second = ugpm::logMap(R0 * ugpm::expMap(delta_r * alpha));
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
        Mat3 R0;
        Mat3 R1;
        
        if(use_cache && !cached_state_poses_.empty())
        {
            R0 = cached_state_poses_[state_id].second;
            R1 = cached_state_poses_[state_id+1].second;

            state_jacobian_0 = cached_state_jacobians_[state_id];
            state_jacobian_1 = cached_state_jacobians_[state_id+1];

            state_R_shift_dw_0 = cached_state_R_shift_bw_[state_id];
            state_R_shift_dw_1 = cached_state_R_shift_bw_[state_id+1];
        }
        else
        {

            R0 = preint_meas_[state_id].delta_R * ugpm::expMap(preint_meas_[state_id].d_delta_R_d_bw * gyr_bias);
            R1 = preint_meas_[state_id+1].delta_R * ugpm::expMap(preint_meas_[state_id+1].d_delta_R_d_bw * gyr_bias);

            Vec3 dw_shift = gyr_bias;
            for(int j = 0; j < 3; ++j)
            {
                dw_shift[j] += eps;
                state_R_shift_dw_0[j] = preint_meas_[state_id].delta_R * ugpm::expMap(preint_meas_[state_id].d_delta_R_d_bw * dw_shift);
                state_R_shift_dw_1[j] = preint_meas_[state_id+1].delta_R * ugpm::expMap(preint_meas_[state_id+1].d_delta_R_d_bw * dw_shift);
                dw_shift[j] -= eps;
            }
        }

        query_pose.first = p0 + alpha * (p1 - p0);
        Vec3 delta_r = ugpm::logMap(R0.transpose() * R1);
        query_pose.second = ugpm::logMap(R0 * ugpm::expMap(delta_r * alpha));

        // Compute the jacobian
        for(int j = 0; j < 4; ++j)
        {
            query_jacobian[j].first = state_jacobian_0[j] + alpha * (state_jacobian_1[j] - state_jacobian_0[j]);

            if(j == 1)
            {
                for(int k = 0; k < 3; ++k)
                {
                    Vec3 r_shift;
                    if(use_cache && !cached_state_poses_.empty())
                    {
                        r_shift = ugpm::logMap(state_R_shift_dw_0[k] * ugpm::expMap( cached_delta_r_shift_bw_[state_id][k] * alpha));
                    }
                    else
                    {
                        r_shift = ugpm::logMap(state_R_shift_dw_0[k] * ugpm::expMap( ugpm::logMap(state_R_shift_dw_0[k].transpose() * state_R_shift_dw_1[k]) * alpha));
                    }
                    query_jacobian[j].second.col(k) = (r_shift - query_pose.second) / eps;
                }
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
    cached_state_R_shift_bw_.clear();
    cached_delta_r_shift_bw_.clear();
    for(int i = 0; i < nb_state_; ++i)
    {
        Vec3 p = preint_meas_[i].delta_p + preint_meas_[i].d_delta_p_d_bf * acc_bias + preint_meas_[i].d_delta_p_d_bw * gyr_bias + vel*preint_meas_[i].dt + gravity*preint_meas_[i].dt_sq_half;
        Mat3 R = preint_meas_[i].delta_R * ugpm::expMap(preint_meas_[i].d_delta_R_d_bw * gyr_bias);
        cached_state_poses_.push_back({p, R});


        std::array<Mat3, 4> state_jacobian;
        state_jacobian[0] = preint_meas_[i].d_delta_p_d_bf;
        state_jacobian[1] = preint_meas_[i].d_delta_p_d_bw;
        state_jacobian[2] = Mat3::Identity()*preint_meas_[i].dt_sq_half;
        state_jacobian[3] = Mat3::Identity()*preint_meas_[i].dt;
        cached_state_jacobians_.push_back(state_jacobian);

        Vec3 dw_shift = gyr_bias;
        std::array<Mat3, 3> R_shift_dw;
        for(int j = 0; j < 3; ++j)
        {
            dw_shift[j] += eps_;
            R_shift_dw[j] = preint_meas_[i].delta_R * ugpm::expMap(preint_meas_[i].d_delta_R_d_bw * dw_shift);
            dw_shift[j] -= eps_;
        }
        cached_state_R_shift_bw_.push_back(R_shift_dw);


        if(i > 0)
        {
            std::array<Vec3, 3> delta_r_shift_bw;
            for(int j = 0; j < 3; ++j)
            {
                delta_r_shift_bw[j] = ugpm::logMap(cached_state_R_shift_bw_.at(i-1)[j].transpose() * cached_state_R_shift_bw_.at(i)[j]);
            }
            cached_delta_r_shift_bw_.push_back(delta_r_shift_bw);
        }
    }
}

