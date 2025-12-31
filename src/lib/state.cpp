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
        , const double dt
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

            R = preint.delta_R * ugpm::expMap(preint.d_delta_R_d_bw * gyr_bias + preint.d_delta_R_d_t * dt);
            if(mode_ == LidarOdometryMode::GYR)
            {
                p = vel * (state_time_.at(i) - start_t_);
            }
            else
            {
                p = preint.delta_p + preint.d_delta_p_d_bf * acc_bias + preint.d_delta_p_d_bw * gyr_bias + preint.d_delta_p_d_t * dt + vel*preint.dt + gravity*preint.dt_sq_half;
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
        , const double dt
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
            p0 = preint_meas_[state_id].delta_p + preint_meas_[state_id].d_delta_p_d_bf * acc_bias + preint_meas_[state_id].d_delta_p_d_bw * gyr_bias + preint_meas_[state_id].d_delta_p_d_t * dt + vel*preint_meas_[state_id].dt + gravity*preint_meas_[state_id].dt_sq_half;
            p1 = preint_meas_[state_id+1].delta_p + preint_meas_[state_id+1].d_delta_p_d_bf * acc_bias + preint_meas_[state_id+1].d_delta_p_d_bw * gyr_bias + preint_meas_[state_id+1].d_delta_p_d_t * dt + vel*preint_meas_[state_id+1].dt + gravity*preint_meas_[state_id+1].dt_sq_half;
        }
        const Mat3 R0 = preint_meas_[state_id].delta_R * ugpm::expMap(preint_meas_[state_id].d_delta_R_d_bw * gyr_bias + preint_meas_[state_id].d_delta_R_d_t * dt);
        const Mat3 R1 = preint_meas_[state_id+1].delta_R * ugpm::expMap(preint_meas_[state_id+1].d_delta_R_d_bw * gyr_bias + preint_meas_[state_id+1].d_delta_R_d_t * dt);

        query_pose.first = p0 + alpha * (p1 - p0);
        Vec3 delta_r = ugpm::logMap(R0.transpose() * R1);
        query_pose.second = ugpm::logMap(R0 * ugpm::expMap(delta_r * alpha));
    }

    return query_pose;
}

// Overload to query a single time
std::tuple<std::pair<Vec3, Vec3>,
        std::vector<std::pair<MatX, MatX> > > State::queryWthJacobian(
        const double query_time
        , const Vec3& acc_bias
        , const Vec3& gyr_bias
        , const Vec3& gravity
        , const Vec3& vel
        , const double dt
        ) const
{
    std::pair<Vec3, Vec3> query_pose;
    std::vector<std::pair<MatX, MatX> > query_jacobian(5);

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
        query_jacobian[4].first = Mat3::Zero();
        query_jacobian[4].second = Mat3::Zero();

    }
    else
    {
        // Get the pose at the state time
        std::vector<MatX> state_jacobian_0(5);
        std::vector<MatX> state_jacobian_1(5);


        std::vector<Mat3> state_R_shift_dw_0(3);
        std::vector<Mat3> state_R_shift_dw_1(3);

        double eps = 1e-6;


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
            state_jacobian_0[4] = Vec3::Zero();

            state_jacobian_1[0] = Mat3::Zero();
            state_jacobian_1[1] = Mat3::Zero();
            state_jacobian_1[2] = Mat3::Zero();
            state_jacobian_1[3] = Mat3::Identity()* (state_time_[state_id+1] - start_t_);
            state_jacobian_1[4] = Vec3::Zero();
        }
        else
        {
            p0 = preint_meas_[state_id].delta_p + preint_meas_[state_id].d_delta_p_d_bf * acc_bias + preint_meas_[state_id].d_delta_p_d_bw * gyr_bias + preint_meas_[state_id].d_delta_p_d_t * dt + vel*preint_meas_[state_id].dt + gravity*preint_meas_[state_id].dt_sq_half;
            p1 = preint_meas_[state_id+1].delta_p + preint_meas_[state_id+1].d_delta_p_d_bf * acc_bias + preint_meas_[state_id+1].d_delta_p_d_bw * gyr_bias + preint_meas_[state_id+1].d_delta_p_d_t * dt + vel*preint_meas_[state_id+1].dt + gravity*preint_meas_[state_id+1].dt_sq_half;

            state_jacobian_0[0] = preint_meas_[state_id].d_delta_p_d_bf;
            state_jacobian_0[1] = preint_meas_[state_id].d_delta_p_d_bw;
            state_jacobian_0[2] = Mat3::Identity()*preint_meas_[state_id].dt_sq_half;
            state_jacobian_0[3] = Mat3::Identity()*preint_meas_[state_id].dt;
            state_jacobian_0[4] = preint_meas_[state_id].d_delta_p_d_t;

            state_jacobian_1[0] = preint_meas_[state_id+1].d_delta_p_d_bf;
            state_jacobian_1[1] = preint_meas_[state_id+1].d_delta_p_d_bw;
            state_jacobian_1[2] = Mat3::Identity()*preint_meas_[state_id+1].dt_sq_half;
            state_jacobian_1[3] = Mat3::Identity()*preint_meas_[state_id+1].dt;
            state_jacobian_1[4] = preint_meas_[state_id+1].d_delta_p_d_t;
        }
        const Mat3 R0 = preint_meas_[state_id].delta_R * ugpm::expMap(preint_meas_[state_id].d_delta_R_d_bw * gyr_bias + preint_meas_[state_id].d_delta_R_d_t * dt);
        const Mat3 R1 = preint_meas_[state_id+1].delta_R * ugpm::expMap(preint_meas_[state_id+1].d_delta_R_d_bw * gyr_bias + preint_meas_[state_id+1].d_delta_R_d_t * dt);


        Mat3 state_R_shift_dt_0 = preint_meas_[state_id].delta_R * ugpm::expMap(preint_meas_[state_id].d_delta_R_d_bw * gyr_bias + preint_meas_[state_id].d_delta_R_d_t * (dt+eps));
        Mat3 state_R_shift_dt_1 = preint_meas_[state_id+1].delta_R * ugpm::expMap(preint_meas_[state_id+1].d_delta_R_d_bw * gyr_bias + preint_meas_[state_id+1].d_delta_R_d_t * (dt+eps));

        Vec3 dw_shift = gyr_bias;
        for(int j = 0; j < 3; ++j)
        {
            dw_shift[j] += eps;
            state_R_shift_dw_0[j] = preint_meas_[state_id].delta_R * ugpm::expMap(preint_meas_[state_id].d_delta_R_d_bw * dw_shift + preint_meas_[state_id].d_delta_R_d_t * dt);
            state_R_shift_dw_1[j] = preint_meas_[state_id+1].delta_R * ugpm::expMap(preint_meas_[state_id+1].d_delta_R_d_bw * dw_shift + preint_meas_[state_id+1].d_delta_R_d_t * dt);
            dw_shift[j] -= eps;
        }

        query_pose.first = p0 + alpha * (p1 - p0);
        Vec3 delta_r = ugpm::logMap(R0.transpose() * R1);
        query_pose.second = ugpm::logMap(R0 * ugpm::expMap(delta_r * alpha));

        // Compute the jacobian
        for(int j = 0; j < 5; ++j)
        {
            query_jacobian[j].first = state_jacobian_0[j] + alpha * (state_jacobian_1[j] - state_jacobian_0[j]);

            if(j == 1)
            {
                query_jacobian[j].second.resize(3,3);
                for(int k = 0; k < 3; ++k)
                {
                    Vec3 r_shift = ugpm::logMap(state_R_shift_dw_0[k] * ugpm::expMap( ugpm::logMap(state_R_shift_dw_0[k].transpose() * state_R_shift_dw_1[k]) * alpha));
                    query_jacobian[j].second.col(k) = (r_shift - query_pose.second) / eps;
                }
            }
            else if(j == 4)
            {
                Vec3 r_shift = ugpm::logMap(state_R_shift_dt_0 * ugpm::expMap( ugpm::logMap(state_R_shift_dt_0.transpose() * state_R_shift_dt_1) * alpha));
                query_jacobian[j].second = (r_shift - query_pose.second) / eps;
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
        , const double dt
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

            Mat3 R = preint.delta_R * ugpm::expMap(preint.d_delta_R_d_bw * gyr_bias + preint.d_delta_R_d_t * dt);
            Vec3 v;
            if(mode_ == LidarOdometryMode::GYR)
            {
                v = vel;
            }
            else
            {
                v = vel + preint.delta_v + preint.d_delta_v_d_bf * acc_bias + preint.d_delta_v_d_bw * gyr_bias + preint.d_delta_v_d_t * dt + gravity*preint.dt;
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



void testStateMonoJacobians(LidarOdometryMode mode)
{
    std::cout << "============== testStateJacobians (analytical then numerical)" << std::endl;

    // Create fake IMU data with sinuses
    ugpm::ImuData imu_data;
    imu_data.acc.resize(100);
    imu_data.gyr.resize(100);
    for(int i = 0; i < 100; ++i)
    {
        double t = i * 0.01;
        imu_data.acc[i].data[0] = std::sin(t);
        imu_data.acc[i].data[1] = std::cos(t);
        imu_data.acc[i].data[2] = std::sin(t);
        imu_data.gyr[i].data[0] = std::cos(t);
        imu_data.gyr[i].data[1] = std::sin(t);
        imu_data.gyr[i].data[2] = std::cos(t);
        imu_data.acc[i].t = t;
        imu_data.gyr[i].t = t;
    }

    // Create a state
    State state(imu_data, 0, 9, mode);

    Vec3 bf = Vec3::Random();
    Vec3 bw = Vec3::Random();
    Vec3 gravity = Vec3::Random();
    Vec3 vel = Vec3::Random();
    double dt = vel[0];
    vel = Vec3::Random();

    // Query the state
    double query_time = 0.085;
    auto [query_pose, query_jacobian] = state.queryWthJacobian(query_time, bf, bw, gravity, vel, dt);

    // Compute the jacobian numerically
    double eps = 1e-6;
    std::vector<std::pair<MatX, MatX> > query_jacobian_num(5);

    for(int i = 0; i < 5; ++i)
    {
        for(int j = 0; j < std::max(query_jacobian[i].first.cols(), query_jacobian[i].second.cols()); ++j)
        {
            Vec3 bf_eps = bf;
            Vec3 bw_eps = bw;
            Vec3 gravity_eps = gravity;
            Vec3 vel_eps = vel;
            double dt_eps = dt;
            switch(i)
            {
                case 0:
                    bf_eps[j] += eps;
                    break;
                case 1:
                    bw_eps[j] += eps;
                    break;
                case 2:
                    gravity_eps[j] += eps;
                    break;
                case 3:
                    vel_eps[j] += eps;
                    break;
                case 4:
                    dt_eps += eps;
                    break;
            }

            std::pair<Vec3, Vec3> query_pose_eps = state.query(query_time, bf_eps, bw_eps, gravity_eps, vel_eps, dt_eps);

            if(query_jacobian[i].first.size() > 0)
            {
                if(j == 0)
                {
                    query_jacobian_num[i].first.resize(3, query_jacobian[i].first.cols());
                }
                query_jacobian_num[i].first.col(j) = (query_pose_eps.first - query_pose.first) / eps;
            }
            if(query_jacobian[i].second.size() > 0)
            {
                if(j == 0)
                {
                    query_jacobian_num[i].second.resize(3, query_jacobian[i].second.cols());
                }
                query_jacobian_num[i].second.col(j) = (query_pose_eps.second - query_pose.second) / eps;
            }
        }
    }

    // Compare the jacobians
    std::cout << std::endl << std::endl;
    for(int j = 0; j < 5; ++j)
    {
        if(query_jacobian[j].first.size() > 0)
        {
            std::cout << "Jacobian query state " << j << " pos" << std::endl;
            std::cout << query_jacobian[j].first << std::endl;
            std::cout << "vs" << std::endl;
            std::cout << query_jacobian_num[j].first << std::endl;
        }
        else
        {
            std::cout << "Jacobian query state " << j << " pos" << std::endl;
            std::cout << "Empty" << std::endl;
        }
        if(query_jacobian[j].second.size() > 0)
        {
            std::cout << "Jacobian query state " << j << " rot" << std::endl;
            std::cout << query_jacobian[j].second << std::endl;
            std::cout << "vs" << std::endl;
            std::cout << query_jacobian_num[j].second << std::endl;
        }
        else
        {
            std::cout << "Jacobian query state " << j << " rot" << std::endl;
            std::cout << "Empty" << std::endl;
        }
    }



}
