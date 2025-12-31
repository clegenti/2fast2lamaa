#pragma once

#include "lice/types.h"
#include "preint/preint.h"




class State
{

    private:
        int nb_state_;
        std::vector<double> state_time_;
        std::vector<ugpm::PreintMeas> preint_meas_;
        double state_period_;
        double start_t_;
        LidarOdometryMode mode_ = LidarOdometryMode::IMU;

    public:

        State(const ugpm::ImuData& imu_data, const double first_t, const double state_freq, const LidarOdometryMode mode);
        State(){};


        std::vector<std::pair<Vec3, Vec3> > queryApprox(
                const std::vector<double>& query_time
                , const Vec3& acc_bias
                , const Vec3& gyr_bias
                , const Vec3& gravity
                , const Vec3& vel
                , const double dt
                ) const;


        // Overload to query a single time
        std::pair<Vec3, Vec3> query(
                const double query_time
                , const Vec3& acc_bias
                , const Vec3& gyr_bias
                , const Vec3& gravity
                , const Vec3& vel
                , const double dt
                ) const;

        // Overload to query a single time
        std::tuple<std::pair<Vec3, Vec3>,
                std::vector<std::pair<MatX, MatX> > > queryWthJacobian(
                const double query_time
                , const Vec3& acc_bias
                , const Vec3& gyr_bias
                , const Vec3& gravity
                , const Vec3& vel
                , const double dt
                ) const;


        // Query the linear (first) and angular (second) velocity at the query time
        std::pair<Vec3, Vec3> queryTwist(
                const double query_time
                , const Vec3& acc_bias
                , const Vec3& gyr_bias
                , const Vec3& gravity
                , const Vec3& vel
                , const double dt
                ) const;
};

void testStateMonoJacobians(LidarOdometryMode mode);
