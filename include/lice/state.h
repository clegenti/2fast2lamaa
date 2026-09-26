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

        std::vector<std::pair<Vec3, Vec3> > cached_state_poses_;
        std::vector<std::array<Mat3,4> > cached_state_jacobians_;
        std::vector<Mat3> cached_state_dr_dw_;
        //std::vector<std::array<Mat3,3> > cached_state_R_shift_bw_;
        //std::vector<std::array<Vec3,3> > cached_delta_r_shift_bw_;

        double eps_ = 1e-6;

        // Position and rotation at state time i
        std::pair<Vec3, Mat3> statePose(
                const int i
                , const Vec3& acc_bias
                , const Vec3& gyr_bias
                , const Vec3& gravity
                , const Vec3& vel
                ) const;

    public:

        State(const ugpm::ImuData& imu_data, const double first_t, const double state_freq, const LidarOdometryMode mode);
        State(){};


        std::vector<std::pair<Vec3, Vec3> > queryApprox(
                const std::vector<double>& query_time
                , const Vec3& acc_bias
                , const Vec3& gyr_bias
                , const Vec3& gravity
                , const Vec3& vel
                ) const;

        // Poses (position, rotation) at all the state times, and the index of the state time closest
        // to a query time: the pose without interpolation, for when a few milliseconds do not
        // matter (the data association, see LidarOdometry::projectPoints)
        std::vector<std::pair<Vec3, Mat3> > statePoses(
                const Vec3& acc_bias
                , const Vec3& gyr_bias
                , const Vec3& gravity
                , const Vec3& vel
                ) const;
        int closestStateId(const double query_time) const;


        // Overload to query a single time
        std::pair<Vec3, Vec3> query(
                const double query_time
                , const Vec3& acc_bias
                , const Vec3& gyr_bias
                , const Vec3& gravity
                , const Vec3& vel
                , const bool use_cache = false
                ) const;

        // Overload to query a single time
        std::tuple<std::pair<Vec3, Vec3>,
                std::array<std::pair<Mat3, Mat3>, 4> > queryWthJacobian(
                const double query_time
                , const Vec3& acc_bias
                , const Vec3& gyr_bias
                , const Vec3& gravity
                , const Vec3& vel
                , const bool use_cache = false
                ) const;


        // Query the linear (first) and angular (second) velocity at the query time
        std::pair<Vec3, Vec3> queryTwist(
                const double query_time
                , const Vec3& acc_bias
                , const Vec3& gyr_bias
                , const Vec3& gravity
                , const Vec3& vel
                ) const;

        void computeCache(const Vec3& acc_bias, const Vec3& gyr_bias, const Vec3& gravity, const Vec3& vel);
};


void testState();