#pragma once

#include "types.h"
#include "math_utils.h"
#include "map_distance_field.h"
#include <atomic>
#include <filesystem>
#include <mutex>
#include <thread>
#include <fstream>
#include "utils.h"


#include "preint/preint.h"
#include "lice/imu_window_estimator.h"


const double kMinNodeDist = 1.0;

// Default number of solver iterations of a registration, named so that a caller that only wants to
// pass the arguments after it does not have to repeat the value
const int kDefaultRegistrationIterations = 12;

// IMU noise standard deviations until setImuNoise is called (the odometry's defaults)
const double kDefaultImuAccStd = 0.02;     // m/s^2
const double kDefaultImuGyrStd = 0.005;    // rad/s


class SubmapManager
{
    public:
        SubmapManager(GpMapPublisher* publisher, const MapDistFieldOptions& options, const bool localization, const bool using_submaps, const double submap_length, const double submap_overlap, const std::string& map_path, const bool reverse_path=false, const double node_search_dist=20.0);
        ~SubmapManager();


        // Use the current map to register the points. `pts_cov` holds the position covariance of each
        // point, and is only read when the `use_point_covariances` option is set.
        Mat4 registerPts(const std::vector<Pointd>& pts, const Mat4& prior, const int64_t current_time, const bool approximate=false, const double loss_scale=0.5, const int max_iterations=kDefaultRegistrationIterations, const std::vector<Mat3>& pts_cov=std::vector<Mat3>(), const bool disable_odom_prior = false);

        // Add points to the current map (and next map if using submaps)
        void addPts(const std::vector<Pointd>& pts, const Mat4& pose, const int64_t time);

        // Save a scan to the scan folder if one is configured. Called by addPts, and directly by the
        // node when localizing (the scans are not added to the map in that case).
        void writeScan(const std::vector<Pointd>& pts, const int64_t time);

        // Scale of the scans as last estimated by the registration, 1.0 when the estimation is off
        double getScale() const { return (current_map_ != nullptr) ? current_map_->getScale() : 1.0; }

        // Set the weights of the odometry prior for the registrations to come. They are kept in the
        // options too, so that a submap created later starts with the same ones.
        void setOdomPriorWeights(const double weight_pos, const double weight_rot)
        {
            options_.odom_prior_weight_pos = weight_pos;
            options_.odom_prior_weight_rot = weight_rot;
            if(current_map_ != nullptr)
            {
                current_map_->setOdomPriorWeights(weight_pos, weight_rot);
            }
        }


        // The IMU samples, which may be given from another thread than the rest of the calls (they
        // only take the IMU buffer's own lock). A sample not newer than the last one of its stream
        // is dropped.
        void addGyrMeasurement(const Vec3& gyr, const int64_t time_ns);

        void addAccMeasurement(const Vec3& acc, const int64_t time_ns);

        // Noise standard deviations of the IMU samples, for the covariances of their preintegration
        void setImuNoise(const double acc_std, const double gyr_std);

        void addVelocity(const Vec3& vel, const int64_t time_ns);


        // Get the current map points
        std::vector<Pointd> getPts();


        // Query the distance field at the given points
        std::vector<double> queryDistField(const std::vector<Vec3>& query_pts);


        void writeMap();


        void set2D(const bool is_2d);

        // Sliding-window IMU estimator fed with the IMU samples and the registered poses (off until
        // enabled; call before the IMU samples start coming)
        void enableImuEstimator(const ImuWindowEstimatorOptions& options);
        // The final pose of a scan (after all its registrations), for the IMU estimator
        void addEstimatorPose(const Mat4& pose, const int64_t time_ns);
        // Null when not enabled
        const ImuWindowEstimator* imuEstimator() const { return imu_estimator_.get(); }

    private:
        GpMapPublisher* publisher_ = nullptr;
        MapDistFieldOptions options_;
        bool localization_ = false;
        double submap_length_ = -1.0;
        double submap_overlap_ = 0.1;
        bool using_submaps_ = false;
        std::string map_path_;
        bool reverse_path_ = false;
        bool is_2d_ = false;
        // Distance (in meters, along the path) over which the graph nodes are searched when looking
        // for the submap to switch to
        double node_search_dist_ = 20.0;

        std::shared_ptr<MapDistField> current_map_ = nullptr;
        std::vector<std::pair<int64_t, Mat4>> current_map_poses_;
        std::shared_ptr<MapDistField> next_map_ = nullptr;
        std::vector<std::pair<int64_t, Mat4>> next_map_poses_;
        //std::shared_ptr<MapDistField> previous_map_ = nullptr;
        int submap_counter_ = 0;
        int64_t last_registered_time_ = -1;

        int num_submaps_ = 0;
        std::vector<std::pair<Vec3, int>> graph_nodes_;
        std::vector<std::string> submap_paths_;

        int current_map_id_ = 0;
        int current_node_id_ = 0;

        Mat4 last_registered_pose_ = Mat4::Identity();

        double path_length_ = -1.0;
        double path_angle_change_ = 0.0;

        std::unique_ptr<ImuWindowEstimator> imu_estimator_;
        // The covariance of the last registration, for the IMU estimator (when it uses them)
        bool registration_cov_ = false;
        int64_t last_cov_time_ = -1;
        Mat6 last_cov_ = Mat6::Zero();

        // Guards `imu_data_`: the IMU samples are added from their own thread
        std::mutex imu_mutex_;
        ugpm::ImuData imu_data_;
        Vec3 gravity_ = Vec3::Zero();
        Vec3 bias_acc_ = Vec3::Zero();
        Vec3 bias_gyr_ = Vec3::Zero();
        // Time of the first sample of the stream received second, -1 until both have one
        std::atomic<int64_t> first_imu_time_ns_{-1};

        std::map<int64_t, Vec3> body_velocities_;
        std::vector<Mat4> imu_poses_;
        std::vector<int64_t> imu_times_;
        std::vector<Vec3> imu_velocities_;
        std::vector<ugpm::PreintMeas> preint_meas_vec_;
        double gravity_angle_std_ = -1.0;

        void attemptGravityBiasInit();

        void cleanBodyVelocities();

        GravityFactorFunctor* computeGravityFactor(const int64_t current_time);

        void writeCurrentSubmap();
};