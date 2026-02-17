#pragma once

#include "types.h"
#include "math_utils.h"
#include "map_distance_field.h"
#include <filesystem>
#include <thread>
#include <fstream>
#include "utils.h"


#include "preint/preint.h"


const double kMinNodeDist = 1.0;
const int kNumAdjacentNodesToCheck = 20;


class SubmapManager
{
    public:
        SubmapManager(GpMapPublisher* publisher, const MapDistFieldOptions& options, const bool localization, const bool using_submaps, const double submap_length, const double submap_overlap, const std::string& map_path, const bool reverse_path=false);
        ~SubmapManager();


        // Use the current map to register the points
        Mat4 registerPts(const std::vector<Pointd>& pts, const Mat4& prior, const int64_t current_time, const bool approximate=false, const double loss_scale=0.5, const int max_iterations=12);

        // Add points to the current map (and next map if using submaps)
        void addPts(const std::vector<Pointd>& pts, const Mat4& pose, const int64_t time);


        void addGyrMeasurement(const Vec3& gyr, const int64_t time_ns);

        void addAccMeasurement(const Vec3& acc, const int64_t time_ns);

        void addVelocity(const Vec3& vel, const int64_t time_ns);


        // Get the current map points
        std::vector<Pointd> getPts();


        // Query the distance field at the given points
        std::vector<double> queryDistField(const std::vector<Vec3>& query_pts);


        void writeMap();


        void set2D(const bool is_2d);

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

        ugpm::ImuData imu_data_;
        Vec3 gravity_ = Vec3::Zero();
        Vec3 bias_acc_ = Vec3::Zero();
        Vec3 bias_gyr_ = Vec3::Zero();
        uint64_t first_imu_time_ns_ = -1;

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