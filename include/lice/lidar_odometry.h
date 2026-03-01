#pragma once

#include "types.h"
#include <mutex>
#include <thread>
#include "preint/preint.h"
#include "lice/utils.h"
#include "lice/cost_functions.h"
#include "lice/state.h"
#include "ankerl/unordered_dense.h"
#include "lice/pointcloud_utils.h"


#include <ceres/ceres.h>
#include <ceres/rotation.h>


const double kPullPeriod = 0.001;
const double kAssociationFilterAngQuantum = 1.0;
//const double kAssociationFilterLinQuantum = 0.75;

const int kMaxNbAssociationPerBin = 3;

struct LidarOdometryParams
{
    bool low_latency = false;
    bool dense_pc_output = false; // If true, also output dense point cloud
    double min_range = 1.0;
    double max_range = 150.0;
    double min_feature_dist = 0.05;
    double max_feature_dist = 0.5;
    double max_feature_range = 150.0;
    double feature_voxel_size = 0.3;
    double loss_function_scale = 0.25;
    uint32_t max_associations_per_type = 500;
    double state_frequency = 10;
    double g = 9.80;
    double calib_px = 0.0;
    double calib_py = 0.0;
    double calib_pz = 0.0;
    double calib_rx = 0.0;
    double calib_ry = 0.0;
    double calib_rz = 0.0;

    double gyr_std = 0.005;
    double acc_std = 0.02;
    double lidar_std = 0.02;

    double intensity_threshold = -1.0; // If > 0, only use points with intensity above this threshold for feature extraction

    double association_filter_lin_quantum = 0.45;

    bool unsorted_pc = false; // If true, the incoming point clouds are not sorted by time

    bool planar_only = false; // If true, only use planar features (no edge features)

    LidarOdometryMode mode = LidarOdometryMode::IMU;
};




class LidarOdometryPublisher
{
    public:
        virtual void publishTransform(const int64_t t, const Vec3& pos, const Vec3& rot) = 0;
        virtual void publishPc(const int64_t t, const std::vector<Pointd>& pc) = 0;
        virtual void publishGlobalOdom(const int64_t t, const Vec3& pos, const Vec3& rot, const Vec3& vel, const Vec3& ang_vel) = 0;
        virtual void publishPcDense(const int64_t t, const std::vector<Pointd>& pc) = 0;
        virtual void publishTwist(const int64_t t, const Vec3& linear, const Vec3& angular) = 0;
};





class LidarOdometry
{

    public:

        LidarOdometry(const LidarOdometryParams& params, LidarOdometryPublisher* node);

        void addPc(const std::shared_ptr<std::vector<Pointd>>& pc, const int64_t t);

        void addAccSample(const Vec3& acc, const int64_t t);
        void addGyroSample(const Vec3& gyro, const int64_t t);

        void stop();
        void run();
        std::shared_ptr<std::thread> runThread();

        void setIs2D(const bool is_2d)
        {
            is_2d_ = is_2d;
            if(is_2d_ && (params_.mode == LidarOdometryMode::IMU))
            {
                throw std::runtime_error("LidarOdometry::setIs2D: Cannot set is_2d flag when using IMU mode");
            }
        }


        // Destructor
        ~LidarOdometry()
        {
            delete loss_function_;
        }

    private:

        LidarOdometryParams params_;
        LidarOdometryPublisher* node_;

        // Flag to indicate if the node is running
        bool running_ = false;

        // Mutex to access the incoming data (point clouds and IMU)
        std::mutex mutex_;
        std::mutex imu_mutex_;
        std::mutex pc_mutex_;

        // Estimate the average scan time
        int64_t last_pc_time_ = -1;
        int64_t scan_time_sum_ = 0;
        int scan_count_ = 0;

        // Storing the incoming point clouds
        std::vector<std::shared_ptr<std::vector<Pointd> > > pc_;
        std::vector<int64_t> pc_t_;

        // Storing the incoming IMU data
        ugpm::ImuData imu_data_;
        // Storing an time offset to interact with the UGPM data structure
        bool first_ = true;
        int64_t imu_time_offset_ = 0;
        bool is_2d_ = false;

        // Storing the chunks of point clouds
        std::vector<std::shared_ptr<std::vector<Pointd> > > pc_chunks_;
        std::vector<std::shared_ptr<std::vector<Pointd> > > pc_chunk_features_;
        std::vector<std::shared_ptr<std::vector<Pointd> > > pc_chunk_features_sparse_;
        std::vector<int64_t> pc_chunks_t_;

        ////// Variables to perform scan splitting (from raw point cloud chunks)
        int64_t median_dt_ = -1;


        // Current odometry state
        Vec3 current_pos_ = Vec3::Zero();
        Vec3 current_rot_ = Vec3::Zero();
        std::mutex current_pose_mutex_;
        int64_t current_time_ = 0;

        // State variables (blocks in ceres)
        // acc_bias, gyr_bias, gravity, vel
        std::vector<Vec3> state_blocks_;
        double time_offset_ = 0.0;
        Vec7 state_calib_;

        // Optimisation related flags and objects
        bool first_optimisation_ = true;
        ceres::LossFunction* loss_function_;
        double lidar_weight_ = 1.0;

        State prev_state_;
        std::vector<Vec3> prev_state_blocks_;
        double prev_time_offset_ = 0.0;
        Vec7 prev_state_calib_;


        // For bias initialisation
        Vec3 acc_bias_sum_ = Vec3::Zero();
        Vec3 gyr_bias_sum_ = Vec3::Zero();
        int bias_count_ = 0;



        // Function to convert point cloud times to IMU data times
        double nanosToImuTime(const int64_t nanos) const;



        // Split the point cloud into chunks and downsample to get the features
        void splitAndFeatureExtraction(std::shared_ptr<std::vector<Pointd> > pc, const int64_t t);


        // Get the data for optimisation: features and IMU data
        std::tuple<std::vector<std::shared_ptr<std::vector<Pointd> > >, std::vector<std::shared_ptr<std::vector<Pointd> > >, ugpm::ImuData, int64_t, int64_t> getDataForOptimisation();

        // Perform the data association and optimisation
        void optimise();

        // Initialise the state
        void initState(const ugpm::ImuData& imu_data);


        // Sub function of optimise() that does the heavy lifting: data association and optimisation
        std::vector<DataAssociation> createProblemAssociateAndOptimise(
                const std::vector<std::shared_ptr<std::vector<Pointd> > >& pts
                , const std::vector<std::shared_ptr<std::vector<Pointd> > >& sparse_pts
                , const State& state
                , const std::set<int>& types
                , const int nb_iter=50
                , bool vel_only = false
                );


        // Helper function to project points using the continuous preintegrated state
        std::vector<std::shared_ptr<std::vector<Pointd> > > projectPoints(
            const std::vector<std::shared_ptr<std::vector<Pointd> > >& pts,
            const State& state,
            const std::vector<Vec3>& state_blocks,
            const double time_offset,
            const Vec7& state_calib) const;


        std::vector<DataAssociation> getDataAssociations(
                const std::set<int>& types
                , const std::vector<std::shared_ptr<std::vector<Pointd> > >& pts
                , const std::vector<std::shared_ptr<std::vector<Pointd> > >& sparse_pts);


        // Add the state blocks to the problem
        void addBlocks(ceres::Problem& problem, bool vel_only);

        // Add the lidar residuals to the problem
        void addLidarResiduals(
                ceres::Problem& problem
                , const std::vector<DataAssociation>& data_associations
                , const std::vector<std::shared_ptr<std::vector<Pointd> > >& pts
                , const std::vector<std::shared_ptr<std::vector<Pointd> > >& sparse_pts
                , const State& state
                );
        


        // Helper function to print the current state
        void printState();
        void logState();


        // Prepare the state blocks for the next optimisation
        void prepareNextState(const State& state);

        // Update the current pose
        void updateCurrentPose(const int64_t t0, const int64_t t1);

        // Publish the odometry results and launch the point cloud correction/publishing thread
        void publishResults(const State& state);

        // Correct the point cloud chunks and publish them
        void correctAndPublishPc(
            const std::vector<std::shared_ptr<std::vector<Pointd> > > pc_chunks,
            const std::vector<int64_t> pc_chunks_t,
            const State state,
            const std::vector<Vec3> state_blocks,
            const Vec7 state_calib,
            const double time_offset,
            const bool dense = false);



        // To check if useful        
        // Helper function to project points with R and t
        std::vector<std::shared_ptr<std::vector<Pointd> > > projectPoints(
            const std::vector<std::shared_ptr<std::vector<Pointd> > >& pts,
            const Vec3& pos,
            const Vec3& rot);

        void prepareSubmap(const State state, const std::vector<std::shared_ptr<std::vector<Pointd> > > pcs, const std::vector<double> pcs_t, std::vector<Vec3> state_blocks, Vec7 state_calib, double time_offset);


        // For DEBUG
        void visualiseDataAssociation(const std::vector<DataAssociation>& data_association);
        void visualisePoints(const std::vector<std::shared_ptr<std::vector<Pointd> > >& pts);

        void visualiseRawSubmap();
        void visualiseSubmap(const State& state);


};


