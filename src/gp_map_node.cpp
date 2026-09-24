#include "rclcpp/rclcpp.hpp"
#include "ros_utils.h"
#include "lice/utils.h"
#include "lice/math_utils.h"
#include "lice/pointcloud_utils.h"
#include "lice/submap_manager.h"

#include <memory>
#include <thread>
#include <mutex>
#include <deque>

#include "sensor_msgs/msg/point_cloud2.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "sensor_msgs/msg/imu.hpp"
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

#include "ankerl/unordered_dense.h"

#include "ffastllamaa/srv/query_dist_field.hpp"
#include "ffastllamaa/msg/submap_info.hpp"

#include <sys/stat.h>

#include <fstream>


bool folderExists(const std::string& folderPath) {
    struct stat info;
    if (stat(folderPath.c_str(), &info) != 0)
        return false; // Cannot access folder
    else if (info.st_mode & S_IFDIR) // S_IFDIR means it's a directory
        return true; // Folder exists
    else
        return false; // Path exists but it's not a folder
}

bool createFolder(const std::string& folderPath) {
    mode_t mode = 0755; // UNIX style permissions
    int ret = mkdir(folderPath.c_str(), mode);
    if (ret == 0)
        return true; // Folder created successfully
    return false; // Failed to create folder
}

// A gap between two consecutive scans longer than mean + kDropoutSigmaFactor*stdev is taken as a
// dropped frame. The statistics need a few samples before that test means anything, hence the minimum.
constexpr double kDropoutSigmaFactor = 2.0;
constexpr double kMinDropoutSamples = 10.0;

// How many of the latest scan-to-scan motions the velocity used to bridge a dropped frame is averaged
// over
constexpr size_t kScanVelMean = 4;

// The odometry prior weights are multiplied by the mean number of registered points over that many
// scans. A window rather than the current count alone, so that one unusually sparse or dense scan does
// not move the prior on its own.
constexpr size_t kPriorWeightWindow = 5;

// Power the point count is raised to before scaling the prior weights. 1.0 makes the weights
// proportional to the number of points; 0.5 is what keeps the prior's influence RELATIVE to the field
// block constant, since that block has one residual per point and the prior enters squared.
constexpr double kPriorWeightExponent = 1.0;

// Loss scales of the coarse-to-fine registration cascade, from the widest to the narrowest, and the
// number of iterations each of the coarse steps gets
const std::vector<double> kCoarseToFineLossScales = {10.0, 5.0, 2.0};
constexpr int kCoarseToFineIterations = 10;

class GpMapNode: public rclcpp::Node, public GpMapPublisher
{
    public:
        GpMapNode()
            : Node("gp_map")
        {
            // Eigen spawns its own OpenMP threads for large enough products, and the registration
            // calls into it from inside OpenMP regions of its own. The matrices here are small
            // enough that it should never trigger, but if it ever did the two levels would
            // oversubscribe the machine rather than share it, so it is held to one thread.
            Eigen::setNbThreads(1);

            // Read the parameters for options
            voxel_size_ = readRequiredFieldDouble(this, "voxel_size");
            MapDistFieldOptions options;
            options.cell_size = voxel_size_;
            // A non-positive factor disables the downsampling of the scans used for the registration
            downsample_size_ = readFieldDouble(this, "voxel_size_factor_for_registration", 2.0) * voxel_size_;
            if(downsample_size_ <= 0.0)
            {
                RCLCPP_INFO(this->get_logger(), "Registration downsampling disabled (voxel_size_factor_for_registration <= 0): the scans are registered as they come, and max_num_pts_for_registration does not apply");
            }
            options.neighborhood_size = readFieldInt(this, "neighbourhood_size",2.0);

            register_ = readFieldBool(this, "register", true);
            bool with_init_guess = readFieldBool(this, "with_init_guess", true);
            with_init_guess_ = with_init_guess;
            approximate_ = readFieldBool(this, "no_gp", false);
            use_edge_field_ = options.edge_field;

            // Each publish walks every cell of the map and recomputes its normal, while holding
            // map_mutex_, so it competes with (and blocks) the registration. The map is a debug
            // view rather than something the pipeline consumes, so it is published sparingly.
            map_publish_period_ = readFieldDouble(this, "map_publish_period", 2.0);

            options.free_space_carving_radius = readFieldDouble(this, "free_space_carving_radius", -1.0);

            localization_ = readFieldBool(this, "localization_only", false);
            // The first scan of a localization is registered with a coarse to fine schedule, to pull
            // an initial guess that may be far off onto the map. Set this when the guess is known to
            // be good: the first scan is then registered like any other one, which is both faster and
            // keeps the wide losses from dragging it away from a guess that was already right.
            can_trust_init_ = readFieldBool(this, "can_trust_init", false);

            max_nb_pts_ = readFieldInt(this, "max_num_pts_for_registration", 4000);

            options.free_space_carving = false;
            if (options.free_space_carving_radius > 0.0)
            {
                options.free_space_carving = true;
            }
            // Use the input odometry as a prior of the registration and not only as an initial
            // guess. The weights are the inverse of the standard deviation the odometry is trusted
            // with, separately for the translation (1/m) and the rotation (1/rad).
            options.use_odom_prior = readFieldBool(this, "use_odom_prior", false);
            use_odom_prior_ = options.use_odom_prior;
            // These are FACTORS, not absolute weights: they are multiplied by the mean number of
            // registered points over the last kPriorWeightWindow scans, so that the prior follows how
            // much the field block of the cost weighs. The effective weight is logged per scan.
            odom_prior_weight_pos_ = readFieldDouble(this, "odom_prior_weight_pos", 1.0);
            odom_prior_weight_rot_ = readFieldDouble(this, "odom_prior_weight_rot", 1.0);
            options.odom_prior_weight_pos = odom_prior_weight_pos_;
            options.odom_prior_weight_rot = odom_prior_weight_rot_;
            if(options.use_odom_prior && !with_init_guess)
            {
                RCLCPP_WARN(this->get_logger(), "use_odom_prior is set but there is no odometry input (with_init_guess is false): the prior will anchor the registration to the previous pose instead");
            }

            // Weight the registration with the per-point position covariance of the input cloud (the
            // 6 cov_xx..cov_zz fields), propagated through the distance field query. The robust loss
            // is then applied to the mahalanobis distance instead of the euclidean one.
            options.use_point_covariances = readFieldBool(this, "use_point_covariances", false);
            use_point_covariances_ = options.use_point_covariances;
            // The surface normal the covariance is projected onto moves with the pose, so the
            // weighting is refreshed at every evaluation point of the solver. Set to false to freeze
            // it at the initial guess instead.
            options.point_covariances_per_iteration = readFieldBool(this, "point_covariances_per_iteration", true);

            // Estimate a scale of the scans alongside the pose, for a front-end whose reconstruction
            // is not exactly metric. A prior keeps it from moving much between two consecutive scans.
            options.use_scale_optimization = readFieldBool(this, "use_scale_optimization", false);
            options.scale_prior_weight = readFieldDouble(this, "scale_prior_weight", 100.0);

            // Upper bound on the number of cells holding a GP weight block, -1 for no limit. The
            // blocks survive from one scan to the next as long as the map does not change, so this
            // is what trades the memory they take against the cost of rebuilding them.
            options.max_num_alpha_cells = readFieldInt(this, "max_num_alpha_cells", -1);

            // Threads of the OpenMP regions of the registration. The default was a hardcoded 8,
            // which oversubscribes a machine with fewer cores than that, so it is capped by what the
            // host actually has. `hardware_concurrency` is allowed to return 0 when it cannot tell,
            // in which case the old default stands.
            const unsigned int hw_threads = std::thread::hardware_concurrency();
            const int default_num_threads = (hw_threads > 0)
                ? std::min<int>(8, static_cast<int>(hw_threads))
                : 8;
            options.num_threads = std::max(1, readFieldInt(this, "num_threads", default_num_threads));
            RCLCPP_INFO(this->get_logger(), "Registration running on %d thread(s) (%u reported by the host)", options.num_threads, hw_threads);

            use_scale_optimization_ = options.use_scale_optimization;
            if(options.use_scale_optimization && !localization_)
            {
                // SubmapManager::addPts throws in that case, better to say so before the first scan
                RCLCPP_WARN(this->get_logger(), "use_scale_optimization is set outside of localization_only mode: adding points to the map is not implemented with an estimated scale and will throw on the first scan");
            }
            // Spotting a dropped frame from the gap between two scans only means something for an
            // input with a regular rate. An asynchronous one, such as the keyframes of a camera
            // front-end, has no meaningful scan interval and every longer gap would look like a
            // dropout, so the detection is turned off for those.
            use_frame_dropout_detection_ = readFieldBool(this, "use_frame_dropout_detection", true);

            double min_range = readRequiredFieldDouble(this, "min_range");
            options.min_range = min_range;
            options.max_range = readFieldDouble(this, "max_range", 1000.0);

            key_framing_ = readFieldBool(this, "key_framing", false);
            key_framing_dist_thr_ = readFieldDouble(this, "key_framing_dist_thr", 1.0);
            key_framing_rot_thr_ = readFieldDouble(this, "key_framing_rot_thr", 0.1);
            key_framing_time_thr_ = readFieldDouble(this, "key_framing_time_thr", 1.0);


            std::string map_path = readRequiredFieldString(this, "map_path");
            bool reverse_path = false;
            bool using_submaps = readFieldBool(this, "using_submaps", false);

            if(readFieldBool(this, "write_scans", false))
            {
                options.scan_folder = map_path;
                if(options.scan_folder.back() != '/')
                {
                    options.scan_folder += "/";
                }
                // Create the map_path if it does not exist
                if(!folderExists(map_path))
                {
                    if(!createFolder(map_path))
                    {
                        RCLCPP_ERROR(this->get_logger(), "Could not create folder: %s for map output", map_path.c_str());
                        throw std::runtime_error("Could not create folder for map output");
                        return;
                    }
                    RCLCPP_INFO(this->get_logger(), "Created folder: %s for map output", map_path.c_str());
                }
                options.scan_folder += "scans/";
                // Create the folder if it does not exist
                if(folderExists(options.scan_folder))
                {
                    // Remove the folder and its contents
                    std::filesystem::remove_all(options.scan_folder);
                }
                if(!createFolder(options.scan_folder))
                {
                    RCLCPP_ERROR(this->get_logger(), "Could not create folder: %s for scan output", options.scan_folder.c_str());
                    throw std::runtime_error("Could not create folder for scan output");
                    return;
                }
                RCLCPP_INFO(this->get_logger(), "Created folder: %s for scan output", options.scan_folder.c_str());
            }

            if(localization_)
            {
                if(using_submaps)
                {
                    reverse_path = readRequiredFieldBool(this, "reverse_path");
                }
                double init_pose_x = readFieldDouble(this, "init_pose_x", 0.0);
                double init_pose_y = readFieldDouble(this, "init_pose_y", 0.0);
                double init_pose_z = readFieldDouble(this, "init_pose_z", 0.0);
                double init_pose_rx = readFieldDouble(this, "init_pose_rx", 0.0);
                double init_pose_ry = readFieldDouble(this, "init_pose_ry", 0.0);
                double init_pose_rz = readFieldDouble(this, "init_pose_rz", 0.0);

                init_guess_ = Mat4::Identity();
                init_guess_.block<3,1>(0,3) = Vec3(init_pose_x, init_pose_y, init_pose_z);
                init_guess_.block<3,3>(0,0) = expMap(Vec3(init_pose_rx, init_pose_ry, init_pose_rz));
            }

            // If folder does not exist, create it
            if(!folderExists(map_path))
            {
                if(!createFolder(map_path))
                {
                    RCLCPP_ERROR(this->get_logger(), "Could not create folder: %s for map output", map_path.c_str());
                    return;
                }
                RCLCPP_INFO(this->get_logger(), "Created folder: %s for map output", map_path.c_str());
            }



            pc_type_internal_ = readFieldBool(this, "point_cloud_internal_type", true);

            // With the point covariances the residuals are mahalanobis distances, so the loss scale is
            // a number of standard deviations and not a distance in meters
            loss_scale_ = readFieldDouble(this, "loss_function_scale", use_point_covariances_ ? 1.0 : 5.0*voxel_size_/3.0);

            // Adaptive scene scale: the registration voxel and the loss scale of the fine registration
            // are tuned for a given size of scene, and a configuration meeting a much smaller one
            // downsamples structure it cannot spare and carries a loss that rejects nothing. When this
            // is on, both follow the median range of every scan, `adaptive_reference_range` being the
            // median the configured values were tuned for. The configured values stay the ceiling: the
            // mode only ever makes them finer, and never past `adaptive_min_ratio` of them.
            adaptive_scene_scale_ = readFieldBool(this, "adaptive_scene_scale", false);
            adaptive_reference_range_ = readFieldDouble(this, "adaptive_reference_range", 30.0);
            adaptive_min_ratio_ = readFieldDouble(this, "adaptive_min_ratio", 0.25);
            if(adaptive_scene_scale_)
            {
                if(adaptive_reference_range_ <= 0.0)
                {
                    RCLCPP_WARN(this->get_logger(), "adaptive_reference_range must be positive, the adaptive scene scale is disabled");
                    adaptive_scene_scale_ = false;
                }
                else
                {
                    adaptive_min_ratio_ = std::clamp(adaptive_min_ratio_, 0.0, 1.0);
                    RCLCPP_INFO(this->get_logger(), "Adaptive scene scale enabled: reference range %f m, minimum ratio %f (registration voxel %f m and loss scale %f at the reference range)", adaptive_reference_range_, adaptive_min_ratio_, downsample_size_, loss_scale_);
                }
            }
            
            // Write the first line of the trajectory file
            traj_path_ = map_path + "/trajectory.csv";
            createTrajectoryFile(traj_path_);

            std::string log_dir = map_path;
            if(!log_dir.empty() && (log_dir.back() == '/'))
            {
                log_dir.pop_back();
            }
            const size_t last_slash = log_dir.find_last_of("/\\");
            if(last_slash != std::string::npos)
            {
                std::string parent_dir = log_dir.substr(0, last_slash);
                std::string other_logs_dir = parent_dir + "/other_logs";
                if(!folderExists(other_logs_dir))
                {
                    createFolder(other_logs_dir);
                }
                processing_time_log_path_ = other_logs_dir + "/gp_map_processing_time.csv";
                std::ofstream timing_log_file(processing_time_log_path_, std::ios::out | std::ios::trunc);
                if(timing_log_file.is_open())
                {
                    timing_log_file << "time_ms,average_time_ms" << std::endl;
                    timing_log_file.close();
                }
            }

            // Create the ROS related objects
            if(with_init_guess)
            {
                pc_sub_.subscribe(this, "/points_input");
                pose_sub_.subscribe(this, "/pose_input");
                int queue_size = 20;
                sync_ = std::make_shared<message_filters::TimeSynchronizer<sensor_msgs::msg::PointCloud2, geometry_msgs::msg::TransformStamped>>(pc_sub_, pose_sub_, queue_size);
                sync_->registerCallback(std::bind(&GpMapNode::pcPriorCallback, this, std::placeholders::_1, std::placeholders::_2));
            }
            else
            {
                sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("/points_input", 1, std::bind(&GpMapNode::pcCallback, this, std::placeholders::_1));
            }
            map_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/map", 10);
            odom_map_correction_pub_ = this->create_publisher<geometry_msgs::msg::TransformStamped>("/odom_map_correction", 10);
            pose_pub_ = this->create_publisher<geometry_msgs::msg::TransformStamped>("/scan_to_map_pose", 10);
            map_publish_thread_ = std::make_unique<std::thread>(&GpMapNode::mapPublishThread, this);
            query_dist_field_srv_ = this->create_service<ffastllamaa::srv::QueryDistField>("/query_dist_field", std::bind(&GpMapNode::queryDistFieldCallback, this, std::placeholders::_1, std::placeholders::_2));


            gyr_sub_ = this->create_subscription<sensor_msgs::msg::Imu>("/gp_map/gyr", 10, std::bind(&GpMapNode::gyrCallback, this, std::placeholders::_1));
            acc_sub_ = this->create_subscription<sensor_msgs::msg::Imu>("/gp_map/acc", 10, std::bind(&GpMapNode::accCallback, this, std::placeholders::_1));
            twist_sub_ = this->create_subscription<geometry_msgs::msg::TwistStamped>("/twist", 10, std::bind(&GpMapNode::twistCallback, this, std::placeholders::_1));

            submap_info_pub_ = this->create_publisher<ffastllamaa::msg::SubmapInfo>("/submap_info", 10);


            // Create the map manager
            double submap_length = readFieldDouble(this, "submap_length", -1.0);
            double submap_overlap = readFieldDouble(this, "submap_overlap", 0.2);
            if(!localization_)
            {
                using_submaps = (submap_length > 0.0);
            }
            RCLCPP_INFO(this->get_logger(), "Using submaps: %s", using_submaps ? "true" : "false");

            // Distance (in meters, along the path) over which the graph nodes are searched to decide
            // which submap to switch to during localization
            double submap_node_search_dist = readFieldDouble(this, "submap_node_search_dist", 20.0);

            options.use_temporal_weights = submap_length <= 0.0; // If not using submaps, use temporal weights by default
            map_ = std::make_shared<SubmapManager>(this, options, localization_, using_submaps, submap_length, submap_overlap, map_path, reverse_path, submap_node_search_dist);

        }



        void publishSubmapInfo(const std::string& filename, const Vec3& gravity)
        {
            auto msg = ffastllamaa::msg::SubmapInfo();
            msg.ply_file = filename;
            // Get the filename only and the folder path
            size_t last_slash_idx = filename.find_last_of("\\/");
            std::string folder_path = filename.substr(0, last_slash_idx);
            if (std::string::npos != last_slash_idx)
            {
                std::string filename_only = filename.substr(last_slash_idx + 1);
                std::string traj_filename = "trajectory_" + filename_only.replace(filename_only.find(".ply"), 4, ".csv");
                msg.scan_folder = folder_path + "/scans";
                msg.traj_file = folder_path + "/" + traj_filename;
            }
            else
            {
                msg.traj_file = "";
                msg.scan_folder = "";
            }
            msg.raw_output_folder = folder_path;
            msg.map_res = voxel_size_;
            msg.gravity = {gravity[0], gravity[1], gravity[2]};
            submap_info_pub_->publish(msg);
        }




        ~GpMapNode()
        {
            running_ = false;
            map_publish_thread_->join();
        }

    private:
        std::shared_ptr<SubmapManager> map_ = nullptr;
        double map_publish_period_ = 1.0;
        bool key_framing_ = false;
        double key_framing_dist_thr_ = 1.0;
        double key_framing_rot_thr_ = 0.1;
        double key_framing_time_thr_ = 1.0;



        size_t max_nb_pts_ = 4000;
        double voxel_size_ = 0.2;

        std::string traj_path_ = "";
        std::string processing_time_log_path_ = "";

        bool localization_ = false;
        bool use_edge_field_ = true;
        bool use_point_covariances_ = false;
        bool use_scale_optimization_ = false;
        bool can_trust_init_ = false;

        std::mutex map_mutex_;


        // Sub for time synchronised init_guess
        message_filters::Subscriber<sensor_msgs::msg::PointCloud2> pc_sub_;
        message_filters::Subscriber<geometry_msgs::msg::TransformStamped> pose_sub_;
        std::shared_ptr<message_filters::TimeSynchronizer<sensor_msgs::msg::PointCloud2, geometry_msgs::msg::TransformStamped>> sync_;
        // Sub for no init_guess
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
        // Global map publisher
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_pub_;
        rclcpp::Publisher<geometry_msgs::msg::TransformStamped>::SharedPtr odom_map_correction_pub_;
        rclcpp::Publisher<geometry_msgs::msg::TransformStamped>::SharedPtr pose_pub_;
        // Service to query the distance field
        rclcpp::Service<ffastllamaa::srv::QueryDistField>::SharedPtr query_dist_field_srv_;

        rclcpp::Publisher<ffastllamaa::msg::SubmapInfo>::SharedPtr submap_info_pub_;


        // Subscriber for the IMU data
        rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr gyr_sub_;
        rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr acc_sub_;
        
        // Subscriber for the velocities (twist)
        rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr twist_sub_;


        Mat4 current_pose_ = Mat4::Identity();
        
        Mat4 last_input_pose_ = Mat4::Identity();
        Mat4 init_guess_ = Mat4::Identity();
        bool first_ = true;

        bool register_ = true;
        double loss_scale_ = 0.5;

        bool approximate_ = false;
        bool with_init_guess_ = false;

        double downsample_size_ = 0.4;

        // Adaptive scene scale, see the parameter read in the constructor
        bool adaptive_scene_scale_ = false;
        double adaptive_reference_range_ = 30.0;
        double adaptive_min_ratio_ = 0.25;

        std::atomic<bool> running_ = true;
        std::atomic<int> counter_ = 0;
        int previous_counter_ = 0;

        int last_write_counter_ = 0;

        // Running average of the per-scan processing time, logged alongside each scan's own time
        double total_processing_time_ms_ = 0.0;
        size_t nb_processed_scans_ = 0;

        bool pc_type_internal_ = false;
        rclcpp::Time last_pc_time_;
        double key_framing_time_cumulated_ = 0.0;
        double key_framing_dist_cumulated_ = 0.0;

        // Running mean and variance (Welford) of the delta time between two consecutive incoming
        // scans, used to spot a dropped frame. The flag is kept until it has actually triggered a
        // registration: a dropout noticed on a scan that is not a keyframe still leaves the next
        // registered scan with a longer gap than usual to cover.
        double scan_dt_count_ = 0.0;
        double scan_dt_mean_ = 0.0;
        double scan_dt_m2_ = 0.0;
        bool pending_frame_dropout_ = false;
        bool use_frame_dropout_detection_ = true;
        bool use_odom_prior_ = false;

        // The last kScanVelMean trusted odometry increments, each with the interval it spanned. Their
        // moving average is the velocity the motion over a dropped frame is extrapolated at, so that one
        // noisy increment does not decide the prediction on its own.
        struct MotionSample
        {
            Vec3 translation;
            Vec3 rotation;
            double dt;
        };
        std::deque<MotionSample> recent_motions_;

        // Factors the odometry prior weights are given as, and the number of points the registration
        // actually saw over the last kPriorWeightWindow scans, whose mean scales them
        double odom_prior_weight_pos_ = 1.0;
        double odom_prior_weight_rot_ = 1.0;
        std::deque<size_t> recent_nb_pts_;


        std::unique_ptr<std::thread> map_publish_thread_;

        // Store the last point cloud time
        std::atomic<std::chrono::time_point<std::chrono::high_resolution_clock>> last_pc_epoch_time_;

        void queryDistFieldCallback(const std::shared_ptr<ffastllamaa::srv::QueryDistField::Request> request, std::shared_ptr<ffastllamaa::srv::QueryDistField::Response> response)
        {
            if(request->dim != 3)
            {
                RCLCPP_ERROR(this->get_logger(), "Only 3D points are supported");
                return;
            }
            std::vector<Vec3> query_pts;
            for(size_t i = 0; i < request->num_pts; i++)
            {
                query_pts.push_back(Vec3(request->pts.at(i*3), request->pts.at(i*3+1), request->pts.at(i*3+2)));
            }
            map_mutex_.lock();
            StopWatch sw;
            sw.start();
            std::vector<double> dists = map_->queryDistField(query_pts);
            double temp_time = sw.stop();
            map_mutex_.unlock();
            RCLCPP_INFO(this->get_logger(), "Query time (API) with %d points: %f ms", request->num_pts, temp_time);
            for(double dist: dists)
            {
                response->dists.push_back(dist);
            }
        }



        // Standard deviation of the delta time between consecutive scans, from the accumulator
        double scanIntervalStdev() const
        {
            return (scan_dt_count_ > 1.0) ? std::sqrt(scan_dt_m2_/(scan_dt_count_ - 1.0)) : 0.0;
        }

        // Fold one delta time into the running mean and variance. The outliers are folded in as well,
        // so that a genuine change of scan rate is eventually followed instead of being reported as a
        // dropout forever; with enough samples one long gap barely moves the mean.
        void updateScanIntervalStats(const double scan_dt)
        {
            scan_dt_count_ += 1.0;
            const double delta = scan_dt - scan_dt_mean_;
            scan_dt_mean_ += delta/scan_dt_count_;
            scan_dt_m2_ += delta*(scan_dt - scan_dt_mean_);
        }

        // Is the gap since the previous scan long enough to call it a dropped frame? Tested against the
        // statistics of the scans before this one, which are then updated with it.
        bool detectFrameDropout(const double scan_dt)
        {
            //const double threshold = scan_dt_mean_ + kDropoutSigmaFactor*scanIntervalStdev();
            const double threshold = 1.4*scan_dt_mean_;
            const bool dropout = (scan_dt_count_ >= kMinDropoutSamples) && (scan_dt > threshold);
            if(dropout)
            {
                RCLCPP_WARN(this->get_logger(), "Dropped frame: %.1f ms since the last scan, over the %.1f ms threshold (mean %.1f ms, stdev %.1f ms): registering coarse-to-fine",
                        scan_dt*1e3, threshold*1e3, scan_dt_mean_*1e3, scanIntervalStdev()*1e3);
            }
            updateScanIntervalStats(scan_dt);
            return dropout;
        }

        // Coarse-to-fine cascade: registrations with a shrinking loss scale, each starting from the
        // result of the previous one. The wide loss of the early steps pulls in from further away than
        // the single fine registration can, which is what a bad initial guess needs. The caller holds
        // map_mutex_, and the fine registration is left to it.
        Mat4 registerCoarseToFine(const std::vector<Pointd>& pts, const Mat4& prior,
                                  const int64_t time_ns, const std::vector<Mat3>& pts_cov = std::vector<Mat3>(),  bool disable_odom_prior = false)
        {
            Mat4 pose = prior;
            for(const double coarse_loss_scale : kCoarseToFineLossScales)
            {
                pose = map_->registerPts(pts, pose, time_ns, true, coarse_loss_scale,
                                         kCoarseToFineIterations, pts_cov, disable_odom_prior);
            }
            return pose;
        }

        // Scale the odometry prior weights by the mean number of registered points over the last few
        // scans, and hand them to the map. `odom_prior_weight_pos/rot` are the factors, so a scan of
        // 300 points and one of 5000 do not weigh the prior against the field block the same way.
        // Called with the count the registration is about to see, i.e. after the downsampling.
        void updatePriorWeights(const size_t nb_pts)
        {
            if(!use_odom_prior_)
            {
                return;
            }
            recent_nb_pts_.push_back(nb_pts);
            while(recent_nb_pts_.size() > kPriorWeightWindow)
            {
                recent_nb_pts_.pop_front();
            }
            double sum = 0.0;
            for(const size_t count : recent_nb_pts_)
            {
                sum += static_cast<double>(count);
            }
            const double mean_nb_pts = sum/static_cast<double>(recent_nb_pts_.size());
            const double scale = std::pow(std::max(mean_nb_pts, 1.0), kPriorWeightExponent);
            map_->setOdomPriorWeights(odom_prior_weight_pos_*scale, odom_prior_weight_rot_*scale);
            RCLCPP_INFO(this->get_logger(), "Odometry prior weights: %.3f pos, %.3f rot (factors %.3f and %.3f scaled by %.1f points averaged over %zu scan(s))",
                    odom_prior_weight_pos_*scale, odom_prior_weight_rot_*scale,
                    odom_prior_weight_pos_, odom_prior_weight_rot_, mean_nb_pts,
                    recent_nb_pts_.size());
        }

        void updateMap(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg, const Mat4 trans)
        {
            StopWatch sw;
            StopWatch sw2;
            sw.start();


            rclcpp::Time time(msg->header.stamp);
            bool add_to_map = false;

            // Initialize on the first point cloud
            if(first_)
            {
                last_pc_time_ = msg->header.stamp;
                last_input_pose_ = trans;
                add_to_map = true;
            }
            // Check if the point cloud is too old
            if(time < last_pc_time_)
            {
                RCLCPP_WARN(this->get_logger(), "Time diff is negative, skipping point cloud");
                return;
            }

            // Check if the map need to be updated
            bool dropout_now = false;
            const double scan_dt = first_ ? 0.0 : (time - last_pc_time_).seconds();
            if(!first_)
            {
                // A dropped frame leaves more motion than usual between two scans, so the initial
                // guess is further from the solution than the fine registration alone can recover
                if(use_frame_dropout_detection_)
                {
                    dropout_now = detectFrameDropout(scan_dt);
                    pending_frame_dropout_ = pending_frame_dropout_ || dropout_now;
                }

                // Check if we need to update the map
                add_to_map = needMapUpdate(time, trans);
            }

            if(dropout_now)
            {
                // The odometry increment spanning the gap is the one that is likely wrong, so the motion
                // is predicted at the average velocity of the last few scans instead. Only this
                // increment is replaced: the ones that follow span no gap and are kept as they are.
                // `last_input_pose_` is advanced at the end of this function either way, so the discarded
                // increment is not composed in later.
                const Mat4 predicted_delta = predictConstantVelocity(scan_dt);
                init_guess_ = init_guess_*predicted_delta;
                RCLCPP_WARN(this->get_logger(), "Dropped frame: replacing the odometry increment by a constant-velocity prediction over %.1f ms, from the mean velocity of the last %zu scan(s): %.3f m, %.2f deg",
                        scan_dt*1e3, recent_motions_.size(),
                        predicted_delta.block<3,1>(0,3).norm(),
                        logMap(Mat3(predicted_delta.block<3,3>(0,0))).norm()*180.0/M_PI);
            }
            else
            {
                updateInitGuess(trans, scan_dt);
            }


            if(add_to_map)
            {
                // First convert the point cloud message to a vector of points, with the per-point
                // covariances when the registration is set to use them
                std::vector<Mat3> covs;
                auto [pts, is_2d] = getPcFromMsg(msg, use_point_covariances_ ? &covs : nullptr);
                if(use_point_covariances_ && covs.empty() && (pts.size() > 0))
                {
                    RCLCPP_WARN_ONCE(this->get_logger(), "use_point_covariances is set but the incoming clouds carry no cov_xx..cov_zz fields: registering without them");
                }
                int original_pts_size = pts.size();
                filterScan(pts, covs);

                // Scale of the scene, from which the registration voxel and the loss scale of the fine
                // registration follow. Both stay at their configured value when the mode is off.
                auto [scan_downsample_size, scan_loss_scale] = getAdaptiveRegistrationSettings(pts);

                if(is_2d)
                {
                    map_mutex_.lock();
                    map_->set2D(true);
                    map_mutex_.unlock();
                }

                if(localization_ && first_)
                {
                    // Downsample the points
                    std::vector<Mat3> downsampled_covs;
                    std::vector<Pointd> downsampled_pts = downsampleScan(pts, covs, downsampled_covs, false, true, scan_downsample_size);

                    map_mutex_.lock();
                    updatePriorWeights(downsampled_pts.size());
                    // The fine registration below starts from whatever this leaves in current_pose_:
                    // the guess itself when it is trusted, the result of the cascade otherwise
                    current_pose_ = init_guess_;
                    if(!can_trust_init_)
                    {
                        // Coarse to fine, to pull an initial guess that can be far off onto the map.
                        // The wide losses let the registration travel a long way, which is only worth
                        // its cost when the guess is not to be trusted.
                        current_pose_ = registerCoarseToFine(downsampled_pts, init_guess_, getTimeNs(time), downsampled_covs);
                    }
                    current_pose_ = map_->registerPts(downsampled_pts, current_pose_, getTimeNs(time), approximate_, scan_loss_scale, kDefaultRegistrationIterations, downsampled_covs);
                    init_guess_ = current_pose_;
                    map_mutex_.unlock();
                }
                else if(register_ && !first_)
                {
                    sw2.start();

                    // Downsample the points
                    std::vector<Mat3> downsampled_covs;
                    std::vector<Pointd> downsampled_pts = downsampleScan(pts, covs, downsampled_covs, use_edge_field_, false, scan_downsample_size);


                    map_mutex_.lock();
                    updatePriorWeights(downsampled_pts.size());
                    if(!with_init_guess_)
                    {
                        current_pose_ = map_->registerPts(downsampled_pts, current_pose_, getTimeNs(time), true, 10.0*loss_scale_, kDefaultRegistrationIterations, downsampled_covs);
                        init_guess_ = current_pose_;
                    }
                    // After a dropped frame, walk the loss scale down before the fine registration, the
                    // same way the very first scan is registered, rather than trusting an initial guess
                    // that has a longer gap than usual to cover
                    if(pending_frame_dropout_)
                    {
                        // The guess is deliberately stale here, so the odometry prior would anchor the
                        // solution to the very pose the cascade is trying to move away from
                        if(use_odom_prior_)
                        {
                            RCLCPP_WARN(this->get_logger(), "Recovering from a dropped frame: the odometry prior is disabled for the coarse-to-fine registration, as the guess it would anchor to is the pose before the gap");
                        }
                        init_guess_ = registerCoarseToFine(downsampled_pts, init_guess_, getTimeNs(time), downsampled_covs, true);
                        pending_frame_dropout_ = false;
                    }
                    //current_pose_ = map_->registerPts(downsampled_pts, init_guess_, getTimeNs(time), true, 2*loss_scale_, 7);
                    current_pose_ = map_->registerPts(downsampled_pts, init_guess_, getTimeNs(time), approximate_, scan_loss_scale, 25, downsampled_covs);
                    init_guess_ = current_pose_;
                    map_mutex_.unlock();

                    double temp_time = sw2.stop();
                    RCLCPP_INFO(this->get_logger(), "Registration time: %f ms", temp_time);
                    if(use_scale_optimization_)
                    {
                        RCLCPP_INFO(this->get_logger(), "Estimated scan scale: %f", map_->getScale());
                    }

                }
                else
                {
                    // Without registration the pose is dead reckoned from the input odometry. When
                    // localizing, that odometry lives in its own frame, which is not the map frame:
                    // `init_guess_` is the same motion composed on top of the initial map pose, so it
                    // is the one expressed in the map. In mapping mode the two frames are the same and
                    // the input pose is used as it always was.
                    current_pose_ = localization_ ? init_guess_ : trans;
                }
                publishPose(time, current_pose_);
                // Published whatever produced `current_pose_`, so that the map to odom transform is
                // available to the rest of the system even when the registration is disabled
                publishOdomMapCorrection(time, trans);



                map_mutex_.lock();
                if(!localization_ && add_to_map)
                {
                    map_->addPts(pts, current_pose_, getTimeNs(time));
                }
                map_mutex_.unlock();

                if(localization_)
                {
                    // The scans are not added to the map when localizing, but they are still written
                    // if `write_scans` is set, to inspect their alignment with the map afterwards.
                    // Outside of the mutex, writeScan only copies the points for the writing thread.
                    map_->writeScan(pts, getTimeNs(time));
                }


                counter_++;
                last_pc_epoch_time_ = std::chrono::high_resolution_clock::now();
            }



            // Log the pose to the trajectory file
            logPoseToFile(traj_path_, init_guess_, time);


            double time_ms = sw.stop();
            total_processing_time_ms_ += time_ms;
            nb_processed_scans_++;
            const double average_time_ms = total_processing_time_ms_ / nb_processed_scans_;
            RCLCPP_INFO(this->get_logger(), "Total time to process point cloud: %f ms (average: %f ms)", time_ms, average_time_ms);

            if(!processing_time_log_path_.empty())
            {
                std::ofstream timing_log_file(processing_time_log_path_, std::ios::out | std::ios::app);
                if(timing_log_file.is_open())
                {
                    timing_log_file << std::fixed << time_ms << "," << average_time_ms << std::endl;
                    timing_log_file.close();
                }
                else
                {
                    RCLCPP_WARN(this->get_logger(), "Could not open gp_map processing time log: %s", processing_time_log_path_.c_str());
                }
            }



            last_input_pose_ = trans;
            last_pc_time_ = msg->header.stamp;
            first_ = false;
        }



        void publishOdomMapCorrection(const rclcpp::Time& time, const Mat4& trans)
        {
            Mat4 odom_map_correction = current_pose_ * trans.inverse();
            geometry_msgs::msg::TransformStamped odom_map_correction_msg;
            odom_map_correction_msg.header.stamp = time;
            odom_map_correction_msg.header.frame_id = "map";
            odom_map_correction_msg.child_frame_id = "odom";
            odom_map_correction_msg.transform = mat4ToTransform(odom_map_correction);
            odom_map_correction_pub_->publish(odom_map_correction_msg);
        }

        void publishPose(const rclcpp::Time& time, const Mat4& trans)
        {
            // Map-corrected pose of the IMU/body frame (the input point clouds are expressed in
            // that frame, not in the physical lidar one)
            geometry_msgs::msg::TransformStamped pose_msg;
            pose_msg.header.stamp = time;
            pose_msg.header.frame_id = "map";
            pose_msg.child_frame_id = "imu";
            pose_msg.transform = mat4ToTransform(trans);
            pose_pub_->publish(pose_msg);
        }

        void updateInitGuess(const Mat4& trans, const double scan_dt)
        {
            Mat4 delta_trans = last_input_pose_.inverse() * trans;
            // The estimated scale is the scale of the reconstruction the odometry itself comes from:
            // the landmark depths and the trajectory translation share it. Scaling the points but not
            // the motion would leave the guess short by (s-1)*|delta| every scan, along the direction
            // of travel, and the odometry prior (a zero-prior on the correction) would fight the
            // registration correcting it. The rotation is scale-invariant.
            if(use_scale_optimization_)
            {
                delta_trans.block<3,1>(0,3) *= map_->getScale();
            }
            init_guess_ = init_guess_*delta_trans;

            // Keep it among the trusted motions, to extrapolate from should the next scan arrive after
            // a gap
            if(scan_dt > 0.0)
            {
                const Mat3 delta_rot = delta_trans.block<3,3>(0,0);
                recent_motions_.push_back({delta_trans.block<3,1>(0,3), logMap(delta_rot), scan_dt});
                while(recent_motions_.size() > kScanVelMean)
                {
                    recent_motions_.pop_front();
                }
            }
        }

        // Motion over `scan_dt` at the average velocity of the last trusted increments: their total
        // rotation and translation over their total time, integrated over the gap. Used in place of the
        // odometry increment spanning a dropped frame, which is the one not to be trusted. Identity
        // while there is nothing to extrapolate from yet, which leaves the guess where it was.
        //
        // The increments are each expressed in their own body frame, so averaging them assumes the
        // orientation does not change much over the window, which holds for the few scans it spans.
        Mat4 predictConstantVelocity(const double scan_dt) const
        {
            Mat4 delta = Mat4::Identity();
            if(recent_motions_.empty() || (scan_dt <= 0.0))
            {
                return delta;
            }
            Vec3 total_translation = Vec3::Zero();
            Vec3 total_rotation = Vec3::Zero();
            double total_dt = 0.0;
            for(const MotionSample& motion : recent_motions_)
            {
                total_translation += motion.translation;
                total_rotation += motion.rotation;
                total_dt += motion.dt;
            }
            if(total_dt <= 0.0)
            {
                return delta;
            }
            const double ratio = scan_dt/total_dt;
            delta.block<3,3>(0,0) = expMap(Vec3(total_rotation*ratio));
            delta.block<3,1>(0,3) = total_translation*ratio;
            return delta;
        }
        

        bool needMapUpdate(const rclcpp::Time& time, const Mat4& trans)
        {
            if(!key_framing_)
            {
                return true; // No key framing, always update
            }

            bool need_update = false;
            // Update to pose init_guess if there is registering
            Mat4 delta_trans = last_input_pose_.inverse() * trans;
            // Check if we need to register the point cloud if key framing is enabled
            if(key_framing_)
            {
                double time_diff = (rclcpp::Time(time) - rclcpp::Time(last_pc_time_)).seconds();
                key_framing_time_cumulated_ += time_diff;
                key_framing_dist_cumulated_ += delta_trans.block<3, 1>(0, 3).norm();
                if(key_framing_time_cumulated_ >= key_framing_time_thr_ || key_framing_dist_cumulated_ >= key_framing_dist_thr_)
                {
                    need_update = true;
                }

                auto [dist, rot_diff] = distanceBetweenTransforms(current_pose_, init_guess_);
                if( dist >= key_framing_dist_thr_ || rot_diff >= key_framing_rot_thr_)
                {
                    need_update = true;
                }
            }   
            if(need_update)
            {
                key_framing_time_cumulated_ = 0.0;
                key_framing_dist_cumulated_ = 0.0;
            }
            return need_update;
        }

        // Read the incoming cloud. `covariances`, when given, comes back with one position covariance
        // per point if the cloud carries them, and empty otherwise. The internal layout addresses its
        // fields by fixed offset and has no covariance, so only the named-field reader provides them.
        std::pair<std::vector<Pointd>, bool> getPcFromMsg(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg, std::vector<Mat3>* covariances = nullptr)
        {
            std::vector<Pointd> pts;
            bool is_2d = false;
            if(covariances != nullptr)
            {
                covariances->clear();
            }
            if(pc_type_internal_)
            {
                std::tie(pts, is_2d) = pointCloud2MsgToPtsVecInternal(msg);
            }
            else
            {
                bool rubish0, rubish1;
                std::tie(pts, rubish0, rubish1, is_2d) = pointCloud2MsgToPtsVec<double>(msg, 1e-9, false, std::set<int>(), false, covariances);
            }
            return {pts, is_2d};
        }

        // Density filter of the incoming scan, carrying the per-point covariances when they are in use
        void filterScan(std::vector<Pointd>& pts, std::vector<Mat3>& covs)
        {
            if(!covs.empty() && (covs.size() == pts.size()))
            {
                std::tie(pts, covs) = filterPointsDensity(pts, covs, voxel_size_);
            }
            else
            {
                covs.clear();
                pts = filterPointsDensity(pts, voxel_size_);
            }
        }

        // Registration voxel size and loss scale for one scan. With the adaptive scene scale off, the
        // configured values come back untouched.
        //
        // The ratio is the median range of the scan over the range the configuration was tuned for,
        // clamped to [adaptive_min_ratio, 1]: the mode only ever makes the settings finer, never
        // coarser than what the launch file asked for. The voxel follows the ratio directly, the loss
        // takes its square root: the loss parameter enters the Cauchy loss squared (the saturation of
        // `rho(d) = a^2*log(1 + d/a^2)` sits near `a^2` meters), so sqrt(ratio) on the parameter is
        // what scales the saturation distance by `ratio`.
        std::pair<double, double> getAdaptiveRegistrationSettings(const std::vector<Pointd>& pts)
        {
            if(!adaptive_scene_scale_)
            {
                return {downsample_size_, loss_scale_};
            }
            const double median_range = getMedianRange(pts);
            if(median_range <= 0.0)
            {
                RCLCPP_WARN(this->get_logger(), "Adaptive scene scale: the scan holds no usable range, the configured settings are kept for this one");
                return {downsample_size_, loss_scale_};
            }
            const double ratio = std::clamp(median_range/adaptive_reference_range_, adaptive_min_ratio_, 1.0);
            const double downsample_size = ratio*downsample_size_;
            const double loss_scale = std::sqrt(ratio)*loss_scale_;
            RCLCPP_INFO(this->get_logger(), "Adaptive scene scale: median range %.2f m, ratio %.3f, registration voxel %.3f m, loss scale %.3f", median_range, ratio, downsample_size, loss_scale);
            return {downsample_size, loss_scale};
        }

        // Downsample the scan for the registration, carrying the per-point covariances when they are
        // in use. `downsampled_covs` comes back empty otherwise, which is what registerPts expects to
        // register without them.
        // A non-positive `downsample_size` disables the downsampling altogether and the scan is
        // registered as it comes: useful for an already sparse input (the landmarks of a visual
        // front-end for instance), where merging points into voxel centroids only blurs them. It is
        // passed per scan rather than read from the member, as the adaptive scene scale shrinks it.
        std::vector<Pointd> downsampleScan(const std::vector<Pointd>& pts, const std::vector<Mat3>& covs, std::vector<Mat3>& downsampled_covs, const bool per_type, const bool quadrant_balanced, const double downsample_size)
        {
            downsampled_covs.clear();
            const bool with_covs = !covs.empty() && (covs.size() == pts.size());
            if(downsample_size <= 0.0)
            {
                // The invalid points are still dropped, as the downsampling does
                std::vector<Pointd> kept_pts;
                kept_pts.reserve(pts.size());
                for(size_t i = 0; i < pts.size(); ++i)
                {
                    if(pts[i].type == kInvalidPoint)
                    {
                        continue;
                    }
                    kept_pts.push_back(pts[i]);
                    if(with_covs)
                    {
                        downsampled_covs.push_back(covs[i]);
                    }
                }
                return kept_pts;
            }
            const std::vector<Mat3>* covs_in = with_covs ? &covs : nullptr;
            std::vector<Mat3>* covs_out = with_covs ? &downsampled_covs : nullptr;
            if(per_type)
            {
                return downsamplePointCloudPerType<double>(pts, downsample_size, max_nb_pts_, covs_in, covs_out);
            }
            return downsamplePointCloud<double>(pts, downsample_size, max_nb_pts_, quadrant_balanced, covs_in, covs_out);
        }

        void pcPriorCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr pc_msg, const geometry_msgs::msg::TransformStamped::ConstSharedPtr odom_msg)
        {
            updateMap(pc_msg, transformToMat4(odom_msg->transform));
        }


        void pcCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
        {
            updateMap(msg, current_pose_);
        }
        

        void mapPublishThread()
        {

            while(running_)
            {
                auto start = std::chrono::high_resolution_clock::now();

                int counter = counter_;
                if(counter != previous_counter_)
                {
                    previous_counter_ = counter;
                    if(map_pub_->get_subscription_count() > 0)
                    {
                        RCLCPP_INFO(this->get_logger(), "Publishing map points");
                        map_mutex_.lock();
                        std::vector<Pointd> pts = map_->getPts();
                        map_mutex_.unlock();
                        sensor_msgs::msg::PointCloud2 map_msg = ptsVecToPointCloud2MsgInternal(pts, "map", this->now());
                        map_pub_->publish(map_msg);
                    }
                }

                // Check if the last point cloud is too old
                if((last_write_counter_ != counter))
                {
                    std::chrono::time_point<std::chrono::high_resolution_clock> last_time_temp = last_pc_epoch_time_;
                    if(((start-last_time_temp) > std::chrono::duration<double>(5.0*key_framing_time_thr_)) && !localization_)
                    {
                        last_write_counter_ = counter;
                        map_mutex_.lock();
                        map_->writeMap();
                        map_mutex_.unlock();
                    }
                }


                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = end - start;
                std::this_thread::sleep_for(std::chrono::duration<double>(map_publish_period_) - elapsed);
            }
        }

        void createTrajectoryFile(const std::string& path)
        {
            // Create the trajectory file if it does not exist
            std::ofstream trajectory_file(path, std::ios::out | std::ios::trunc);
            if (trajectory_file.is_open())
            {
                trajectory_file << "timestamp, x, y, z, r0, r1, r2" 
                                << std::endl; // Header line
                trajectory_file.close();
                RCLCPP_INFO(this->get_logger(), "Created trajectory file: %s", path.c_str());
            }
            else
            {
                RCLCPP_ERROR(this->get_logger(), "Could not create trajectory file: %s", path.c_str());
                return;
            }
        }

        void logPoseToFile(const std::string& path, const Mat4 & pose, const rclcpp::Time & time)
        {
            // Log the trajectory estimate
            std::ofstream trajectory_file(path, std::ios::out | std::ios::app);
            if (trajectory_file.is_open())
            {
                Mat3 rot_mat = pose.block<3,3>(0,0);
                Vec3 rot_vec = logMap(rot_mat);
                trajectory_file << std::fixed << time.nanoseconds() << ", "
                                << pose(0,3) << ", "
                                << pose(1,3) << ", "
                                << pose(2,3) << ", "
                                << rot_vec(0) << ", "
                                << rot_vec(1) << ", "
                                << rot_vec(2)
                                << std::endl; // Write the current pose to the trajectory file
                trajectory_file.close();
                RCLCPP_INFO(this->get_logger(), "Updated traj file: %s", path.c_str());
            }
            else
            {
                RCLCPP_ERROR(this->get_logger(), "Could not open trajectory file: %s", path.c_str());
                return;
            }
        }

        void gyrCallback(const sensor_msgs::msg::Imu::ConstSharedPtr msg)
        {
            Vec3 gyr(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
            map_mutex_.lock();
            map_->addGyrMeasurement(gyr, getTimeNs(rclcpp::Time(msg->header.stamp)));
            map_mutex_.unlock();
        }
        void accCallback(const sensor_msgs::msg::Imu::ConstSharedPtr msg)
        {
            Vec3 acc(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
            map_mutex_.lock();
            map_->addAccMeasurement(acc, getTimeNs(rclcpp::Time(msg->header.stamp)));
            map_mutex_.unlock();
        }
        void twistCallback(const geometry_msgs::msg::TwistStamped::ConstSharedPtr msg)
        {
            // Only accept the twist expressed in the IMU/body frame
            if(msg->header.frame_id != "imu")
            {
                return;
            }
            Vec3 linear(msg->twist.linear.x, msg->twist.linear.y, msg->twist.linear.z);
            map_mutex_.lock();
            map_->addVelocity(linear, getTimeNs(rclcpp::Time(msg->header.stamp)));
            map_mutex_.unlock();
        }

};




int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<GpMapNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}



