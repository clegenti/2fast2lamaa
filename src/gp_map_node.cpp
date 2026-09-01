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

            // Read the parameters for options
            voxel_size_ = readRequiredFieldDouble(this, "voxel_size");
            MapDistFieldOptions options;
            options.cell_size = voxel_size_;
            downsample_size_ = readFieldDouble(this, "voxel_size_factor_for_registration", 2.0) * voxel_size_;
            options.neighborhood_size = readFieldInt(this, "neighbourhood_size",2.0);

            register_ = readFieldBool(this, "register", true);
            bool with_init_guess = readFieldBool(this, "with_init_guess", true);
            with_init_guess_ = with_init_guess;
            approximate_ = readFieldBool(this, "no_gp", false);
            use_edge_field_ = options.edge_field;

            map_publish_period_ = readFieldDouble(this, "map_publish_period", 0.2);

            options.free_space_carving_radius = readFieldDouble(this, "free_space_carving_radius", -1.0);

            localization_ = readFieldBool(this, "localization_only", false);

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
            options.odom_prior_weight_pos = readFieldDouble(this, "odom_prior_weight_pos", 1.0);
            options.odom_prior_weight_rot = readFieldDouble(this, "odom_prior_weight_rot", 1.0);
            if(options.use_odom_prior && !with_init_guess)
            {
                RCLCPP_WARN(this->get_logger(), "use_odom_prior is set but there is no odometry input (with_init_guess is false): the prior will anchor the registration to the previous pose instead");
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

            loss_scale_ = readFieldDouble(this, "loss_function_scale", 5.0*voxel_size_/3.0);
            
            // Write the first line of the trajectory file
            traj_path_ = map_path + "/trajectory.csv";
            createTrajectoryFile(traj_path_);



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

        bool localization_ = false;
        bool use_edge_field_ = true;

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

        std::atomic<bool> running_ = true;
        std::atomic<int> counter_ = 0;
        int previous_counter_ = 0;

        int last_write_counter_ = 0;

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
            const double threshold = scan_dt_mean_ + kDropoutSigmaFactor*scanIntervalStdev();
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
                                  const int64_t time_ns, const bool disable_odom_prior = false)
        {
            Mat4 pose = prior;
            for(const double coarse_loss_scale : kCoarseToFineLossScales)
            {
                pose = map_->registerPts(pts, pose, time_ns, true, coarse_loss_scale,
                                         kCoarseToFineIterations, disable_odom_prior);
            }
            return pose;
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
                // First convert the point cloud message to a vector of points
                auto [pts, is_2d] = getPcFromMsg(msg);
                int original_pts_size = pts.size();
                pts = filterPointsDensity(pts, voxel_size_);

                if(is_2d)
                {
                    map_mutex_.lock();
                    map_->set2D(true);
                    map_mutex_.unlock();
                }

                if(localization_ && first_)
                {
                    // Downsample the points
                    std::vector<Pointd> downsampled_pts = downsamplePointCloud<double>(pts, downsample_size_, max_nb_pts_, true);

                    map_mutex_.lock();
                    current_pose_ = registerCoarseToFine(downsampled_pts, init_guess_, getTimeNs(time));
                    current_pose_ = map_->registerPts(downsampled_pts, current_pose_, getTimeNs(time), approximate_, loss_scale_);
                    init_guess_ = current_pose_;
                    map_mutex_.unlock();
                }
                else if(register_ && !first_)
                {
                    sw2.start();

                    // Downsample the points
                    std::vector<Pointd> downsampled_pts;
                    if(use_edge_field_)
                    {
                        downsampled_pts = downsamplePointCloudPerType<double>(pts, downsample_size_, max_nb_pts_);
                    }
                    else
                    {
                        downsampled_pts = downsamplePointCloud<double>(pts, downsample_size_, max_nb_pts_, false);
                    }


                    map_mutex_.lock();
                    if(!with_init_guess_)
                    {
                        current_pose_ = map_->registerPts(downsampled_pts, current_pose_, getTimeNs(time), true, 10.0*loss_scale_);
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
                        init_guess_ = registerCoarseToFine(downsampled_pts, init_guess_, getTimeNs(time), true);
                        pending_frame_dropout_ = false;
                    }
                    //current_pose_ = map_->registerPts(downsampled_pts, init_guess_, getTimeNs(time), true, 2*loss_scale_, 7);
                    current_pose_ = map_->registerPts(downsampled_pts, init_guess_, getTimeNs(time), approximate_, loss_scale_, 25);
                    init_guess_ = current_pose_;
                    map_mutex_.unlock();

                    publishOdomMapCorrection(time, trans);


                    double temp_time = sw2.stop();
                    RCLCPP_INFO(this->get_logger(), "Registration time: %f ms", temp_time);

                }
                else
                {
                    current_pose_ = trans;
                }
                publishPose(time, current_pose_);



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
            RCLCPP_INFO(this->get_logger(), "Total time to process point cloud: %f ms", time_ms);



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

        std::pair<std::vector<Pointd>, bool> getPcFromMsg(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg)
        {
            std::vector<Pointd> pts;
            bool is_2d = false;
            if(pc_type_internal_)
            {
                std::tie(pts, is_2d) = pointCloud2MsgToPtsVecInternal(msg);
            }
            else
            {
                bool rubish0, rubish1;
                std::tie(pts, rubish0, rubish1, is_2d) = pointCloud2MsgToPtsVec<double>(msg, 1e-9, false);
            }
            return {pts, is_2d};
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



