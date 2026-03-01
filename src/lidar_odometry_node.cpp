#include "rclcpp/rclcpp.hpp"
#include "ros_utils.h"
#include "lice/utils.h"
#include "lice/types.h"
#include "lice/lidar_odometry.h"
#include <memory>

#include "sensor_msgs/msg/point_cloud2.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "sensor_msgs/msg/imu.hpp"
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>



class LidarOdometryNode : public rclcpp::Node, public LidarOdometryPublisher
{
    public:
        LidarOdometryNode()
            : rclcpp::Node("lidar_odometry")
        {
            RCLCPP_INFO(this->get_logger(), "Starting lidar_odometry node");

            // Read the parameters that are passed to the lidar odometry code
            LidarOdometryParams params;
            params.low_latency = readFieldBool(this, "low_latency", true);
            params.dense_pc_output = readFieldBool(this, "dense_pc_output", false);
            params.min_range = readFieldDouble(this, "min_range", 1.0);
            params.max_range = readFieldDouble(this, "max_range", 150.0);
            params.min_feature_dist = readFieldDouble(this, "min_feature_dist", 0.05);
            params.max_feature_dist = readFieldDouble(this, "max_feature_dist", 0.5);
            params.max_feature_range = readFieldDouble(this, "max_feature_range", 150.0);
            params.feature_voxel_size = readFieldDouble(this, "feature_voxel_size", 0.3);
            params.loss_function_scale = readFieldDouble(this, "loss_function_scale", 1.0);
            params.state_frequency = readFieldDouble(this, "state_freq", 200.0);
            params.gyr_std = readFieldDouble(this, "gyr_std", 0.005);
            params.acc_std = readFieldDouble(this, "acc_std", 0.02);
            params.lidar_std = readFieldDouble(this, "lidar_std", 0.02);
            params.g = readFieldDouble(this, "g", 9.80);
            params.intensity_threshold = readFieldDouble(this, "minimum_intensity", -1.0);
            std::string mode = readFieldString(this, "mode", "imu");
            if(kLidarOdometryModeMap.find(mode) != kLidarOdometryModeMap.end())
            {
                params.mode = kLidarOdometryModeMap.at(mode);
                mode_ = params.mode;
            }
            else
            {
                RCLCPP_ERROR(this->get_logger(), "Invalid mode parameter: %s. Options are 'imu', 'gyr', 'no_imu'. Using 'imu' mode.", mode.c_str());
                throw std::runtime_error("Invalid mode parameter for LidarOdometryNode");
            }

            {
                params.calib_px = readRequiredFieldDouble(this, "calib_px");
                params.calib_py = readRequiredFieldDouble(this, "calib_py");
                params.calib_pz = readRequiredFieldDouble(this, "calib_pz");
                params.calib_rx = readRequiredFieldDouble(this, "calib_rx");
                params.calib_ry = readRequiredFieldDouble(this, "calib_ry");
                params.calib_rz = readRequiredFieldDouble(this, "calib_rz");
            }

            params.max_associations_per_type = readFieldInt(this, "max_associations_per_type", 1000);

            params.unsorted_pc = readFieldBool(this, "unsorted_pc", false);

            params.planar_only = readFieldBool(this, "planar_only", false);

            pc_scale_ = readFieldDouble(this, "point_cloud_scale", 1.0);


            // Read the broken channels
            std::string broken_channels_str = readFieldString(this, "broken_channels", "");
            if(!broken_channels_str.empty())
            {
                std::stringstream ss(broken_channels_str);
                std::string token;
                while(std::getline(ss, token, ','))
                {
                    int channel = std::stoi(token);
                    broken_channels_.insert(channel);
                    RCLCPP_INFO(this->get_logger(), "Adding broken channel: %d (ignoring data from this lidar channel)", channel);
                }
            }

            // Read parameters for pre p
            acc_in_m_s2_ = readFieldBool(this, "acc_in_m_per_s2", true);
            invert_imu_ = readFieldBool(this, "invert_imu", true);

            time_field_multiplier_ = readFieldDouble(this, "point_time_multiplier", 1e-9);
            absolute_time_ = readFieldBool(this, "absolute_time", false);

            lidar_odometry_ = std::make_shared<LidarOdometry>(params, this);

            odom_pub_ = this->create_publisher<geometry_msgs::msg::TransformStamped>("/undistortion_delta_transform", 10);
            global_odom_pub_ = this->create_publisher<geometry_msgs::msg::TransformStamped>("/undistortion_pose", 10);
            odom_twist_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/end_of_scan_odom", 10);
            odom_twist_only_pub_ = this->create_publisher<geometry_msgs::msg::TwistStamped>("/end_of_scan_odom_twist", 10);
            odom_twist_start_pub_ = this->create_publisher<geometry_msgs::msg::TwistStamped>("/start_of_scan_twist", 10);
            pc_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/lidar_scan_undistorted", 10);
            if(params.dense_pc_output)
            {
                pc_dense_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/lidar_scan_undistorted_dense", 10);
            }

            acc_sub_ = this->create_subscription<sensor_msgs::msg::Imu>("/imu/acc", 100, std::bind(&LidarOdometryNode::accCallback, this, std::placeholders::_1));
            gyr_sub_ = this->create_subscription<sensor_msgs::msg::Imu>("/imu/gyr", 100, std::bind(&LidarOdometryNode::gyrCallback, this, std::placeholders::_1));

            lidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("/lidar_raw_points", 100, std::bind(&LidarOdometryNode::pcCallback, this, std::placeholders::_1));

            odom_map_correction_sub_ = this->create_subscription<geometry_msgs::msg::TransformStamped>("/odom_map_correction", 10, std::bind(&LidarOdometryNode::odomMapCorrectionCallback, this, std::placeholders::_1));

            rclcpp::node_interfaces::NodeTopicsInterface::SharedPtr node_topics_handle = this->get_node_topics_interface();
            br_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this, tf2_ros::DynamicBroadcasterQoS(), rclcpp::PublisherOptions());

            thread_ = lidar_odometry_->runThread();
        }

        ~LidarOdometryNode()
        {
            lidar_odometry_->stop();
            thread_->join();
        }


        void publishTransform(const int64_t t, const Vec3& pos, const Vec3& rot)
        {
            rclcpp::Time new_time(t);

            // Send a TF transform
            geometry_msgs::msg::TransformStamped transformStamped;
            transformStamped.header.stamp = new_time;
            transformStamped.header.frame_id = "odom";
            transformStamped.child_frame_id = "lidar";
            transformStamped.transform.translation.x = pos[0];
            transformStamped.transform.translation.y = pos[1];
            transformStamped.transform.translation.z = pos[2];

            Eigen::AngleAxisd aa = Eigen::AngleAxisd(rot.norm(), rot.normalized());
            Eigen::Quaterniond q(aa);
            transformStamped.transform.rotation.x = q.x();
            transformStamped.transform.rotation.y = q.y();
            transformStamped.transform.rotation.z = q.z();
            transformStamped.transform.rotation.w = q.w();

            //static tf2_ros::TransformBroadcaster br;
            mutex_br_.lock();
            geometry_msgs::msg::TransformStamped temp_msg;
            if(odom_map_correction_msg_)
            {
                temp_msg = *odom_map_correction_msg_;
            }
            else
            {
                temp_msg.header.frame_id = "map";
                temp_msg.child_frame_id = "odom";
                temp_msg.transform = mat4ToTransform(Mat4::Identity());
            }
            temp_msg.header.stamp = new_time;
            br_->sendTransform(temp_msg);
            br_->sendTransform(transformStamped);
            global_odom_pub_->publish(transformStamped);
            mutex_br_.unlock();
        }
        void publishGlobalOdom(const int64_t t, const Vec3& pos, const Vec3& rot, const Vec3& vel, const Vec3& ang_vel)
        {
            rclcpp::Time new_time(t);

            nav_msgs::msg::Odometry odom_msg;
            odom_msg.header.stamp = new_time;
            odom_msg.header.frame_id = "odom";
            odom_msg.child_frame_id = "lidar_head";
            odom_msg.pose.pose.position.x = pos[0];
            odom_msg.pose.pose.position.y = pos[1];
            odom_msg.pose.pose.position.z = pos[2];

            Eigen::AngleAxisd aa = Eigen::AngleAxisd(rot.norm(), rot.normalized());
            Eigen::Quaterniond q(aa);
            odom_msg.pose.pose.orientation.x = q.x();
            odom_msg.pose.pose.orientation.y = q.y();
            odom_msg.pose.pose.orientation.z = q.z();
            odom_msg.pose.pose.orientation.w = q.w();

            odom_msg.twist.twist.linear.x = vel[0];
            odom_msg.twist.twist.linear.y = vel[1];
            odom_msg.twist.twist.linear.z = vel[2];

            odom_msg.twist.twist.angular.x = ang_vel[0];
            odom_msg.twist.twist.angular.y = ang_vel[1];
            odom_msg.twist.twist.angular.z = ang_vel[2];

            for (size_t i = 0; i < 36; ++i)
            {
                if (i % 6 == 0)
                {
                    odom_msg.pose.covariance[i] = 1;
                    odom_msg.twist.covariance[i] = 1;
                }
                else
                {
                    odom_msg.pose.covariance[i] = 0.0;
                    odom_msg.twist.covariance[i] = 0.0;
                }
            }

            geometry_msgs::msg::TwistStamped twist_msg;
            twist_msg.twist.linear.x = vel[0];
            twist_msg.twist.linear.y = vel[1];
            twist_msg.twist.linear.z = vel[2];
            twist_msg.twist.angular.x = ang_vel[0];
            twist_msg.twist.angular.y = ang_vel[1];
            twist_msg.twist.angular.z = ang_vel[2];

            twist_msg.header.stamp = new_time;
            twist_msg.header.frame_id = "lidar_head";

            odom_twist_pub_->publish(odom_msg);
            odom_twist_only_pub_->publish(twist_msg);


            geometry_msgs::msg::TransformStamped global_transform;
            global_transform.header.stamp = new_time;
            global_transform.header.frame_id = "odom";
            global_transform.child_frame_id = "lidar_head";
            global_transform.transform.translation.x = pos[0];
            global_transform.transform.translation.y = pos[1];
            global_transform.transform.translation.z = pos[2];
            global_transform.transform.rotation.x = q.x();
            global_transform.transform.rotation.y = q.y();
            global_transform.transform.rotation.z = q.z();
            global_transform.transform.rotation.w = q.w();

            // Publish the global odometry transform

            mutex_br_.lock();
            br_->sendTransform(global_transform);
            mutex_br_.unlock();
        }

        void publishTwist(const int64_t t, const Vec3& linear, const Vec3& angular)
        {
            rclcpp::Time new_time(t);
            geometry_msgs::msg::TwistStamped twist_msg;
            twist_msg.header.stamp = new_time;
            twist_msg.header.frame_id = "lidar";
            twist_msg.twist.linear.x = linear[0];
            twist_msg.twist.linear.y = linear[1];
            twist_msg.twist.linear.z = linear[2];
            twist_msg.twist.angular.x = angular[0];
            twist_msg.twist.angular.y = angular[1];
            twist_msg.twist.angular.z = angular[2];

            odom_twist_start_pub_->publish(twist_msg);
        }

        void publishPc(const int64_t t, const std::vector<Pointd>& pc)
        {
            rclcpp::Time new_time(t);
            RCLCPP_INFO(this->get_logger(), "Publishing point cloud with %zu points at time %f", pc.size(), new_time.seconds());
            sensor_msgs::msg::PointCloud2 pc_msg = ptsVecToPointCloud2MsgInternal(pc, "lidar", new_time);
            mutex_pc_.lock();
            pc_pub_->publish(pc_msg);
            mutex_pc_.unlock();
        }

        void publishPcDense(const int64_t t, const std::vector<Pointd>& pc)
        {
            if(!pc_dense_pub_)
            {
                RCLCPP_WARN(this->get_logger(), "Dense point cloud publisher is not initialized, skipping dense point cloud publishing");
                return;
            }
            rclcpp::Time new_time(t);
            sensor_msgs::msg::PointCloud2 pc_msg = ptsVecToPointCloud2MsgInternal(pc, "lidar", new_time);
            mutex_pc_.lock();
            pc_dense_pub_->publish(pc_msg);
            mutex_pc_.unlock();
        }


    private:
        std::shared_ptr<LidarOdometry> lidar_odometry_;

        rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr acc_sub_;
        rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr gyr_sub_;
        rclcpp::Subscription<geometry_msgs::msg::TransformStamped>::SharedPtr odom_map_correction_sub_;

        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;

        rclcpp::Publisher<geometry_msgs::msg::TransformStamped>::SharedPtr odom_pub_;
        rclcpp::Publisher<geometry_msgs::msg::TransformStamped>::SharedPtr global_odom_pub_;
        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_twist_pub_;
        rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr odom_twist_only_pub_;
        rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr odom_twist_start_pub_;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pc_pub_;

        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pc_dense_pub_;

        LidarOdometryMode mode_ = LidarOdometryMode::IMU;

        std::unique_ptr<tf2_ros::TransformBroadcaster> br_;

        std::mutex mutex_br_;
        std::mutex mutex_pc_;

        int scan_count_ = 0;
        bool invert_imu_ = false;
        double pc_scale_ = 1.0;

        bool first_gyr_ = true;
        bool first_acc_ = true;
        rclcpp::Time last_gyr_time_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
        rclcpp::Time last_acc_time_ = rclcpp::Time(0, 0, RCL_ROS_TIME);

        bool acc_in_m_s2_ = true;
        double time_field_multiplier_ = 1e-9; // Default to seconds
        bool absolute_time_ = false;

        geometry_msgs::msg::TransformStamped::SharedPtr odom_map_correction_msg_;

        std::set<int> broken_channels_;
        std::shared_ptr<std::thread> thread_;

        void odomMapCorrectionCallback(const geometry_msgs::msg::TransformStamped::SharedPtr msg)
        {
            mutex_br_.lock();
            // Save the correction
            odom_map_correction_msg_ = msg;
            mutex_br_.unlock();
        }


        void pcCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr pc_msg)
        {
            StopWatch sw;
            sw.start();

            if( ((first_acc_ || first_gyr_) && (mode_ == LidarOdometryMode::IMU)) || (first_gyr_ && (mode_ == LidarOdometryMode::GYR)) )
            {
                RCLCPP_WARN(this->get_logger(), "Received point cloud before IMU messages, ignoring the point cloud");
                return;
            }
            auto [incoming_pts, temp_has_intensity, temp_has_channel, is_2d] = pointCloud2MsgToPtsVec<double>(pc_msg, time_field_multiplier_, true, broken_channels_, absolute_time_);
            std::shared_ptr<std::vector<Pointd>> incoming_pts_ptr = std::make_shared<std::vector<Pointd>>(std::move(incoming_pts));
            std::cout << "Point cloud with " << incoming_pts_ptr->size() << " points received." << std::endl;
            rclcpp::Time header_time(pc_msg->header.stamp);
            std::cout << "At " << std::fixed << header_time.nanoseconds() << std::endl;
            std::cout << std::fixed << "First point time: " << incoming_pts_ptr->at(0).t << ", last point time: " << incoming_pts_ptr->at(incoming_pts_ptr->size()-1).t << std::endl;

            
            // Scale the point cloud if needed
            if(pc_scale_ != 1.0)
            {
                for(auto& pt : *incoming_pts_ptr)
                {
                    pt.x *= pc_scale_;
                    pt.y *= pc_scale_;
                    pt.z *= pc_scale_;
                }
            }
            lidar_odometry_->setIs2D(is_2d);
            lidar_odometry_->addPc(incoming_pts_ptr, header_time.nanoseconds());

            sw.stop();
            sw.print("++++++++++++   Lidar callback took: ");
        }

        void accCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
        {
            rclcpp::Time header_time = msg->header.stamp;
            // For sanity check on the assumption that the message timestamp is always newer than the last one
            if(first_acc_)
            {
                first_acc_ = false;
                last_acc_time_ = header_time - rclcpp::Duration::from_seconds(0.1);
            }
            if(header_time <= last_acc_time_)
            {
                RCLCPP_WARN(this->get_logger(), "Received IMU acceleration message with timestamp %f that is not newer than the last one %f", header_time.seconds(), last_acc_time_.seconds());
                return;
            }
            last_acc_time_ = header_time;

            Vec3 acc;
            acc << msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z;
            if(!acc_in_m_s2_)
            {
                acc *= 9.81;
            }
            if (invert_imu_)
            {
                acc *= -1;
            }
            lidar_odometry_->addAccSample(acc, header_time.nanoseconds());
        }

        void gyrCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
        {
            rclcpp::Time header_time = msg->header.stamp;
            // For sanity check on the assumption that the message timestamp is always newer than the last one
            if(first_gyr_)
            {
                first_gyr_ = false;
                last_gyr_time_ = header_time - rclcpp::Duration::from_seconds(0.1);
            }
            if(header_time <= last_gyr_time_)
            {
                RCLCPP_WARN(this->get_logger(), "Received IMU gyro message with timestamp %f that is not newer than the last one %f", header_time.seconds(), last_gyr_time_.seconds());
                return;
            }
            last_gyr_time_ = header_time;

            Vec3 gyr;
            gyr << msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z;
            if (invert_imu_)
            {
                gyr *= -1;
            }
            lidar_odometry_->addGyroSample(gyr, header_time.nanoseconds());
        }
};


int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<LidarOdometryNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

