#pragma once

#include "rclcpp/rclcpp.hpp"
#include "lice/ros_utils.h"
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


class LidarOdometry;


class LidarOdometryNode : public rclcpp::Node
{
    public:
        LidarOdometryNode();

        void publishTransform(const int64_t t, const Vec3& pos, const Vec3& rot);
        void publishPc(const int64_t t, const std::vector<Pointd>& pc);
        void publishGlobalOdom(const int64_t t, const Vec3& pos, const Vec3& rot, const Vec3& vel, const Vec3& ang_vel);

        void publishPcDense(const int64_t t, const std::vector<Pointd>& pc);
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


        void pcCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr pc_msg);

        void accCallback(const sensor_msgs::msg::Imu::SharedPtr msg);

        void gyrCallback(const sensor_msgs::msg::Imu::SharedPtr msg);

        void odomMapCorrectionCallback(const geometry_msgs::msg::TransformStamped::SharedPtr msg);
};