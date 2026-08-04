#include "rclcpp/rclcpp.hpp"
#include "ros_utils.h"
#include "lice/types.h"
#include "lice/math_utils.h"

#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <deque>
#include <algorithm>
#include <limits>
#include <cmath>

#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2_ros/static_transform_broadcaster.h"


const int64_t kNanoToSec = 1000000000;


// One wheel encoder reading (the count is unwrapped, thus it is not an integer anymore)
struct EncoderSample
{
    int64_t t = 0;
    double count = 0.0;
};

// One gyroscope reading (angular velocity in the IMU frame, without any bias correction)
struct GyrSample
{
    int64_t t = 0;
    Vec3 ang_vel = Vec3::Zero();
};

// One odometry output (pose of the IMU in the odom frame), used to interpolate the
// sensor pose at the timestamp of each individual lidar point
struct OdomSample
{
    int64_t t = 0;
    Vec3 pos = Vec3::Zero();
    Mat3 rot = Mat3::Identity();
};


// Get the first timestamp of the buffer that is strictly after the given one
// (returns the max int64_t if there is none)
template <typename T>
inline int64_t nextTimeAfter(const std::deque<T>& data, const int64_t t)
{
    auto it = std::upper_bound(data.begin(), data.end(), t,
            [](const int64_t& a, const T& b){ return a < b.t; });
    if(it == data.end())
    {
        return std::numeric_limits<int64_t>::max();
    }
    return it->t;
}

// Linearly interpolate the value of a time-sorted buffer at the given timestamp.
// Returns false if the timestamp is not covered by the buffer.
template <typename T, typename V, typename ValueFn>
inline bool interpolateSamples(const std::deque<T>& data, const int64_t t, ValueFn value_fn, V& output)
{
    if(data.size() < 2)
    {
        return false;
    }
    if((t < data.front().t) || (t > data.back().t))
    {
        return false;
    }
    auto it = std::lower_bound(data.begin(), data.end(), t,
            [](const T& a, const int64_t& b){ return a.t < b; });
    if(it->t == t)
    {
        output = value_fn(*it);
        return true;
    }
    // The range check above guarantees that `it` is neither the first nor the past-the-end iterator
    const T& s1 = *it;
    const T& s0 = *(it - 1);
    const double dt = (double)(s1.t - s0.t);
    if(dt <= 0.0)
    {
        return false;
    }
    const double alpha = (double)(t - s0.t) / dt;
    output = value_fn(s0) + alpha * (value_fn(s1) - value_fn(s0));
    return true;
}

// Linearly interpolate a pose in a time-sorted vector of odometry outputs (slerp for the rotation)
inline bool interpolateOdom(const std::vector<OdomSample>& data, const int64_t t, Vec3& pos, Mat3& rot)
{
    if(data.size() < 2)
    {
        return false;
    }
    if((t < data.front().t) || (t > data.back().t))
    {
        return false;
    }
    auto it = std::lower_bound(data.begin(), data.end(), t,
            [](const OdomSample& a, const int64_t& b){ return a.t < b; });
    if(it->t == t)
    {
        pos = it->pos;
        rot = it->rot;
        return true;
    }
    const OdomSample& s1 = *it;
    const OdomSample& s0 = *(it - 1);
    const double dt = (double)(s1.t - s0.t);
    if(dt <= 0.0)
    {
        return false;
    }
    const double alpha = (double)(t - s0.t) / dt;
    pos = s0.pos + alpha * (s1.pos - s0.pos);
    rot = Eigen::Quaterniond(s0.rot).slerp(alpha, Eigen::Quaterniond(s1.rot)).normalized().toRotationMatrix();
    return true;
}


class WheelGyroOdometryNode : public rclcpp::Node
{
    public:
        WheelGyroOdometryNode()
            : rclcpp::Node("wheel_gyro_odometry")
        {
            RCLCPP_INFO(this->get_logger(), "Starting wheel_gyro_odometry node");

            // Wheel encoder to travelled distance (in meters per encoder count) and encoder wrap-around
            wheel_scale_ = readRequiredFieldDouble(this, "wheel_scale");
            encoder_wrap_ = (double)readFieldInt(this, "encoder_wrap_max", 16777216);

            // Calibration between the IMU and the lidar (T_imu_lidar, same convention as the
            // calib_* parameters of the lidar_scan_odometry node)
            {
                Vec3 calib_pos, calib_rot;
                calib_pos << readRequiredFieldDouble(this, "calib_px"),
                             readRequiredFieldDouble(this, "calib_py"),
                             readRequiredFieldDouble(this, "calib_pz");
                calib_rot << readRequiredFieldDouble(this, "calib_rx"),
                             readRequiredFieldDouble(this, "calib_ry"),
                             readRequiredFieldDouble(this, "calib_rz");
                T_imu_lidar_ = posRotToTransform(calib_pos, calib_rot);
            }

            // Calibration between the IMU and the wheel encoder (T_wheel_imu, same convention as
            // the calib_* parameters: pose of the wheel frame in the IMU frame). The wheel frame
            // is the one the encoder measures the travelled distance along its y axis.
            {
                Vec3 calib_pos, calib_rot;
                calib_pos << readRequiredFieldDouble(this, "wheel_calib_px"),
                             readRequiredFieldDouble(this, "wheel_calib_py"),
                             readRequiredFieldDouble(this, "wheel_calib_pz");
                calib_rot << readRequiredFieldDouble(this, "wheel_calib_rx"),
                             readRequiredFieldDouble(this, "wheel_calib_ry"),
                             readRequiredFieldDouble(this, "wheel_calib_rz");
                T_wheel_imu_ = posRotToTransform(calib_pos, calib_rot);
            }

            // Initial gyroscope bias in the IMU frame (rad/s)
            gyr_bias_ << readFieldDouble(this, "gyr_bias_init_x", 0.0),
                         readFieldDouble(this, "gyr_bias_init_y", 0.0),
                         readFieldDouble(this, "gyr_bias_init_z", 0.0);

            // Online gyroscope bias estimation over the windows where the encoder does not tick.
            // The bias is averaged over `bias_use_time` seconds taken in the middle of a
            // `zero_vel_required_time` seconds long stationary window (the edges of the window are
            // discarded as they are polluted by the start/stop motion). The accumulation window is
            // capped to `bias_max_use_time` seconds.
            estimate_gyr_bias_ = readFieldBool(this, "estimate_gyr_bias", true);
            zero_vel_time_ = readFieldDouble(this, "zero_vel_required_time", 3.0);
            bias_use_time_ = readFieldDouble(this, "bias_use_time", 2.0);
            bias_max_time_ = readFieldDouble(this, "bias_max_use_time", 10.0);
            if(bias_use_time_ > zero_vel_time_)
            {
                RCLCPP_ERROR(this->get_logger(), "bias_use_time (%f) must not be greater than zero_vel_required_time (%f)", bias_use_time_, zero_vel_time_);
                throw std::invalid_argument("Invalid parameter");
            }

            // Rotation increments smaller than that are discarded while the encoder does not tick,
            // this prevents the gyroscope noise from being integrated while standing still
            zero_vel_rot_thr_ = readFieldDouble(this, "zero_vel_rot_threshold", 1e-3);

            // The reference python implementation starts with the wheel frame's y axis (the
            // direction the encoder measures) pointing towards the odom frame's x axis
            init_yaw_ = readFieldDouble(this, "init_yaw", M_PI/2.0);

            undistort_pc_ = readFieldBool(this, "undistort_pc", true);
            time_field_multiplier_ = readFieldDouble(this, "point_time_multiplier", 1e-9);
            absolute_time_ = readFieldBool(this, "absolute_time", false);
            publish_tf_ = readFieldBool(this, "publish_tf", true);

            // How long the odometry output is kept to undistort the point clouds, and how long the
            // raw encoder/gyroscope readings are kept to integrate the odometry
            odom_buffer_time_ = readFieldDouble(this, "odom_buffer_time", 5.0);
            sensor_buffer_time_ = readFieldDouble(this, "sensor_buffer_time", 2.0);
            max_pc_wait_time_ = readFieldDouble(this, "max_pc_wait_time", 2.0);

            odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/wheel_gyro_odom", 100);
            global_odom_pub_ = this->create_publisher<geometry_msgs::msg::TransformStamped>("/undistortion_pose", 10);
            odom_twist_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/end_of_scan_odom", 10);
            pc_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/lidar_scan_undistorted", 10);

            encoder_sub_ = this->create_subscription<sensor_msgs::msg::JointState>("/wheel_encoder", 500, std::bind(&WheelGyroOdometryNode::encoderCallback, this, std::placeholders::_1));
            gyr_sub_ = this->create_subscription<sensor_msgs::msg::Imu>("/imu/gyr", 500, std::bind(&WheelGyroOdometryNode::gyrCallback, this, std::placeholders::_1));
            lidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("/lidar_raw_points", 100, std::bind(&WheelGyroOdometryNode::pcCallback, this, std::placeholders::_1));

            br_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this, tf2_ros::DynamicBroadcasterQoS(), rclcpp::PublisherOptions());

            // The odometry publishes the pose of the IMU/body frame, broadcast the lidar extrinsic
            // so that the raw point clouds (which are in the physical lidar frame) can be located
            static_br_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(*this);
            {
                geometry_msgs::msg::TransformStamped imu_to_lidar;
                imu_to_lidar.header.stamp = this->now();
                imu_to_lidar.header.frame_id = "imu";
                imu_to_lidar.child_frame_id = "lidar";
                imu_to_lidar.transform = mat4ToTransform(T_imu_lidar_);
                static_br_->sendTransform(imu_to_lidar);
            }

            thread_ = std::thread(&WheelGyroOdometryNode::undistortionThread, this);
        }

        ~WheelGyroOdometryNode()
        {
            running_.store(false);
            pc_cv_.notify_all();
            {
                std::lock_guard<std::mutex> lock(data_mutex_);
                odom_cv_.notify_all();
            }
            if(thread_.joinable())
            {
                thread_.join();
            }
        }

    private:
        // Status of a point cloud waiting to be undistorted
        enum class PcStatus { WAITING, READY, TOO_OLD };

        void encoderCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
        {
            if(msg->position.empty())
            {
                RCLCPP_WARN(this->get_logger(), "Encoder message without any position, ignoring it");
                return;
            }

            const int64_t t = rclcpp::Time(msg->header.stamp).nanoseconds();
            const double raw_count = msg->position[0];

            std::lock_guard<std::mutex> lock(data_mutex_);

            // The unwrapping state is only updated for the readings that are actually used
            if(!encoder_samples_.empty() && (t <= encoder_samples_.back().t))
            {
                RCLCPP_WARN(this->get_logger(), "Encoder reading with a non-increasing timestamp, ignoring it");
                return;
            }

            if(encoder_samples_.empty() && !has_encoder_)
            {
                has_encoder_ = true;
                last_raw_count_ = raw_count;
                unwrapped_count_ = raw_count;
            }
            else
            {
                // Unwrap the encoder count (the encoder counter rolls over at `encoder_wrap_max`)
                double diff = raw_count - last_raw_count_;
                if(std::abs(diff) > (0.5*encoder_wrap_))
                {
                    diff -= std::copysign(encoder_wrap_, diff);
                }
                unwrapped_count_ += diff;
                last_raw_count_ = raw_count;
            }

            encoder_samples_.push_back({t, unwrapped_count_});
            this->integrateOdometry();
        }

        void gyrCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
        {
            const int64_t t = rclcpp::Time(msg->header.stamp).nanoseconds();

            std::lock_guard<std::mutex> lock(data_mutex_);
            if(!gyr_samples_.empty() && (t <= gyr_samples_.back().t))
            {
                RCLCPP_WARN(this->get_logger(), "Gyroscope reading with a non-increasing timestamp, ignoring it");
                return;
            }

            GyrSample sample;
            sample.t = t;
            sample.ang_vel << msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z;
            gyr_samples_.push_back(sample);
            this->integrateOdometry();
        }

        void pcCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg)
        {
            if(!undistort_pc_)
            {
                pc_pub_->publish(*msg);
                return;
            }

            {
                std::lock_guard<std::mutex> lock(pc_mutex_);
                pending_pc_.push_back(msg);
            }
            pc_cv_.notify_one();
        }

        // Integrate the wheel encoder and the gyroscope up to the last timestamp both sensors cover.
        // The two sensors are not assumed to be synchronised: the integration steps through the
        // union of the two sets of timestamps and interpolates both signals at each step.
        // The `data_mutex_` must be held by the caller.
        void integrateOdometry()
        {
            if((encoder_samples_.size() < 2) || (gyr_samples_.size() < 2))
            {
                return;
            }

            const int64_t t_limit = std::min(encoder_samples_.back().t, gyr_samples_.back().t);

            if(!odom_initialised_)
            {
                const int64_t t_start = std::max(encoder_samples_.front().t, gyr_samples_.front().t);
                if(t_start >= t_limit)
                {
                    return;
                }
                odom_initialised_ = true;
                current_time_ = t_start;
                pos_ = Vec3::Zero();
                rot_ = expMap(Vec3(0.0, 0.0, init_yaw_));
                this->pushOdom(current_time_, Vec3::Zero(), Vec3::Zero());
            }

            while(current_time_ < t_limit)
            {
                const int64_t next_time = std::min({nextTimeAfter(encoder_samples_, current_time_),
                                                    nextTimeAfter(gyr_samples_, current_time_),
                                                    t_limit});
                if(next_time <= current_time_)
                {
                    break;
                }
                const double dt = (double)(next_time - current_time_) / (double)kNanoToSec;

                double count_0 = 0.0;
                double count_1 = 0.0;
                Vec3 gyr_0 = Vec3::Zero();
                Vec3 gyr_1 = Vec3::Zero();
                auto get_count = [](const EncoderSample& s){ return s.count; };
                auto get_ang_vel = [](const GyrSample& s){ return s.ang_vel; };
                if(!interpolateSamples(encoder_samples_, current_time_, get_count, count_0)
                    || !interpolateSamples(encoder_samples_, next_time, get_count, count_1)
                    || !interpolateSamples(gyr_samples_, current_time_, get_ang_vel, gyr_0)
                    || !interpolateSamples(gyr_samples_, next_time, get_ang_vel, gyr_1))
                {
                    break;
                }

                // Mean angular velocity over the step, in the IMU frame then in the wheel frame
                const Vec3 gyr_raw = 0.5*(gyr_0 + gyr_1);
                const Vec3 ang_vel_imu = gyr_raw - gyr_bias_;
                const Vec3 ang_vel_wheel = T_wheel_imu_.block<3, 3>(0, 0) * ang_vel_imu;

                // The encoder measures the travelled distance along the y axis of the wheel frame
                const double dist = (count_1 - count_0) * wheel_scale_;
                total_travelled_distance_ += std::abs(dist);
                Vec3 ang = ang_vel_wheel * dt;

                if(dist == 0.0)
                {
                    // The encoder did not tick: the platform is considered stationary
                    this->updateGyrBias(next_time, gyr_raw);
                    if(ang.norm() < zero_vel_rot_thr_)
                    {
                        ang = Vec3::Zero();
                    }
                }
                else
                {
                    bias_window_.clear();
                }

                // The position is integrated with the orientation at the beginning of the step
                pos_ += rot_ * Vec3(0.0, dist, 0.0);
                rot_ = rot_ * expMap(ang);

                this->pushOdom(next_time, Vec3(0.0, dist/dt, 0.0), ang_vel_wheel);
                if(dist != 0.0)
                {
                    RCLCPP_INFO(this->get_logger(), "Travelled distance from start: %.3f m", total_travelled_distance_);
                }
                current_time_ = next_time;
            }

            this->trimBuffers();
        }

        // Accumulate the raw gyroscope readings of a stationary window and average its middle part.
        // The `data_mutex_` must be held by the caller.
        void updateGyrBias(const int64_t t, const Vec3& gyr_raw)
        {
            if(!estimate_gyr_bias_)
            {
                return;
            }

            bias_window_.push_back({t, gyr_raw});
            const int64_t window_time = bias_window_.back().t - bias_window_.front().t;
            if(window_time <= (int64_t)(zero_vel_time_*kNanoToSec))
            {
                return;
            }

            // Discard the edges of the window as they are polluted by the start/stop motion
            const int64_t offset = (int64_t)(0.5*(zero_vel_time_ - bias_use_time_)*kNanoToSec);
            const int64_t t_min = bias_window_.front().t + offset;
            const int64_t t_max = bias_window_.back().t - offset;
            Vec3 sum = Vec3::Zero();
            int num = 0;
            for(const auto& sample : bias_window_)
            {
                if((sample.t >= t_min) && (sample.t <= t_max))
                {
                    sum += sample.ang_vel;
                    ++num;
                }
            }
            if(num > 0)
            {
                gyr_bias_ = sum / (double)num;
            }

            // Limit the length of the accumulation window
            const int64_t t_drop = t - (int64_t)(bias_max_time_*kNanoToSec) - 2*offset;
            while(!bias_window_.empty() && (bias_window_.front().t < t_drop))
            {
                bias_window_.pop_front();
            }
        }

        // Store and publish the odometry at the current state. The velocities are given in the
        // wheel frame. The `data_mutex_` must be held by the caller.
        void pushOdom(const int64_t t, const Vec3& vel_wheel, const Vec3& ang_vel_wheel)
        {
            const Mat4 T_odom_wheel = posRotToTransform(pos_, logMap(rot_));
            const Mat4 T_odom_imu = T_odom_wheel * T_wheel_imu_;

            OdomSample sample;
            sample.t = t;
            sample.pos = T_odom_imu.block<3, 1>(0, 3);
            sample.rot = T_odom_imu.block<3, 3>(0, 0);
            odom_history_.push_back(sample);

            // Velocity of the IMU origin, expressed in the IMU frame (nav_msgs/Odometry expresses
            // the twist in the child frame)
            const Mat3 R_imu_wheel = T_imu_wheel_.block<3, 3>(0, 0);
            const Vec3 p_wheel_imu = T_wheel_imu_.block<3, 1>(0, 3);
            const Vec3 vel_imu = R_imu_wheel * (vel_wheel + ang_vel_wheel.cross(p_wheel_imu));
            const Vec3 ang_vel_imu = R_imu_wheel * ang_vel_wheel;

            nav_msgs::msg::Odometry msg;
            msg.header.stamp = rclcpp::Time(t);
            msg.header.frame_id = "odom";
            msg.child_frame_id = "imu";
            msg.pose.pose.position.x = sample.pos[0];
            msg.pose.pose.position.y = sample.pos[1];
            msg.pose.pose.position.z = sample.pos[2];
            const Eigen::Quaterniond q(sample.rot);
            msg.pose.pose.orientation.x = q.x();
            msg.pose.pose.orientation.y = q.y();
            msg.pose.pose.orientation.z = q.z();
            msg.pose.pose.orientation.w = q.w();
            msg.twist.twist.linear.x = vel_imu[0];
            msg.twist.twist.linear.y = vel_imu[1];
            msg.twist.twist.linear.z = vel_imu[2];
            msg.twist.twist.angular.x = ang_vel_imu[0];
            msg.twist.twist.angular.y = ang_vel_imu[1];
            msg.twist.twist.angular.z = ang_vel_imu[2];
            odom_pub_->publish(msg);

            // Drop the odometry that is too old to be of any use for the undistortion
            const int64_t t_min = odom_history_.back().t - (int64_t)(odom_buffer_time_*kNanoToSec);
            while((odom_history_.size() > 2) && (odom_history_[1].t < t_min))
            {
                odom_history_.pop_front();
            }

            // Only wake the undistortion thread up if it is actually waiting for the odometry
            if(num_pc_waiting_ > 0)
            {
                odom_cv_.notify_all();
            }
        }

        // Drop the raw readings that have already been integrated.
        // The `data_mutex_` must be held by the caller.
        void trimBuffers()
        {
            const int64_t t_min = current_time_ - (int64_t)(sensor_buffer_time_*kNanoToSec);
            while((encoder_samples_.size() > 2) && (encoder_samples_[1].t < t_min))
            {
                encoder_samples_.pop_front();
            }
            while((gyr_samples_.size() > 2) && (gyr_samples_[1].t < t_min))
            {
                gyr_samples_.pop_front();
            }
        }

        // The `data_mutex_` must be held by the caller.
        PcStatus pcStatus(const int64_t t_min, const int64_t t_max) const
        {
            if(odom_history_.size() < 2)
            {
                return PcStatus::WAITING;
            }
            if(odom_history_.front().t > t_min)
            {
                // The odometry will never cover the beginning of that scan
                return PcStatus::TOO_OLD;
            }
            if(odom_history_.back().t < t_max)
            {
                return PcStatus::WAITING;
            }
            return PcStatus::READY;
        }

        // Copy the odometry samples bracketing [t_min, t_max] so that the undistortion does not
        // have to hold the `data_mutex_` while interpolating every single point.
        // The `data_mutex_` must be held by the caller.
        void getOdomWindow(const int64_t t_min, const int64_t t_max, std::vector<OdomSample>& output) const
        {
            auto it_first = std::upper_bound(odom_history_.begin(), odom_history_.end(), t_min,
                    [](const int64_t& a, const OdomSample& b){ return a < b.t; });
            if(it_first != odom_history_.begin())
            {
                --it_first;
            }
            auto it_last = std::lower_bound(odom_history_.begin(), odom_history_.end(), t_max,
                    [](const OdomSample& a, const int64_t& b){ return a.t < b; });
            if(it_last != odom_history_.end())
            {
                ++it_last;
            }
            output.assign(it_first, it_last);
        }

        void undistortionThread()
        {
            while(running_.load())
            {
                sensor_msgs::msg::PointCloud2::ConstSharedPtr pc_msg;
                {
                    std::unique_lock<std::mutex> lock(pc_mutex_);
                    pc_cv_.wait(lock, [this](){ return !running_.load() || !pending_pc_.empty(); });
                    if(!running_.load())
                    {
                        return;
                    }
                    pc_msg = pending_pc_.front();
                    pending_pc_.pop_front();
                }

                std::vector<Pointd> pts;
                bool rubish0, rubish1, is_2d;
                std::tie(pts, rubish0, rubish1, is_2d) = pointCloud2MsgToPtsVec<double>(pc_msg, time_field_multiplier_, true, {}, absolute_time_);
                if(pts.size() < 1)
                {
                    continue;
                }

                int64_t t_min = pts.front().t;
                int64_t t_max = pts.front().t;
                for(const auto& pt : pts)
                {
                    t_min = std::min(t_min, pt.t);
                    t_max = std::max(t_max, pt.t);
                }

                // Wait for the odometry (integrated in the subscription callbacks) to cover the scan
                PcStatus status = PcStatus::WAITING;
                std::vector<OdomSample> odom_window;
                {
                    std::unique_lock<std::mutex> lock(data_mutex_);
                    ++num_pc_waiting_;
                    odom_cv_.wait_for(lock, std::chrono::duration<double>(max_pc_wait_time_),
                        [&](){
                            status = this->pcStatus(t_min, t_max);
                            return !running_.load() || (status != PcStatus::WAITING);
                        });
                    --num_pc_waiting_;
                    if(status == PcStatus::READY)
                    {
                        this->getOdomWindow(t_min, t_max, odom_window);
                    }
                }
                if(!running_.load())
                {
                    return;
                }
                if(status != PcStatus::READY)
                {
                    RCLCPP_WARN(this->get_logger(), "Dropping a point cloud: the odometry does not cover it (%s)",
                            (status == PcStatus::TOO_OLD) ? "too old" : "timed out");
                    continue;
                }

                // The points are undistorted in the IMU frame at the end of the scan, as done by
                // the lidar_scan_odometry node
                Vec3 pos_ref = Vec3::Zero();
                Mat3 rot_ref = Mat3::Identity();
                if(!interpolateOdom(odom_window, t_max, pos_ref, rot_ref))
                {
                    RCLCPP_WARN(this->get_logger(), "Dropping a point cloud: could not interpolate the reference pose");
                    continue;
                }
                const Mat3 rot_ref_inv = rot_ref.transpose();
                const Mat3 R_imu_lidar = T_imu_lidar_.block<3, 3>(0, 0);
                const Vec3 p_imu_lidar = T_imu_lidar_.block<3, 1>(0, 3);

                std::vector<Pointd> pts_corrected;
                pts_corrected.reserve(pts.size());
                for(const auto& pt : pts)
                {
                    Vec3 pos_t = Vec3::Zero();
                    Mat3 rot_t = Mat3::Identity();
                    if(!interpolateOdom(odom_window, pt.t, pos_t, rot_t))
                    {
                        continue;
                    }
                    const Vec3 p_imu = R_imu_lidar * pt.vec3() + p_imu_lidar;
                    const Vec3 p_odom = rot_t * p_imu + pos_t;
                    const Vec3 p_ref = rot_ref_inv * (p_odom - pos_ref);
                    pts_corrected.push_back(Pointd(p_ref, pt.t, pt.i, pt.channel, pt.type));
                }

                this->publishPc(t_max, pts_corrected);
                this->publishPose(t_max, pos_ref, logMap(rot_ref));
            }
        }

        // The undistorted points are expressed in the IMU/body frame at the end of the scan: the
        // lidar extrinsic has already been applied during the undistortion
        void publishPc(const int64_t t, const std::vector<Pointd>& pc)
        {
            sensor_msgs::msg::PointCloud2 pc_msg = ptsVecToPointCloud2MsgInternal(pc, "imu", rclcpp::Time(t));
            pc_pub_->publish(pc_msg);
        }

        // Pose of the IMU/body frame at the end of the scan ("imu_head" is the same physical frame
        // as "imu", only taken at the scan reference time)
        void publishPose(const int64_t t, const Vec3& pos, const Vec3& rot)
        {
            const rclcpp::Time new_time(t);

            geometry_msgs::msg::TransformStamped transform_stamped;
            transform_stamped.header.stamp = new_time;
            transform_stamped.header.frame_id = "odom";
            transform_stamped.child_frame_id = "imu";
            transform_stamped.transform = mat4ToTransform(posRotToTransform(pos, rot));

            nav_msgs::msg::Odometry odom_msg;
            odom_msg.header.stamp = new_time;
            odom_msg.header.frame_id = "odom";
            odom_msg.child_frame_id = "imu_head";
            odom_msg.pose.pose.position.x = pos[0];
            odom_msg.pose.pose.position.y = pos[1];
            odom_msg.pose.pose.position.z = pos[2];
            odom_msg.pose.pose.orientation = transform_stamped.transform.rotation;

            if(publish_tf_)
            {
                geometry_msgs::msg::TransformStamped map_to_odom;
                map_to_odom.header.stamp = new_time;
                map_to_odom.header.frame_id = "map";
                map_to_odom.child_frame_id = "odom";
                map_to_odom.transform = mat4ToTransform(Mat4::Identity());
                br_->sendTransform(map_to_odom);
                br_->sendTransform(transform_stamped);
            }
            global_odom_pub_->publish(transform_stamped);
            odom_twist_pub_->publish(odom_msg);
        }

        double wheel_scale_ = 1.0;
        double encoder_wrap_ = 16777216.0;
        Mat4 T_imu_lidar_ = Mat4::Identity();
        Mat4 T_imu_wheel_ = Mat4::Identity();
        Mat4 T_wheel_imu_ = Mat4::Identity();

        Vec3 gyr_bias_ = Vec3::Zero();
        bool estimate_gyr_bias_ = true;
        double zero_vel_time_ = 3.0;
        double bias_use_time_ = 2.0;
        double bias_max_time_ = 10.0;
        double zero_vel_rot_thr_ = 1e-3;

        double init_yaw_ = M_PI/2.0;
        bool undistort_pc_ = true;
        double time_field_multiplier_ = 1e-9;
        bool absolute_time_ = false;
        bool publish_tf_ = true;
        double odom_buffer_time_ = 5.0;
        double sensor_buffer_time_ = 2.0;
        double max_pc_wait_time_ = 2.0;

        // Encoder unwrapping state
        bool has_encoder_ = false;
        double last_raw_count_ = 0.0;
        double unwrapped_count_ = 0.0;

        // Odometry state (pose of the wheel frame in the odom frame)
        bool odom_initialised_ = false;
        int64_t current_time_ = 0;
        Vec3 pos_ = Vec3::Zero();
        Mat3 rot_ = Mat3::Identity();
        double total_travelled_distance_ = 0.0;

        std::deque<EncoderSample> encoder_samples_;
        std::deque<GyrSample> gyr_samples_;
        std::deque<GyrSample> bias_window_;
        std::deque<OdomSample> odom_history_;
        int num_pc_waiting_ = 0;

        std::mutex data_mutex_;
        std::condition_variable odom_cv_;

        std::mutex pc_mutex_;
        std::condition_variable pc_cv_;
        std::deque<sensor_msgs::msg::PointCloud2::ConstSharedPtr> pending_pc_;

        std::atomic<bool> running_{true};
        std::thread thread_;

        rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr encoder_sub_;
        rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr gyr_sub_;
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;

        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
        rclcpp::Publisher<geometry_msgs::msg::TransformStamped>::SharedPtr global_odom_pub_;
        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_twist_pub_;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pc_pub_;

        std::unique_ptr<tf2_ros::TransformBroadcaster> br_;
        std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_br_;
};


int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<WheelGyroOdometryNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
