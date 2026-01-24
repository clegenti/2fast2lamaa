#include "rclcpp/rclcpp.hpp"
#include "lice/ros_utils.h"

#include "sensor_msgs/msg/point_cloud.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"


class LidarConversionNode: public rclcpp::Node
{
    public:
        LidarConversionNode()
            : Node("lidar_conversion")
        {

            // Read parameters
            ascending_time_ = readRequiredFieldBool(this, "ascending_time");
            distortion_free_ = readFieldBool(this, "distortion_free", false);
            pc_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("converted_pointcloud", 10);
            pc_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud>(
                "input_pointcloud", 100,
                std::bind(&LidarConversionNode::pointCloudCallback, this, std::placeholders::_1));
            scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
                "input_laserscan", 100,
                std::bind(&LidarConversionNode::laserScanCallback, this, std::placeholders::_1));
        }

    private:
        // Subscriber for pointcloud messages
        rclcpp::Subscription<sensor_msgs::msg::PointCloud>::SharedPtr pc_sub_;
        // Subscriber for lidar scan messages
        rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;

        // Publisher for converted pointcloud messages
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pc_pub_;

        // Average time between scans
        double sum_scan_time_ = 0;
        // Number of scans received
        int scan_count_ = 0;
        // Time of the last received scan
        rclcpp::Time last_scan_time_;

        std::vector<Pointd> first_pc_;

        bool ascending_time_ = true;
        bool distortion_free_ = false;
        bool need_time_scaling_ = false;


        std::vector<Pointd> convertPointCloud(const sensor_msgs::msg::PointCloud::ConstSharedPtr pc_msg)
        {
            std::vector<Pointd> pts;
            pts.reserve(pc_msg->points.size());
            bool has_time_field = false;
            for(const auto& point : pc_msg->points)
            {
                double x = point.x;
                double y = point.y;
                double z = 0.0;
                int64_t t = 0;
                float intensity = 0.0;
                for(const auto& field : pc_msg->channels)
                {
                    if(field.name == "intensity" && field.values.size() == pc_msg->points.size())
                    {
                        intensity = field.values[&point - &pc_msg->points[0]];
                    }
                    else if((field.name == "time" || field.name == "timestamp" || field.name == "stamp" || field.name == "t") && field.values.size() == pc_msg->points.size())
                    {
                        t = static_cast<int64_t>(field.values[&point - &pc_msg->points[0]] * 1e9); // assuming time is in seconds
                        has_time_field = true;
                    }
                }
                pts.emplace_back(x, y, z, t, intensity);
            }
            if(!has_time_field && pts.size() > 0)
            {
                need_time_scaling_ = true;
                pts[0].t = rclcpp::Time(pc_msg->header.stamp).nanoseconds();
            }
            return pts;
        }


        std::vector<Pointd> convertLaserScan(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg)
        {
            need_time_scaling_ = true;
            std::vector<Pointd> pts;
            size_t num_points = static_cast<size_t>((scan_msg->angle_max - scan_msg->angle_min) / scan_msg->angle_increment) + 1;
            pts.reserve(num_points);
            for(size_t i = 0; i < num_points; ++i)
            {
                double angle = scan_msg->angle_min + i * scan_msg->angle_increment;
                double range = scan_msg->ranges[i];
                if(std::isfinite(range))
                {
                    double x = range * cos(angle);
                    double y = range * sin(angle);
                    double z = 0.0;
                    int64_t t = static_cast<int64_t>( (scan_msg->header.stamp.sec * 1e9 + scan_msg->header.stamp.nanosec) + (i * scan_msg->time_increment * 1e9) );
                    pts.emplace_back(x, y, z, t, 0.0);
                }
            }
            if(pts.size() > 0)
            {
                pts[0].t = rclcpp::Time(scan_msg->header.stamp).nanoseconds();
            }
            return pts;
        }


        void updateAverageScanTime(const rclcpp::Time& current_time)
        {
            if(scan_count_ > 0)
            {
                double time_diff = (current_time - last_scan_time_).seconds();
                sum_scan_time_ += time_diff;
            }
            last_scan_time_ = current_time;
            scan_count_++;
        }

        void scaleTimeField(std::vector<Pointd>& pts, const double period)
        {
            if(pts.size() < 2)
            {
                return;
            }
            if(distortion_free_)
            {
                for(auto& pt : pts)
                {
                    pt.t = pts.front().t;
                }
                return;
            }

            // Compute the percentage along the scan for each point (azimuth based) and scale the time field accordingly (using the first point time as reference)
            int64_t start_time = pts.front().t;
            double angle_start = std::atan2(pts.front().y, pts.front().x);
            double angle_end = std::atan2(pts.back().y, pts.back().x);
            double angle_diff = angle_end - angle_start;
            if(angle_diff <= 0)
            {
                angle_diff += 2 * M_PI;
            }

            pts[0].t = start_time;
            std::vector<double> angles;
            angles.push_back(0.0);
            for(size_t i = 1; i < pts.size(); ++i)
            {
                double delta_angle = std::atan2(pts[i].y, pts[i].x) - std::atan2(pts[i-1].y, pts[i-1].x);
                if(delta_angle < 0)
                {
                    delta_angle += 2 * M_PI;
                }
                angles.push_back(angles.back() + delta_angle);
            }

            // Now scale the time field to match the period (normalizing fist to the angle_diff)
            for(size_t i = 1; i < pts.size(); ++i)
            {
                double angle = (angles[i] / angles.back()) * angle_diff;
                double percentage = angle / (2 * M_PI);
                if(!ascending_time_)
                {
                    percentage = 1.0 - percentage;
                }
                pts[i].t = static_cast<int64_t>(percentage * period * 1e9) + start_time;
            }
        }

        
        void pointCloudCallback(const sensor_msgs::msg::PointCloud::ConstSharedPtr pc_msg)
        {
            auto current_time = rclcpp::Time(pc_msg->header.stamp);
            updateAverageScanTime(current_time);
            std::vector<Pointd> pts = convertPointCloud(pc_msg);
            if(scan_count_ == 1)
            {
                first_pc_ = pts;
                return;
            }
            if(need_time_scaling_)
            {
                double avg_period = sum_scan_time_ / (scan_count_ - 1);
                if(scan_count_ == 2)
                {
                    auto temp_time = rclcpp::Time(first_pc_.front().t);
                    scaleTimeField(first_pc_, avg_period);
                    pc_pub_->publish(ptsVecToPointCloud2MsgInternal(first_pc_, "lidar", temp_time));
                }
                scaleTimeField(pts, avg_period);
                pc_pub_->publish(ptsVecToPointCloud2MsgInternal(pts, "lidar", current_time));
            }
            return;
        }

        void laserScanCallback(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg)
        {
            auto current_time = rclcpp::Time(scan_msg->header.stamp);
            updateAverageScanTime(current_time);
            std::vector<Pointd> pts = convertLaserScan(scan_msg);
            if(scan_count_ == 1)
            {
                first_pc_ = pts;
                return;
            }
            if(need_time_scaling_)
            {
                double avg_period = sum_scan_time_ / (scan_count_ - 1);
                if(scan_count_ == 2)
                {
                    auto temp_time = rclcpp::Time(first_pc_.front().t);
                    scaleTimeField(first_pc_, avg_period);
                    pc_pub_->publish(ptsVecToPointCloud2MsgInternal(first_pc_, "lidar", temp_time));
                }
                scaleTimeField(pts, avg_period);
                pc_pub_->publish(ptsVecToPointCloud2MsgInternal(pts, "lidar", current_time));
            }
            return;
        }
        
};




int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<LidarConversionNode>();

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}