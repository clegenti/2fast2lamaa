#include "rclcpp/rclcpp.hpp"
#include "ros_utils.h"
#include "lice/utils.h"
#include "lice/math_utils.h"
#include "lice/pointcloud_utils.h"
#include "lice/map_distance_field.h"

#include <filesystem>
#include <fstream>

#include "ffastllamaa/msg/submap_info.hpp"

#include <opencv2/opencv.hpp>

const float kCapHeight = 0.5; // Maximum height to consider for the submap images


class PoseGraphNode: public rclcpp::Node
{
    public:
        PoseGraphNode()
            : Node("pose_graph_node")
        {
            RCLCPP_INFO(this->get_logger(), "Pose Graph Node has been started.");
            
            free_space_carving_radius_ = readFieldDouble(this, "free_space_carving_radius", 40.0);

            
            sub_ = this->create_subscription<ffastllamaa::msg::SubmapInfo>(
                "/submap_info", 10,
                std::bind(&PoseGraphNode::submapCallback, this, std::placeholders::_1)
            );


            map_options_.cell_size = -1.0;
            map_options_.free_space_carving = free_space_carving_radius_ > 0.0;
            map_options_.free_space_carving_radius = free_space_carving_radius_;
            map_options_.last_scan_carving = false;
            map_options_.min_range = 0.0;
            map_options_.max_range = std::numeric_limits<double>::max();

        }

    private:
        double free_space_carving_radius_ = -1.0;
        double image_res_ = 0.5;
        double map_res_ = 0.15;

        std::vector<std::string> map_paths_;
        std::vector<std::string> traj_files_;
        std::string scans_folder_ = "";
        std::string output_folder_ = "";

        std::vector<std::shared_ptr<MapDistField> > maps_;
        std::vector<std::pair<int64_t, int64_t> > map_time_ranges_;
        std::vector<Mat4> submap_original_poses_;
        std::vector<Mat2_3> cam_mats_;

        MapDistFieldOptions map_options_;

        std::map<int64_t, std::shared_ptr<Vec7> > time_and_pose_;
        int64_t last_odometry_time_ = -1;
        bool has_been_gravity_aligned_ = false;

        // Add the storage of the submap images
        std::vector<cv::Mat> submap_images_;


        rclcpp::Subscription<ffastllamaa::msg::SubmapInfo>::SharedPtr sub_;


        void submapCallback(const ffastllamaa::msg::SubmapInfo::SharedPtr msg)
        {
            RCLCPP_INFO(this->get_logger(), "Received submap info: %s", msg->traj_file.c_str());
            map_paths_.push_back(msg->ply_file);
            traj_files_.push_back(msg->traj_file);

            if(scans_folder_ == "" && msg->scan_folder != "")
            {
                scans_folder_ = msg->scan_folder;
                image_res_ = msg->map_res*2.5;
                map_res_ = msg->map_res;
            }
            if(map_options_.cell_size < 0.0)
            {
                map_options_.cell_size = msg->map_res;
            }
            if(output_folder_ == "")
            {
                output_folder_ = msg->raw_output_folder;
            }

            // Load the submap
            maps_.emplace_back(std::make_shared<MapDistField>(map_options_, nullptr));
            maps_.back()->loadMap(msg->ply_file);

            // Load the trajectory
            std::vector<std::pair<int64_t, Vec7> > trajectory = readTrajectoryFile(msg->traj_file);
            addSubmapTrajectoryToState(trajectory);

            map_time_ranges_.emplace_back(trajectory.front().first, trajectory.back().first);
            submap_original_poses_.emplace_back(posQuatToTransform(trajectory.front().second));
            cam_mats_.emplace_back(Mat2_3::Zero());

            if(free_space_carving_radius_ > 0.0)
            {
                saveOriginalMap(msg->ply_file);
                cleanMap(maps_.back(), scans_folder_, trajectory);
                maps_.back()->writeMap(msg->ply_file);
                RCLCPP_INFO(this->get_logger(), "Cleaned map saved to: %s", msg->ply_file.c_str());
            }

            Vec3 gravity = Vec3(msg->gravity[0], msg->gravity[1], msg->gravity[2]);
            if(!has_been_gravity_aligned_ && gravity.norm() > 1.0)
            {
                alignGravity(gravity);
                has_been_gravity_aligned_ = true;
            }

            // Create the submap image
            cv::Mat image = getSubmapImage(maps_.size() - 1);
            submap_images_.push_back(image);
            std::cout << "Created submap image for map " << maps_.size() - 1 << std::endl;

            writeFullTrajectory();
        }


        void alignGravity(const Vec3& gravity)
        {
            // Align the gravity vector with the Z-axis
            Vec3 z_axis(0.0, 0.0, 1.0);
            Vec3 gravity_normalized = gravity.normalized();
            Vec3 rotation_axis = gravity_normalized.cross(z_axis);
            double angle = std::acos(gravity_normalized.dot(z_axis));
            if(rotation_axis.norm() < 1e-6)
            {
                if(gravity_normalized.dot(z_axis) < 0)
                {
                    // Gravity is opposite to Z-axis, rotate 180 degrees around X-axis
                    rotation_axis = Vec3(1.0, 0.0, 0.0);
                    angle = M_PI;
                }
                else
                {
                    rotation_axis = Vec3(0.0, 0.0, 1.0); // No rotation needed, but we need a valid axis
                    angle = 0.0;
                }
            }
            else
            {
                rotation_axis.normalize();
            }
            Mat3 rotation_matrix = expMap(rotation_axis * angle);

            // Apply the rotation to all poses in the state
            Mat4 transform = Mat4::Identity();
            transform.block<3,3>(0,0) = rotation_matrix;

            for(auto& [timestamp, pose_ptr] : time_and_pose_)
            {
                Mat4 pose_mat = posQuatToTransform(*pose_ptr);
                Mat4 rotated_pose_mat = transform * pose_mat;
                *pose_ptr = transformToPosQuat(rotated_pose_mat);
            }
        }



        void saveOriginalMap(const std::string& map_path)
        {
            std::string original_map_path = map_path;
            original_map_path.replace(original_map_path.end() - 4, original_map_path.end(), "_original.ply");
            std::filesystem::copy_file(map_path, original_map_path, std::filesystem::copy_options::overwrite_existing);
            RCLCPP_INFO(this->get_logger(), "Copied original map to: %s", original_map_path.c_str());
        }

        std::vector<std::pair<int64_t, Vec7> >readTrajectoryFile(const std::string& traj_file)
        {
            std::vector<std::pair<int64_t, Vec7> > trajectory;
            std::ifstream file(traj_file);
            if (!file.is_open())
            {
                RCLCPP_ERROR(this->get_logger(), "Could not open trajectory file: %s", traj_file.c_str());
                return trajectory;
            }
            std::string line;
            // Skip the header line
            std::getline(file, line);
            while (std::getline(file, line))
            {
                std::istringstream ss(line);
                std::string token;
                std::vector<std::string> tokens;
                while (std::getline(ss, token, ','))
                {
                    tokens.push_back(token);
                }
                if (tokens.size() != 7)
                {
                    RCLCPP_ERROR(this->get_logger(), "Invalid trajectory file format: %s", traj_file.c_str());
                    continue;
                }
                int64_t timestamp = std::stoll(tokens[0]);
                Vec7 pose = transformToPosQuat(posRotToTransform(Vec3(
                    std::stod(tokens[1]),
                    std::stod(tokens[2]),
                    std::stod(tokens[3])
                ), Vec3(
                    std::stod(tokens[4]),
                    std::stod(tokens[5]),
                    std::stod(tokens[6])
                )));
                trajectory.emplace_back(timestamp, pose);
            }
            file.close();
            RCLCPP_INFO(this->get_logger(), "Loaded %zu poses from trajectory file: %s", trajectory.size(), traj_file.c_str());
            return trajectory;
        }


        void addSubmapTrajectoryToState(const std::vector<std::pair<int64_t, Vec7> >& trajectory)
        {
            for(size_t i = 0; i < trajectory.size(); i++)
            {
                int64_t timestamp = trajectory[i].first;
                Vec7 pose = trajectory[i].second;

                if(time_and_pose_.size() == 0)
                {
                    std::cout << "Adding first pose with timestamp " << timestamp << std::endl;
                    auto pose_ptr = std::make_shared<Vec7>(pose);
                    time_and_pose_[timestamp] = pose_ptr;
                    continue;
                }
                if(i == 0 && (time_and_pose_.find(timestamp) == time_and_pose_.end()))
                {
                    std::string error_message = "First pose of the submap trajectory has a timestamp that does not exist in the state. This means that the submaps do not overlap and that should not happen. Timestamp: " + std::to_string(timestamp);
                    RCLCPP_ERROR(this->get_logger(), "%s", error_message.c_str());
                    throw std::runtime_error(error_message);
                }


                if(time_and_pose_.find(timestamp) == time_and_pose_.end())
                {
                    Mat4 last_pose_mat = posQuatToTransform(trajectory[i-1].second);
                    Mat4 new_pose_mat = posQuatToTransform(pose);
                    Mat4 delta_mat = last_pose_mat.inverse() * new_pose_mat;

                    auto pose = std::make_shared<Vec7>(transformToPosQuat<double>(posQuatToTransform<double>(*time_and_pose_[trajectory[i-1].first]) * delta_mat));
                    time_and_pose_[timestamp] = pose;
                }
            }
            RCLCPP_INFO(this->get_logger(), "Added %zu poses to the state from trajectory.", trajectory.size());
        }



        void writeFullTrajectory()
        {
            if(output_folder_ == "")
            {
                RCLCPP_ERROR(this->get_logger(), "Output folder is not set. Cannot write full trajectory.");
                return;
            }
            std::string full_traj_file = output_folder_ + "/pose_graph_trajectory.csv";
            std::ofstream file(full_traj_file, std::ios::out | std::ios::trunc);
            if (!file.is_open())
            {
                RCLCPP_ERROR(this->get_logger(), "Could not open full trajectory file for writing: %s", full_traj_file.c_str());
                return;
            }
            file << "timestamp, x, y, z, rx, ry, rz" << std::endl; // Header
            for (const auto& [timestamp, pose_ptr] : time_and_pose_)
            {
                Vec6 pos_rot_vec = posQuatToPosRot(*pose_ptr);
                file << timestamp << ", "
                     << pos_rot_vec(0) << ", "
                     << pos_rot_vec(1) << ", "
                     << pos_rot_vec(2) << ", "
                     << pos_rot_vec(3) << ", "
                     << pos_rot_vec(4) << ", "
                     << pos_rot_vec(5) << std::endl;
            }
            file.close();
            RCLCPP_INFO(this->get_logger(), "Wrote full trajectory with %zu poses to file: %s", time_and_pose_.size(), full_traj_file.c_str());
        }



        void cleanMap(std::shared_ptr<MapDistField> map,
                      const std::string& scans_folder,
                      const std::vector<std::pair<int64_t, Vec7> >& trajectory)
        {
            if(scans_folder == "")
            {
                RCLCPP_ERROR(this->get_logger(), "Scans folder is not set. Cannot perform map cleaning.");
                return;
            }
            int count = 0;
            for(const auto& [timestamp, pose] : trajectory)
            {
                std::cout << "Processing scan " << count++ << " out of " << trajectory.size() << std::endl;
                std::string scan_file = scans_folder + "/" + std::to_string(timestamp) + ".ply";
                if(!std::filesystem::exists(scan_file))
                {
                    RCLCPP_WARN(this->get_logger(), "Scan file does not exist: %s", scan_file.c_str());
                    continue;
                }
                // Load the scan
                std::vector<Pointd> scan = loadPointCloudFromPly(scan_file);
                if(scan.size() == 0)
                {
                    RCLCPP_WARN(this->get_logger(), "Scan file is empty: %s", scan_file.c_str());
                    continue;
                }
                // Get the pose
                Mat4 pose_mat = posQuatToTransform(pose);
                // Carve free space
                map->freeSpaceCarving(scan, pose_mat);
                RCLCPP_INFO(this->get_logger(), "Carved free space from scan: %s", scan_file.c_str());
            }
        }


        cv::Mat getSubmapImage(int map_id)
        {
            std::shared_ptr<MapDistField> map = maps_[map_id];
            std::vector<Pointd> pts = map->getPts();
            if(pts.size() == 0)
            {
                RCLCPP_ERROR(this->get_logger(), "Map has no points. Cannot create submap image.");
                throw std::runtime_error("Map has no points. Cannot create submap image.");
            }

            // Get the points in the reference frame of the first submap_pose
            Mat4 first_submap_pose = submap_original_poses_[map_id];
            Mat4 current_submap_pose = posQuatToTransform(*time_and_pose_[map_time_ranges_[map_id].first]);
            Mat4 transform = current_submap_pose * first_submap_pose.inverse();
            std::vector<Vec3> transformed_pts;
            double min_x = std::numeric_limits<double>::max();
            double max_x = std::numeric_limits<double>::lowest();
            double min_y = std::numeric_limits<double>::max();
            double max_y = std::numeric_limits<double>::lowest();
            for(const auto& pt : pts)
            {
                Vec3 transformed_vec = transform.block<3,3>(0,0) * pt.vec3() + transform.block<3,1>(0,3);
                transformed_vec(2) -= current_submap_pose(2,3); // Remove the height to create a 2D image
                transformed_pts.emplace_back(transformed_vec(0), transformed_vec(1), transformed_vec(2));
                min_x = std::min(min_x, transformed_vec(0));
                max_x = std::max(max_x, transformed_vec(0));
                min_y = std::min(min_y, transformed_vec(1));
                max_y = std::max(max_y, transformed_vec(1));
            }

            // Create an image with the given resolution
            int width = static_cast<int>(std::ceil((max_x - min_x) / image_res_))+1;
            int height = static_cast<int>(std::ceil((max_y - min_y) / image_res_))+1;
            MatX counter = MatX::Zero(height, width);
            MatX sum_squared = MatX::Zero(height, width);
            MatX sum = MatX::Zero(height, width);
            cv::Mat image(height, width, CV_32F, cv::Scalar(0.0f));

            Mat2_3 cam_mat;
            cam_mat << 1.0/image_res_, 0.0, -min_x/image_res_,
                            0.0, 1.0/image_res_, -min_y/image_res_;
            cam_mats_[map_id] = cam_mat;

            // Fill the image with the height values
            for(const auto& pt : transformed_pts)
            {
                Vec2 cam_pt = cam_mat * Vec3(pt(0), pt(1), 1.0);
                int x = static_cast<int>(std::floor(cam_pt(0)));
                int y = static_cast<int>(std::floor(cam_pt(1)));
                if(x >= 0 && x < width && y >= 0 && y < height)
                {
                    counter(y, x) += 1.0f;
                    sum(y, x) += pt(2);
                    sum_squared(y, x) += pt(2) * pt(2);
                }
            }
            for(int y = 1; y < height-1; ++y)
            {
                for(int x = 1; x < width-1; ++x)
                {
                    int local_count = 0;
                    double local_sum = 0;
                    if(counter(y, x) == 0)
                    {
                        continue;
                    }
                    if(counter(y-1, x) > 0)
                    {
                        local_count++;
                        local_sum += sum(y-1, x) / counter(y-1, x);
                    }
                    if(counter(y+1, x) > 0)
                    {
                        local_count++;
                        local_sum += sum(y+1, x) / counter(y+1, x);
                    }
                    if(counter(y, x-1) > 0)
                    {
                        local_count++;
                        local_sum += sum(y, x-1) / counter(y, x-1);
                    }
                    if(counter(y, x+1) > 0)
                    {
                        local_count++;
                        local_sum += sum(y, x+1) / counter(y, x+1);
                    }
                    image.at<float>(y, x) = static_cast<float>((local_sum - local_count*sum(y, x) / counter(y, x))/(4.0*kCapHeight));
                    if(image.at<float>(y, x) < -1.0f)
                    {
                        image.at<float>(y, x) = -1.0f;
                    }
                    else if(image.at<float>(y, x) > 1.0f)
                    {
                        image.at<float>(y, x) = 1.0f;
                    }
                }
            }
            // Apply a simple gaussian blur to smooth the image
            cv::GaussianBlur(image, image, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);


            // For debug, save the image
            std::string image_file = output_folder_ + "/submap_image_" + std::to_string(map_id) + ".png";
            cv::Mat debug_img;
            image.convertTo(debug_img, CV_8U, 127.5, 127.5); // Scale to [0, 255] for visualization
            cv::imwrite(image_file, debug_img);
            std::cout << "Saved submap image to: " << image_file << std::endl;

            return image;
        }


};




int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PoseGraphNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}