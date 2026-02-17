#include "rclcpp/rclcpp.hpp"
#include "ros_utils.h"
#include "lice/utils.h"
#include "lice/math_utils.h"
#include "lice/pointcloud_utils.h"
#include "lice/map_distance_field.h"

#include <filesystem>
#include <fstream>

#include "ffastllamaa/msg/submap_info.hpp"

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

        std::vector<std::string> map_paths_;
        std::vector<std::string> traj_files_;
        std::string scans_folder_ = "";
        std::string output_folder_ = "";

        std::vector<std::shared_ptr<MapDistField> > maps_;

        MapDistFieldOptions map_options_;

        std::vector<std::shared_ptr<Vec7> > poses_;
        ankerl::unordered_dense::map<int64_t, std::shared_ptr<Vec7> > time_to_pose_map_;

        // Add the storage of the submap images
        

        rclcpp::Subscription<ffastllamaa::msg::SubmapInfo>::SharedPtr sub_;


        void submapCallback(const ffastllamaa::msg::SubmapInfo::SharedPtr msg)
        {
            RCLCPP_INFO(this->get_logger(), "Received submap info: %s", msg->traj_file.c_str());
            map_paths_.push_back(msg->ply_file);
            traj_files_.push_back(msg->traj_file);

            if(scans_folder_ == "" && msg->scan_folder != "")
            {
                scans_folder_ = msg->scan_folder;
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



            if(free_space_carving_radius_ > 0.0)
            {
                saveOriginalMap(msg->ply_file);
                cleanMap(maps_.back(), scans_folder_, trajectory);
                maps_.back()->writeMap(msg->ply_file);
                RCLCPP_INFO(this->get_logger(), "Cleaned map saved to: %s", msg->ply_file.c_str());
            }
            writeFullTrajectory();
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
            Vec7 last_pose;
            int64_t last_timestamp = -1;
            for(size_t i = 0; i < trajectory.size(); i++)
            {
                int64_t timestamp = trajectory[i].first;
                Vec7 pose = trajectory[i].second;

                if(time_to_pose_map_.find(timestamp) == time_to_pose_map_.end())
                {
                    if(i > 0 && last_timestamp != -1)
                    {
                        Mat4 last_pose_mat = posQuatToTransform(last_pose);
                        Mat4 new_pose_mat = posQuatToTransform(pose);
                        Mat4 delta_mat = last_pose_mat.inverse() * new_pose_mat;

                        poses_.emplace_back(std::make_shared<Vec7>(transformToPosQuat<double>(posQuatToTransform<double>(*time_to_pose_map_[last_timestamp]) * delta_mat)));
                        time_to_pose_map_[timestamp] = poses_.back();
                    }
                    else
                    {
                        std::cout << "Adding initial pose i = " << i << " timestamp " << timestamp << std::endl;
                        poses_.emplace_back(std::make_shared<Vec7>(pose));
                        time_to_pose_map_[timestamp] = poses_.back();
                    }
                }
                last_pose = pose;
                last_timestamp = timestamp;
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
            for (const auto& [timestamp, pose_ptr] : time_to_pose_map_)
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
            RCLCPP_INFO(this->get_logger(), "Wrote full trajectory with %zu poses to file: %s", time_to_pose_map_.size(), full_traj_file.c_str());
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


};




int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PoseGraphNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}