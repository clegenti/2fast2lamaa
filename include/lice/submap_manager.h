#pragma once

#include "types.h"
#include "math_utils.h"
#include "map_distance_field.h"
#include <filesystem>
#include <thread>
#include <fstream>
#include "utils.h"


const double kMinNodeDist = 1.0;
const int kNumAdjacentNodesToCheck = 20;


class SubmapManager
{
    public:
        SubmapManager(const MapDistFieldOptions options, const bool localization, const bool using_submaps, const double submap_length, const double submap_overlap, const std::string& map_path, const bool reverse_path=false)
            : options_(options)
            , localization_(localization)
            , submap_length_(submap_length)
            , submap_overlap_(submap_overlap)
            , using_submaps_(using_submaps)
            , map_path_(map_path)
            , reverse_path_(reverse_path)
        {
            if(submap_length_ > 0.0 && submap_overlap_ >= 1.0)
            {
                throw std::runtime_error("Submap overlap must be less than 1.0");
            }

            // Check if map_path_ finishes /, if not add it
            if(!map_path_.empty() && map_path_.back() != '/')
            {
                map_path_ += "/";
            }
            // If the map path does not exist, create it
            if(!map_path_.empty() && !std::filesystem::exists(map_path_))
            {
                std::filesystem::create_directories(map_path_);
            }


            if(localization_)
            {
                options_.min_range = 0;
                options_.max_range = std::numeric_limits<double>::max();
            }

            current_map_ = std::make_shared<MapDistField>(options_);
            current_map_->set2D(is_2d_);

            if(localization_)
            {
                std::cout << "Loading map from: " << map_path_ << std::endl;
                if(using_submaps_)
                {
                    // Read the submap files
                    int map_ptr = 0;
                    bool loop = true;
                    std::vector<int64_t> prev_times;
                    std::map<int64_t, int> time_to_index;
                    std::vector<std::pair<int64_t, int64_t>> overlaps;
                    while(loop)
                    {
                        std::string ply_path = map_path_ + "submap_" + std::to_string(map_ptr) + ".ply";
                        std::string traj_path = map_path_ + "trajectory_submap_" + std::to_string(map_ptr) + ".csv";

                        // If both map and trajectory exist, load them
                        if(std::filesystem::exists(ply_path) && std::filesystem::exists(traj_path))
                        {
                            submap_paths_.push_back(ply_path);
                            overlaps.push_back({std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::min()});
                            
                            // Load the trajectory
                            std::ifstream traj_file(traj_path);
                            if(!traj_file)
                            {
                                throw std::runtime_error("Failed to open trajectory file: " + traj_path);
                            }
                            std::string line;
                            // Skip the header
                            std::getline(traj_file, line);
                            int64_t temp_time;
                            Vec3 temp_pos;
                            while(std::getline(traj_file, line))
                            {
                                std::istringstream ss(line);
                                std::string token;
                                std::vector<std::string> tokens;
                                while(std::getline(ss, token, ','))
                                {
                                    tokens.push_back(token);
                                }
                                // Process the tokens as needed
                                temp_time = std::stoll(tokens[0]);
                                temp_pos(0) = std::stod(tokens[1]);
                                temp_pos(1) = std::stod(tokens[2]);
                                temp_pos(2) = std::stod(tokens[3]);

                                // Add the time and position to the graph nodes
                                if(prev_times.size() == 0 || (temp_time > prev_times.back()))
                                {
                                    time_to_index[temp_time] = prev_times.size();
                                    prev_times.push_back(temp_time);
                                    graph_nodes_.push_back({temp_pos, map_ptr});
                                }
                                // If the time already exists, it means there is an overlap
                                else
                                {
                                    overlaps.back().first = std::min(overlaps.back().first, temp_time);
                                    overlaps.back().second = std::max(overlaps.back().second, temp_time);
                                }
                            }
                            traj_file.close();
                            
                        }
                        else
                        {
                            loop = false;
                        }
                        map_ptr++;
                    }

                    // Correct the map index at the overlaps
                    for(size_t i = 0; i < overlaps.size(); i++)
                    {
                        if(overlaps[i].first != std::numeric_limits<int64_t>::max())
                        {
                            // Change the map index of the nodes in the first half of the overlap to the previous map index
                            int mid_index = (time_to_index[overlaps[i].first] + time_to_index[overlaps[i].second]) / 2;
                            for(int j = mid_index; j <= time_to_index[overlaps[i].second]; j++)
                            {
                                graph_nodes_[j].second = i;
                            }
                        }

                    }

                    // Prune the graph nodes that are too close to each other
                    std::vector<std::pair<Vec3, int>> pruned_graph_nodes;
                    Vec3 last_node = graph_nodes_[0].first;
                    for(size_t i = 1; i < graph_nodes_.size(); i++)
                    {
                        if((graph_nodes_[i].first - last_node).norm() > kMinNodeDist)
                        {
                            pruned_graph_nodes.push_back(graph_nodes_[i]);
                            last_node = graph_nodes_[i].first;
                        }
                    }
                    graph_nodes_ = pruned_graph_nodes;

                    num_submaps_ = submap_paths_.size();

                    if(!reverse_path)
                    {
                        current_map_->loadMap(submap_paths_[0]);
                        current_map_id_ = 0;
                        current_node_id_ = 0;
                    }
                    else
                    {
                        current_map_->loadMap(submap_paths_.back());
                        current_map_id_ = num_submaps_ - 1;
                        current_node_id_ = graph_nodes_.size() - 1;
                    }

                }
                else
                {
                    current_map_->loadMap(map_path_ + "map.ply");
                }
            }
        }
        ~SubmapManager() {}


        // Use the current map to register the points
        Mat4 registerPts(const std::vector<Pointd>& pts, const Mat4& prior, const int64_t current_time, const bool approximate=false, const double loss_scale=0.5, const int max_iterations=12)
        {
            if(current_map_ == nullptr)
            {
                throw std::runtime_error("No current map available for registration");
            }

            Mat4 updated_pose = current_map_->registerPts(pts, prior, current_time, approximate, loss_scale, max_iterations);
            last_registered_time_ = current_time;
            if(localization_ && using_submaps_ && graph_nodes_.size() > 0)
            {
                // Check if we need to change the current map based on the updated pose
                Vec3 current_pos = updated_pose.block<3,1>(0,3);
                int best_node_id = current_node_id_;
                double best_dist = (current_pos - graph_nodes_[current_node_id_].first).norm();
                // Check the next kNumAdjacentNodesToCheck nodes
                int start = reverse_path_ ? std::max(0, current_node_id_ - kNumAdjacentNodesToCheck) : current_node_id_;
                int end = reverse_path_ ? current_node_id_ : std::min((int)graph_nodes_.size(), current_node_id_ + kNumAdjacentNodesToCheck);
                for(int node_id = start; node_id < end; node_id++)
                {
                    double dist = (current_pos - graph_nodes_[node_id].first).norm();
                    if(dist < best_dist)
                    {
                        best_dist = dist;
                        best_node_id = node_id;
                    }
                }

                if(best_node_id != current_node_id_)
                {
                    int new_map_id = graph_nodes_[best_node_id].second;
                    if(new_map_id != current_map_id_)
                    {
                        std::cout << "Switching from submap " << current_map_id_ << " to submap " << new_map_id << "\n\n\n\n\n" << std::endl;
                        current_map_ = std::make_shared<MapDistField>(options_);
                        current_map_->loadMap(submap_paths_[new_map_id]);
                        current_map_->set2D(is_2d_);
                        current_map_id_ = new_map_id;
                    }
                }
                current_node_id_ = best_node_id;
            }
            return updated_pose;
        }


        // Add points to the current map (and next map if using submaps)
        void addPts(const std::vector<Pointd>& pts, const Mat4& pose, const int64_t time)
        {
            if((options_.scan_folder != "") && (!localization_))
            {
                // Create an anonymous function to save the scan in a separate thread
                StopWatch sw;
                sw.start();
                std::string scan_path = options_.scan_folder + "/" + std::to_string(time) + ".ply";
                auto save_scan = [](const std::vector<Pointd>& pts_in, const std::string& scan_path_in)
                {
                    StopWatch sw_in;
                    sw_in.start();
                    // Save the scan to the folder
                    savePointCloudToPly(scan_path_in, pts_in);
                    sw_in.stop();
                    sw_in.print("Time to save scan :");
                };
                // Launch the save_scan function in a separate thread
                std::thread scan_saving_thread(save_scan, pts, scan_path);
                scan_saving_thread.detach();
                sw.stop();
                sw.print("Time to launch scan saving thread: ");
            }

            if(localization_)
            {
                throw std::runtime_error("So far we cannot add point in localization mode");
            }

            if(current_map_ == nullptr)
            {
                throw std::runtime_error("No current map available to add points");
            }
            current_map_->addPts(pts, pose);
            if(last_registered_time_ >= 0 && time == last_registered_time_)
            {
                current_map_poses_.push_back({time, pose});
            }
            if(using_submaps_)
            {
                if((current_map_->getPathLength() > submap_length_ * (1.0 - submap_overlap_)) && (next_map_ == nullptr))
                {
                    next_map_ = std::make_shared<MapDistField>(options_);
                    next_map_->set2D(is_2d_);
                }
                if(next_map_)
                {
                    next_map_->addPts(pts, pose);
                    next_map_poses_.push_back({time, pose});
                }
                if(current_map_->getPathLength() > submap_length_)
                {
                    writeCurrentSubmap();
                    submap_counter_++;
                    current_map_ = next_map_;
                    current_map_poses_ = next_map_poses_;
                    next_map_ = nullptr;
                    next_map_poses_.clear();
                }
            }
                
        }



        // Get the current map points
        std::vector<Pointd> getPts()
        {
            if(current_map_ == nullptr)
            {
                throw std::runtime_error("No current map available");
            }
            return current_map_->getPts();
        }


        // Query the distance field at the given points
        std::vector<double> queryDistField(const std::vector<Vec3>& query_pts)
        {
            if(current_map_ == nullptr)
            {
                throw std::runtime_error("No current map available");
            }
            return current_map_->queryDistField(query_pts);
        }


        void writeMap()
        {
            if(current_map_ == nullptr)
            {
                throw std::runtime_error("No current map available");
            }
            writeCurrentSubmap();
        }


        void set2D(const bool is_2d)
        {
            is_2d_ = is_2d;
            if(current_map_)
            {
                current_map_->set2D(is_2d);
            }
        }            

    private:
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

        void writeCurrentSubmap()
        {
            if(current_map_ == nullptr)
            {
                throw std::runtime_error("No current map available");
            }
            std::string ply_path;
            if(using_submaps_)
            {
                ply_path = map_path_ + "submap_" + std::to_string(submap_counter_) + ".ply";
            }
            else
            {
                ply_path = map_path_ + "map.ply";
            }
            std::cout << "Writing map to: " << ply_path << std::endl;

            auto lambda = [] (std::shared_ptr<MapDistField> map, const std::string& path) {
                map->writeMap(path);
            };
            std::thread write_thread(lambda, current_map_, ply_path);
            write_thread.detach();

            // Write the trajectory
            std::string traj_path;
            if(using_submaps_)
            {
                traj_path = map_path_ + "trajectory_submap_" + std::to_string(submap_counter_) + ".csv";
            }
            else
            {
                traj_path = map_path_ + "trajectory_map.csv";
            }

            std::cout << "Writing trajectory to: " << traj_path << std::endl;            
            std::ofstream traj_file(traj_path);
            if(!traj_file)
            {
                throw std::runtime_error("Failed to open trajectory file");
            }
            // Write the header
            traj_file << "timestamp, x, y, z, r0, r1, r2" << std::endl;
            // Write the poses
            for(const auto& pose : current_map_poses_)
            {
                Mat3 rot_mat = pose.second.block<3,3>(0,0);
                Vec3 rot_vec = logMap(rot_mat);
                traj_file << std::fixed << pose.first << ", "
                          << pose.second(0,3) << ", "
                          << pose.second(1,3) << ", "
                          << pose.second(2,3) << ", "
                          << rot_vec(0) << ", "
                          << rot_vec(1) << ", "
                          << rot_vec(2)
                          << std::endl;
            }
            traj_file.close();
        }
};