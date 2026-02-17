#include "lice/submap_manager.h"
#include "lice/pointcloud_utils.h"

#include <ceres/manifold.h>
#include <ceres/rotation.h>




// Class for the cost function of the optimization problem for gravity and bias estimation
struct GravityBiasCostFunctor
{
    GravityBiasCostFunctor(const ugpm::PreintMeas& preint_meas, const Mat4& pose_A, const Mat4& pose_B, const double delta_t)
        : preint_meas_(preint_meas)
        , pose_A_(pose_A)
        , pose_B_(pose_B)
        , delta_t_(delta_t)
    {}
    
    template<typename T>
    bool operator()(const T* const gravity, const T* const bias_acc, const T* const bias_gyr, const T* const vel_A, const T* const vel_B, T* residuals) const
    {
        // Convert inputs to Eigen types
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> gravity_vec(gravity);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> bias_acc_vec(bias_acc);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> bias_gyr_vec(bias_gyr);

        Eigen::Map<Eigen::Matrix<T, 9, 1>> residuals_vec(residuals);

        Eigen::Matrix<T, 3, 1> p_A = pose_A_.block(0, 3, 3, 1).cast<T>();
        Eigen::Matrix<T, 3, 1> p_B = pose_B_.block(0, 3, 3, 1).cast<T>();
        Eigen::Matrix<T, 3, 3> R_A = pose_A_.block(0, 0, 3, 3).cast<T>();
        Eigen::Matrix<T, 3, 3> R_B = pose_B_.block(0, 0, 3, 3).cast<T>();

        Eigen::Matrix<T, 3, 1> v_A(vel_A[0], vel_A[1], vel_A[2]);
        Eigen::Matrix<T, 3, 1> v_B(vel_B[0], vel_B[1], vel_B[2]);

        Eigen::Matrix<T, 3, 1> delta_r_correction = preint_meas_.d_delta_R_d_bw.template cast<T>() * bias_gyr_vec;
        Eigen::Matrix<T, 3, 3> delta_R_correction;
        ceres::AngleAxisToRotationMatrix(delta_r_correction.data(), delta_R_correction.data());

        Eigen::Matrix<T, 3, 3> delta_R = preint_meas_.delta_R.template cast<T>() * delta_R_correction;

        Eigen::Matrix<T, 3, 1> delta_v = preint_meas_.delta_v.template cast<T>() + preint_meas_.d_delta_v_d_bf.template cast<T>() * bias_acc_vec + preint_meas_.d_delta_v_d_bw.template cast<T>() * bias_gyr_vec;

        Eigen::Matrix<T, 3, 1> delta_p = preint_meas_.delta_p.template cast<T>() + preint_meas_.d_delta_p_d_bf.template cast<T>() * bias_acc_vec + preint_meas_.d_delta_p_d_bw.template cast<T>() * bias_gyr_vec;


        Eigen::Matrix<T, 3, 3> rot_error = R_A.transpose() * R_B * delta_R.transpose();
        Eigen::Matrix<T, 3, 1> rot_error_vec;
        ceres::RotationMatrixToAngleAxis(rot_error.data(), rot_error_vec.data());

        residuals_vec.template block<3,1>(0,0) = rot_error_vec;
        residuals_vec.template block<3,1>(3,0) = R_A.transpose() * (p_B - p_A - v_A*delta_t_ - 0.5*gravity_vec*delta_t_*delta_t_) - delta_p;
        residuals_vec.template block<3,1>(6,0) = R_A.transpose() * (v_B - v_A - gravity_vec*delta_t_) - delta_v;

        return true;
    }

    private:
        const ugpm::PreintMeas& preint_meas_;
        const Mat4& pose_A_;
        const Mat4& pose_B_;
        const double delta_t_;
};



SubmapManager::SubmapManager(GpMapPublisher* publisher, const MapDistFieldOptions& options, const bool localization, const bool using_submaps, const double submap_length, const double submap_overlap, const std::string& map_path, const bool reverse_path)
    : publisher_(publisher)
    , options_(options)
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

    current_map_ = std::make_shared<MapDistField>(options_, publisher_);
    current_map_->set2D(is_2d_);
    current_map_->setGravity(gravity_);

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
SubmapManager::~SubmapManager() {}


// Use the current map to register the points
Mat4 SubmapManager::registerPts(const std::vector<Pointd>& pts, const Mat4& prior, const int64_t current_time, const bool approximate, const double loss_scale, const int max_iterations)
{
    GravityFactorFunctor* gravity_factor = nullptr;
    if(!localization_)
    {
        if((gravity_.squaredNorm() < 1))
        {
            attemptGravityBiasInit();
        }
        else
        {
            gravity_factor = computeGravityFactor(current_time);
        }
    }



    if(current_map_ == nullptr)
    {
        throw std::runtime_error("No current map available for registration");
    }

    Mat4 updated_pose = current_map_->registerPts(pts, prior, current_time, approximate, loss_scale, max_iterations, gravity_factor);
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
                current_map_ = std::make_shared<MapDistField>(options_, publisher_);
                current_map_->setGravity(gravity_);
                current_map_->loadMap(submap_paths_[new_map_id]);
                current_map_->set2D(is_2d_);
                current_map_id_ = new_map_id;
            }
        }
        current_node_id_ = best_node_id;
    }


    // Store the updated pose in the poses_imu_ map and update the path length and angle change
    if((gravity_.squaredNorm() < 1) && (first_imu_time_ns_ >= 0) && (current_time >= first_imu_time_ns_))
    {
        if(body_velocities_.find(current_time) != body_velocities_.end())
        {
            imu_poses_.push_back(updated_pose);
            imu_times_.push_back(current_time);
            imu_velocities_.push_back(updated_pose.block<3,3>(0,0) * body_velocities_[current_time]);
        }
    }
    if(path_length_ >= 0.0)
    {
        path_length_ += (updated_pose.block<3,1>(0,3) - last_registered_pose_.block<3,1>(0,3)).norm();
        Mat3 R_diff = last_registered_pose_.block<3,3>(0,0).transpose() * updated_pose.block<3,3>(0,0);
        double angle_diff = logMap(R_diff).norm();
        path_angle_change_ += angle_diff;
        std::cout << "Path length: " << path_length_ << ", Path angle change (deg): " << path_angle_change_ * 180.0 / M_PI << std::endl;
    }
    else
    {
        path_length_ = 0.0;
        path_angle_change_ = 0.0;
    }
    last_registered_pose_ = updated_pose;

    return updated_pose;
}


// Add points to the current map (and next map if using submaps)
void SubmapManager::addPts(const std::vector<Pointd>& pts, const Mat4& pose, const int64_t time)
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
            next_map_ = std::make_shared<MapDistField>(options_, publisher_);
            next_map_->set2D(is_2d_);
            next_map_->setGravity(gravity_);
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


void SubmapManager::addGyrMeasurement(const Vec3& gyr, const int64_t time_ns)
{
    ugpm::ImuSample imu_sample;
    imu_sample.data[0] = gyr[0];
    imu_sample.data[1] = gyr[1];
    imu_sample.data[2] = gyr[2];
    imu_sample.t = time_ns * 1e-9;

    imu_data_.gyr.push_back(imu_sample);

    if((imu_data_.acc.size() > 0) && (imu_data_.gyr.size() == 1))
    {
        first_imu_time_ns_ = time_ns;
    }
}

void SubmapManager::addAccMeasurement(const Vec3& acc, const int64_t time_ns)
{
    ugpm::ImuSample imu_sample;
    imu_sample.data[0] = acc[0];
    imu_sample.data[1] = acc[1];
    imu_sample.data[2] = acc[2];
    imu_sample.t = time_ns * 1e-9;

    imu_data_.acc.push_back(imu_sample);
    if((imu_data_.acc.size() == 1) && (imu_data_.gyr.size() > 0))
    {
        first_imu_time_ns_ = time_ns;
    }
}

void SubmapManager::addVelocity(const Vec3& vel, const int64_t time_ns)
{
    body_velocities_[time_ns] = vel;
}

void SubmapManager::attemptGravityBiasInit()
{
    if((imu_data_.acc.size() > 2)
        && (imu_data_.gyr.size() > 2)
        && (imu_times_.size() >= 2)
        && ((imu_times_.size() - 1) > preint_meas_vec_.size()))
    {
        int64_t last_imu_time_ns = std::min(imu_data_.acc.back().t, imu_data_.gyr.back().t) * 1e9;

        while((imu_times_.size() - 1) > preint_meas_vec_.size())
        {
            int64_t time_A = imu_times_[preint_meas_vec_.size()];
            int64_t time_B = imu_times_[preint_meas_vec_.size() + 1];
            ugpm::PreintOption opts;
            opts.type = ugpm::PreintType::LPM;
            opts.correlate = false;
            ugpm::ImuPreintegration preint(imu_data_, time_A * 1e-9, time_B * 1e-9, opts, ugpm::PreintPrior(), false, 1);
            preint_meas_vec_.push_back(preint.get());

            imu_data_ = imu_data_.get((time_A * 1e-9) - 0.1, std::numeric_limits<double>::max());
        }
    }


    if(path_length_ > 5.0 && path_angle_change_ > 120.0*M_PI/180.0 && (imu_poses_.size() >= 30))
    {
        std::cout << "Initializing gravity" << std::endl;
        int num_poses = preint_meas_vec_.size() + 1;

        // Create an optimization problem to optimize for the gravity vector and the biases
        ceres::Problem problem;
        gravity_ = -preint_meas_vec_[0].delta_v.normalized() * 9.81;
        ceres::SphereManifold<3>* manifold = new ceres::SphereManifold<3>();
        problem.AddParameterBlock(gravity_.data(), 3, manifold);
        problem.AddParameterBlock(bias_acc_.data(), 3);
        problem.AddParameterBlock(bias_gyr_.data(), 3);
        for(size_t i = 0; i < num_poses; i++)
        {
            problem.AddParameterBlock(imu_velocities_[i].data(), 3);
        }

        for(size_t i = 1; i < num_poses; i++)
        {
            double t_start = imu_times_[i-1] * 1e-9;
            double t_end = imu_times_[i] * 1e-9;

            ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<GravityBiasCostFunctor, 9, 3, 3, 3, 3, 3>(
                new GravityBiasCostFunctor(preint_meas_vec_[i-1], imu_poses_[i-1], imu_poses_[i], t_end - t_start));

            problem.AddResidualBlock(cost_function, nullptr, gravity_.data(), bias_acc_.data(), bias_gyr_.data(), imu_velocities_[i-1].data(), imu_velocities_[i].data());
        }

        ceres::Solver::Options options;
        options.minimizer_progress_to_stdout = true;
        options.max_num_iterations = 100;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.FullReport() << std::endl;

        std::cout << "\n\n\n\n\n\n\n\n\n\n\n\n\nOptimized gravity: " << gravity_.transpose() << std::endl;
        std::cout << "Optimized bias acc: " << bias_acc_.transpose() << std::endl;
        std::cout << "Optimized bias gyr: " << bias_gyr_.transpose() << std::endl;

        std::vector<double> gravity_residuals;
        for(size_t i = 1; i < num_poses; i++)
        {
            Vec3 corrected_delta_v = preint_meas_vec_[i-1].delta_v + preint_meas_vec_[i-1].d_delta_v_d_bf * bias_acc_ + preint_meas_vec_[i-1].d_delta_v_d_bw * bias_gyr_;
            Mat3 corrected_delta_R = preint_meas_vec_[i-1].delta_R * expMap(preint_meas_vec_[i-1].d_delta_R_d_bw * bias_gyr_);
            int64_t time_A = imu_times_[i-1];
            int64_t time_B = imu_times_[i];

            Vec3 vel_A = body_velocities_[time_A];
            Vec3 vel_B = body_velocities_[time_B];
            Vec3 local_g = (vel_B - corrected_delta_R.transpose()*(vel_A + corrected_delta_v)) / ((time_B - time_A) * 1e-9);
            Vec3 global_g = imu_poses_[i].block<3,3>(0,0) * local_g;

            double diff_angle = std::acos(global_g.dot(gravity_) / (global_g.norm() * gravity_.norm()));
            gravity_residuals.push_back(diff_angle);
            
            std::cout << "Local gravity diff: " << (local_g - gravity_).norm() << "  diff angle (deg): " << diff_angle * 180.0 / M_PI << std::endl;
        }
        double sq_sum = std::inner_product(gravity_residuals.begin(), gravity_residuals.end(), gravity_residuals.begin(), 0.0);
        gravity_angle_std_ = std::sqrt(sq_sum / gravity_residuals.size());
        std::cout << "Zero-mean Gravity residuals stdev (deg): " << gravity_angle_std_ * 180.0 / M_PI << std::endl;

        // Clean the data used to initialize the gravity and biases
        cleanBodyVelocities();
        imu_poses_.clear();
        imu_times_.clear();
        imu_velocities_.clear();
        preint_meas_vec_.clear();

        if(current_map_)
        {
            current_map_->setGravity(gravity_);
        }
        if(next_map_)
        {
            next_map_->setGravity(gravity_);
        }
    }

}

void SubmapManager::cleanBodyVelocities()
{
    // Delete the velocities that are too old (before last_registered_time_)
    std::vector<int64_t> times_to_delete;
    for(const auto& kv : body_velocities_)
    {
        if(kv.first < last_registered_time_)
        {
            times_to_delete.push_back(kv.first);
        }
    }
    for(const auto& time : times_to_delete)
    {
        body_velocities_.erase(time);
    }
}

// Get the current map points
std::vector<Pointd> SubmapManager::getPts()
{
    if(current_map_ == nullptr)
    {
        throw std::runtime_error("No current map available");
    }
    return current_map_->getPts();
}


// Query the distance field at the given points
std::vector<double> SubmapManager::queryDistField(const std::vector<Vec3>& query_pts)
{
    if(current_map_ == nullptr)
    {
        throw std::runtime_error("No current map available");
    }
    return current_map_->queryDistField(query_pts);
}


void SubmapManager::writeMap()
{
    if(current_map_ == nullptr)
    {
        throw std::runtime_error("No current map available");
    }
    writeCurrentSubmap();
}


void SubmapManager::set2D(const bool is_2d)
{
    is_2d_ = is_2d;
    if(current_map_)
    {
        current_map_->set2D(is_2d);
    }
}            


void SubmapManager::writeCurrentSubmap()
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

    auto lambda = [] (std::shared_ptr<MapDistField> map, const std::string& path) {
        map->writeMap(path);
    };
    std::thread write_thread(lambda, current_map_, ply_path);
    write_thread.detach();
}


GravityFactorFunctor* SubmapManager::computeGravityFactor(const int64_t current_time)
{
    int64_t last_imu_time_ns = std::min(imu_data_.acc.back().t, imu_data_.gyr.back().t) * 1e9;
    if(imu_data_.acc.size() < 2 || imu_data_.gyr.size() < 2 || current_time > last_imu_time_ns)
    {
        return nullptr;
    }

    int64_t time_A = last_registered_time_;
    int64_t time_B = current_time;

    if(body_velocities_.find(time_A) == body_velocities_.end() || body_velocities_.find(time_B) == body_velocities_.end())
    {
        return nullptr;
    }

    Vec3 body_vel_A = body_velocities_[time_A];
    Vec3 body_vel_B = body_velocities_[time_B];

    ugpm::PreintOption opts;
    opts.type = ugpm::PreintType::LPM;
    opts.correlate = false;
    ugpm::ImuPreintegration preint(imu_data_, time_A * 1e-9, time_B * 1e-9, opts, ugpm::PreintPrior(), false, 1);
    ugpm::PreintMeas preint_meas = preint.get();

    Vec3 corrected_delta_v = preint_meas.delta_v + preint_meas.d_delta_v_d_bf * bias_acc_ + preint_meas.d_delta_v_d_bw * bias_gyr_;
    Mat3 corrected_delta_R = preint_meas.delta_R * expMap(preint_meas.d_delta_R_d_bw * bias_gyr_);
    Vec3 local_g = (body_vel_B - corrected_delta_R.transpose()*(body_vel_A + corrected_delta_v)) / ((time_B - time_A) * 1e-9);
    
    GravityFactorFunctor* gravity_factor = new GravityFactorFunctor(local_g, gravity_, gravity_angle_std_);



    cleanBodyVelocities();
    imu_data_ = imu_data_.get((time_B * 1e-9) - 0.1, std::numeric_limits<double>::max());

    return gravity_factor;

}