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
#include <opencv2/features2d.hpp>

const float kCapHeight = 0.5; // Maximum height to consider for the submap images
const double kNeighborRadius = 60.0;


struct RelativePoseCostFunctor
{
    RelativePoseCostFunctor(const Mat4& measured_rel_pose, const Mat6& cov)
        : measured_rel_pose_(measured_rel_pose)
    {
        // LLt decomposition for weight matrix
        Eigen::LLT<Mat6> llt(cov.inverse());
        weight_ = llt.matrixL();
    }

    template <typename T>
    bool operator()(const T* const pose_i, const T* const pose_j, T* residuals) const
    {
        // Convert poses to transformation matrices
        Eigen::Matrix<T, 4, 4> Ti = posQuatToTransform(pose_i);
        Eigen::Matrix<T, 4, 4> Tj = posQuatToTransform(pose_j);
        Eigen::Matrix<T, 4, 4> Tij_measured = measured_rel_pose_.template cast<T>();
        // Compute the predicted relative transformation
        Eigen::Matrix<T, 4, 4> Tij_predicted = Ti.inverse() * Tj;
        // Compute the error transformation
        Eigen::Matrix<T, 4, 4> Te = Tij_measured.inverse() * Tij_predicted;
        // Convert the error transformation to a 6D vector (3 for rotation, 3 for translation)
        Eigen::Matrix<T, 6, 1> error = transformToPoseVector(Te);
        // Scale the residuals by the weight
        Eigen::Map<Eigen::Matrix<T, 6, 1>> residual_map(residuals);
        residual_map = weight_.template cast<T>() * error;
        return true;
    }

    Mat4 measured_rel_pose_;
    Mat6 weight_;
};

struct RelativePositionCostFunctor
{
    RelativePositionCostFunctor(const Vec3& measured_rel_pos, const Mat3& cov)
        : measured_rel_pos_(measured_rel_pos)
    {
        // LLt decomposition for weight matrix
        Eigen::LLT<Mat3> llt(cov.inverse());
        weight_ = llt.matrixL();
    }

    template <typename T>
    bool operator()(const T* const pose_i, const T* const pose_j, T* residuals) const
    {
        // Convert poses to transformation matrices
        Eigen::Matrix<T, 4, 4> Ti = posQuatToTransform(pose_i);
        Eigen::Matrix<T, 4, 4> Tj = posQuatToTransform(pose_j);
        // Compute the predicted relative transformation
        Eigen::Matrix<T, 4, 4> Tij_predicted = Ti.inverse() * Tj;
        // Compute the error transformation
        Eigen::Map<Eigen::Matrix<T, 3, 1>> residual_map(residuals);
        residual_map = weight_.template cast<T>() * (Tij_predicted.template block<3,1>(0,3) - measured_rel_pos_.template cast<T>());
        return true;
    }

    Vec3 measured_rel_pos_;
    Mat3 weight_;
};

struct RelativeRotationCostFunctor
{
    RelativeRotationCostFunctor(const Mat3& measured_rel_rot, const Mat3& cov)
        : measured_rel_rot_inv_(measured_rel_rot.inverse())
    {
        // LLt decomposition for weight matrix
        Eigen::LLT<Mat3> llt(cov.inverse());
        weight_ = llt.matrixL();
    }

    template <typename T>
    bool operator()(const T* const pose_i, const T* const pose_j, T* residuals) const
    {
        // Convert poses to transformation matrices
        Eigen::Matrix<T, 4, 4> Ti = posQuatToTransform(pose_i);
        Eigen::Matrix<T, 4, 4> Tj = posQuatToTransform(pose_j);
        // Compute the predicted relative transformation
        Eigen::Matrix<T, 4, 4> Tij_predicted = Ti.inverse() * Tj;
        // Compute the error transformation
        Eigen::Map<Eigen::Matrix<T, 3, 1>> residual_map(residuals);
        Eigen::Matrix<T, 3, 3> rot_diff = measured_rel_rot_inv_.template cast<T>() * Tij_predicted.template block<3,3>(0,0);
        residual_map = weight_.template cast<T>() * rotMatToAngleAxis(rot_diff);
        return true;
    }

    Mat3 measured_rel_rot_inv_;
    Mat3 weight_;
};




class PoseGraphNode: public rclcpp::Node
{
    public:
        PoseGraphNode()
            : Node("pose_graph_node")
        {
            RCLCPP_INFO(this->get_logger(), "Pose Graph Node has been started.");
            
            free_space_carving_radius_ = readFieldDouble(this, "free_space_carving_radius", 40.0);
            max_drift_ = readFieldDouble(this, "max_drift", 0.02);
            voxel_size_factor_ = readFieldDouble(this, "voxel_size_factor_for_registration", 2.0);
            max_nb_points_ = readFieldInt(this, "max_num_pts_for_registration", 8000);
            loss_scale_ = readFieldDouble(this, "loss_function_scale", 0.5);

            odom_error_pos_ = readFieldDouble(this, "odom_typical_pos_error", 0.01);
            odom_rotation_error_ = readFieldDouble(this, "odom_typical_rot_error_deg_per_m", 0.01) * M_PI / 180.0;


            loop_loss_scale_pos_ = readFieldDouble(this, "loop_closure_loss_scale_pos", 0.5);
            loop_loss_scale_rot_ = readFieldDouble(this, "loop_closure_loss_scale_rot", 0.5) * M_PI / 180.0;
            loop_pos_std = readFieldDouble(this, "loop_closure_std_pos", 0.50);
            loop_rot_std = readFieldDouble(this, "loop_closure_std_rot", 0.50) * M_PI / 180.0;


            
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
            map_options_.num_threads = 1;

            solver_options_.max_num_iterations = 100;
            solver_options_.num_threads = 1;
            solver_options_.minimizer_progress_to_stdout = true;

        }

    private:
        double free_space_carving_radius_ = -1.0;
        double image_res_ = 0.5;
        double map_res_ = 0.15;
        bool low_ram_mode_ = false;
        double max_drift_ = 0.02;
        double voxel_size_factor_ = 2.0;
        int max_nb_points_ = 8000;
        double loss_scale_ = 0.5;
        double odom_error_pos_ = 0.01;
        double odom_rotation_error_ = 0.01 * M_PI / 180.0;
        double loop_loss_scale_pos_ = 0.5;
        double loop_loss_scale_rot_ = 0.5 * M_PI / 180.0;
        double loop_pos_std = 0.50;
        double loop_rot_std = 0.50 * M_PI / 180.0;

        int64_t last_time_register_ = -1;

        ceres::Problem pose_graph_problem_;
        ceres::Solver::Options solver_options_;

        std::vector<std::string> map_paths_;
        std::vector<std::string> traj_files_;
        std::string scans_folder_ = "";
        std::string output_folder_ = "";

        std::vector<std::shared_ptr<MapDistField> > maps_;
        std::vector<std::vector<std::pair<int64_t, Mat4> > > map_scans_poses_;
        std::vector<Mat4> submap_original_poses_;
        std::vector<Mat4> submap_init_poses_;
        std::vector<Mat2_3> cam_mats_;

        MapDistFieldOptions map_options_;

        std::map<int64_t, std::shared_ptr<Vec7> > time_and_pose_;
        std::vector<int64_t> times_in_order_;
        std::vector<int> pose_to_submap_id_;
        std::vector<double> distance_travelled_;
        int last_closed_pose_index_ = -1;
        bool has_been_gravity_aligned_ = false;
        

        // Add the storage of the submap images
        std::vector<MatX> submap_images_;
        std::vector<std::vector<cv::KeyPoint> > submap_keypoints_;
        std::vector<cv::Mat> submap_descriptors_;


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

            submap_original_poses_.emplace_back(posQuatToTransform(trajectory.front().second));
            submap_init_poses_.emplace_back(posQuatToTransform(*(time_and_pose_[trajectory.front().first])));
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
            auto [laplace_image, height_image] = getSubmapImages(maps_.size() - 1);
            submap_images_.push_back(height_image); // Store the height image for later use
            std::cout << "Created submap image for map " << maps_.size() - 1 << std::endl;
            
            auto [keypoints, descriptors] = extractFeatures(laplace_image);
            submap_keypoints_.push_back(keypoints);
            submap_descriptors_.push_back(descriptors);
            
            std::set<int> submap_canditates = getSubmapIdsInRadius();

            std::cout << "Submaps in radius for the last pose: ";
            for(int submap_id : submap_canditates)
            {
                std::cout << submap_id << " ";
            }
            std::cout << std::endl;


            std::vector<std::pair<int, Mat4> > submap_id_and_coarse_poses = attemptVisualRegistration(submap_canditates);

            std::vector<std::tuple<int64_t, int64_t, Mat4> > pose_graph_edges = attemptFineRegistration(submap_id_and_coarse_poses);

            addLoopClosuresToPoseGraph(pose_graph_edges);

            writeFullTrajectory();
        }

        void addLoopClosuresToPoseGraph(const std::vector<std::tuple<int64_t, int64_t, Mat4> >& pose_graph_edges)
        {
            if(pose_graph_edges.size() > 0)
            {

std::cout << "Adding " << pose_graph_edges.size() << " loop closures to the pose graph optimization problem." << std::endl;
                initializeLoopClosureState(pose_graph_edges);
std::cout << "State initialized for loop closure optimization." << std::endl;

                for(const auto& [timestamp_i, timestamp_j, rel_pose] : pose_graph_edges)
                {
std::cout << "Adding loop closure edge between timestamps " << timestamp_i << " and " << timestamp_j << std::endl;
                    Mat3 cov = Mat3::Zero();
                    cov = Mat3::Identity() * loop_pos_std * loop_pos_std;
                    ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<RelativePositionCostFunctor, 3, 7, 7>(
                        new RelativePositionCostFunctor(rel_pose.block<3,1>(0,3), cov)
                    );
                    pose_graph_problem_.AddResidualBlock(cost_function, new ceres::CauchyLoss(loop_loss_scale_pos_), time_and_pose_[timestamp_i]->data(), time_and_pose_[timestamp_j]->data());

                    cov = Mat3::Identity() * loop_rot_std * loop_rot_std;
                    cost_function = new ceres::AutoDiffCostFunction<RelativeRotationCostFunctor, 3, 7, 7>(
                        new RelativeRotationCostFunctor(rel_pose.block<3,3>(0,0), cov)
                    );
                    pose_graph_problem_.AddResidualBlock(cost_function, new ceres::CauchyLoss(loop_loss_scale_rot_), time_and_pose_[timestamp_i]->data(), time_and_pose_[timestamp_j]->data());

                    last_time_register_ = std::max(last_time_register_, std::max(timestamp_i, timestamp_j));
                }

                ceres::Solver::Summary summary;
                ceres::Solve(solver_options_, &pose_graph_problem_, &summary);
                std::cout << summary.FullReport() << std::endl;
            }
        }


        void initializeLoopClosureState(const std::vector<std::tuple<int64_t, int64_t, Mat4> >& pose_graph_edges)
        {
            // Get the most recent pose in the state that is in the new edges
            int64_t most_recent_timestamp_j_in_edges = -1;
            int64_t most_recent_timestamp_i_in_edges = -1;
            Mat4 most_recent_rel_pose = Mat4::Identity();
            for(const auto& [timestamp_i, timestamp_j, rel_pose] : pose_graph_edges)
            {
                if(timestamp_j > most_recent_timestamp_j_in_edges)
                {
                    most_recent_timestamp_j_in_edges = timestamp_j;
                    most_recent_timestamp_i_in_edges = timestamp_i;
                    most_recent_rel_pose = rel_pose;
                }
            }
std::cout << "Most recent edge in new loop closures is between timestamps " << most_recent_timestamp_i_in_edges << " and " << most_recent_timestamp_j_in_edges << std::endl;

            // Get the delta pose between the current state and the most recent edge pose
            Mat4 current_delta_pose = posQuatToTransform(*time_and_pose_[most_recent_timestamp_i_in_edges]).inverse() * posQuatToTransform(*time_and_pose_[most_recent_timestamp_j_in_edges]);
            Mat4 delta_to_spread = current_delta_pose.inverse() * most_recent_rel_pose;
            Vec3 delta_rot_vec = logMap(delta_to_spread.block<3,3>(0,0));
            Vec3 delta_pos = delta_to_spread.block<3,1>(0,3);

            int64_t timestamp_to_spread = std::max(most_recent_timestamp_i_in_edges, last_time_register_);
            int64_t delta_time = most_recent_timestamp_j_in_edges - timestamp_to_spread;

std::cout << "Spreading loop closure correction to poses between timestamps " << timestamp_to_spread << " and " << most_recent_timestamp_j_in_edges << std::endl;

            // Loop the poses from the end
            for(int i = times_in_order_.size() - 1; (i >= 0) && (times_in_order_[i] >= timestamp_to_spread); i--)
            {
std::cout << "Updating pose at timestamp " << times_in_order_[i] << std::endl;
                if(times_in_order_[i] > most_recent_timestamp_j_in_edges)
                {
std::cout << "Pose is after the most recent timestamp in edges, applying full correction." << std::endl;
                    Mat4 last_j_to_i = posQuatToTransform(*time_and_pose_[most_recent_timestamp_j_in_edges]).inverse() * posQuatToTransform(*time_and_pose_[times_in_order_[i]]);
                    Mat4 new_pose = posQuatToTransform(*time_and_pose_[most_recent_timestamp_j_in_edges]) * delta_to_spread * last_j_to_i;
                    *time_and_pose_[times_in_order_[i]] = transformToPosQuat(new_pose);
                }
                else
                {
std::cout << "Pose is between the timestamp to spread and the most recent timestamp in edges, applying partial correction." << std::endl;
                    Mat4 delta_pose = Mat4::Identity();
                    double ratio = double(times_in_order_[i] - timestamp_to_spread) / double(delta_time);
                    delta_pose.block<3,3>(0,0) = expMap(delta_rot_vec * ratio);
                    delta_pose.block<3,1>(0,3) = delta_pos * ratio;
                    Mat4 new_pose = posQuatToTransform(*time_and_pose_[times_in_order_[i]]) * delta_pose;
                    *time_and_pose_[times_in_order_[i]] = transformToPosQuat(new_pose);
                }
            }
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
            map_scans_poses_.emplace_back();
            // Add 4 poses spread across the trajectory
            Mat4 first_pose_mat_inv = posQuatToTransform(trajectory.front().second).inverse();
            for(size_t i = 0; i < trajectory.size(); i++)
            {
                int64_t timestamp = trajectory[i].first;
                Vec7 pose = trajectory[i].second;
                map_scans_poses_.back().emplace_back(timestamp, first_pose_mat_inv * posQuatToTransform(pose));
            }

            for(size_t i = 0; i < trajectory.size(); i++)
            {
                int64_t timestamp = trajectory[i].first;
                Vec7 pose = trajectory[i].second;

                if(time_and_pose_.size() == 0)
                {
                    std::cout << "Adding first pose with timestamp " << timestamp << std::endl;
                    auto pose_ptr = std::make_shared<Vec7>(pose);
                    time_and_pose_[timestamp] = pose_ptr;
                    times_in_order_.push_back(timestamp);
                    distance_travelled_.push_back(0.0);
                    pose_to_submap_id_.push_back(maps_.size() - 1);
                    pose_graph_problem_.AddParameterBlock(pose_ptr->data(), 7);
                    pose_graph_problem_.SetParameterBlockConstant(pose_ptr->data());
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
                    times_in_order_.push_back(timestamp);
                    double distance = (posQuatToTransform(*time_and_pose_[trajectory[i-1].first]).block<3,1>(0,3) - posQuatToTransform(*pose).block<3,1>(0,3)).norm();
                    distance_travelled_.push_back(distance_travelled_.back() + distance);
                    pose_to_submap_id_.push_back(maps_.size() - 1);

                    Mat6 cov = Mat6::Zero();
                    cov.block<3,3>(0,0) = Mat3::Identity() * (distance+0.5) * odom_error_pos_;
                    cov.block<3,3>(3,3) = Mat3::Identity() * (distance+0.5) * odom_rotation_error_;
                    cov = cov*cov;
                    ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<RelativePoseCostFunctor, 6, 7, 7>(
                        new RelativePoseCostFunctor(delta_mat, cov)
                    );
                    pose_graph_problem_.AddResidualBlock(
                        cost_function,
                        nullptr,
                        time_and_pose_[trajectory[i-1].first]->data(),
                        pose->data());
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

        std::string getScanPath(int64_t timestamp)
        {
            if(scans_folder_ == "")
            {
                RCLCPP_ERROR(this->get_logger(), "Scans folder is not set. Cannot get scan path.");
                return "";
            }
            std::string scan_file = scans_folder_ + "/" + std::to_string(timestamp) + ".ply";
            if(!std::filesystem::exists(scan_file))
            {
                RCLCPP_WARN(this->get_logger(), "Scan file does not exist: %s", scan_file.c_str());
                return "";
            }
            return scan_file;
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
                std::string scan_file = getScanPath(timestamp);
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


        std::pair<cv::Mat, MatX> getSubmapImages(int map_id)
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
            Mat4 current_submap_pose = posQuatToTransform(*time_and_pose_[map_scans_poses_[map_id].front().first]);
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
            int rows = static_cast<int>(std::ceil((max_x - min_x) / image_res_))+1;
            int cols = static_cast<int>(std::ceil((max_y - min_y) / image_res_))+1;
            MatX counter = MatX::Zero(rows, cols);
            MatX sum_squared = MatX::Zero(rows, cols);
            MatX sum = MatX::Zero(rows, cols);
            cv::Mat laplace_image(rows, cols, CV_32F, cv::Scalar(0.0f));
            MatX height_image = MatX::Constant(rows, cols, std::numeric_limits<double>::quiet_NaN());

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
                if(x >= 0 && x < rows && y >= 0 && y < cols)
                {
                    counter(x, y) += 1.0f;
                    sum(x, y) += pt(2);
                    sum_squared(x, y) += pt(2) * pt(2);
                }
            }

            for(int x = 0; x < rows; ++x)
            {
                for(int y = 0; y < cols; ++y)
                {
                    if(counter(x, y) > 0)
                    {
                        height_image(x, y) = sum(x, y) / counter(x, y);
                    }
                }
            }

            for(int x = 1; x < rows-1; ++x)
            {
                for(int y = 1; y < cols-1; ++y)
                {
                    int local_count = 0;
                    double local_sum = 0;
                    if(counter(x, y) == 0)
                    {
                        continue;
                    }
                    if(counter(x-1, y) > 0)
                    {
                        local_count++;
                        local_sum += sum(x-1, y) / counter(x-1, y);
                    }
                    if(counter(x+1, y) > 0)
                    {
                        local_count++;
                        local_sum += sum(x+1, y) / counter(x+1, y);
                    }
                    if(counter(x, y-1) > 0)
                    {
                        local_count++;
                        local_sum += sum(x, y-1) / counter(x, y-1);
                    }
                    if(counter(x, y+1) > 0)
                    {
                        local_count++;
                        local_sum += sum(x, y+1) / counter(x, y+1);
                    }
                    laplace_image.at<float>(x, y) = static_cast<float>((local_sum - local_count*sum(x, y) / counter(x, y))/(4.0*kCapHeight));
                    if(laplace_image.at<float>(x, y) < -1.0f)
                    {
                        laplace_image.at<float>(x, y) = -1.0f;
                    }
                    else if(laplace_image.at<float>(x, y) > 1.0f)
                    {
                        laplace_image.at<float>(x, y) = 1.0f;
                    }
                }
            }
            // Apply a simple gaussian blur to smooth the image
            cv::GaussianBlur(laplace_image, laplace_image, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
            // Normalize to [0, 255] for visualization
            cv::normalize(laplace_image, laplace_image, 0, 255, cv::NORM_MINMAX);
            // Convert to 8-bit image
            laplace_image.convertTo(laplace_image, CV_8U);

            return {laplace_image, height_image};
        }

        std::pair<std::vector<cv::KeyPoint>, cv::Mat> extractFeatures(const cv::Mat& image)
        {
            // Use SIFT to extract features from the image
            cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
            std::vector<cv::KeyPoint> keypoints;
            cv::Mat descriptors;
            sift->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
            RCLCPP_INFO(this->get_logger(), "Extracted %zu keypoints from the submap image.", keypoints.size());
            return {keypoints, descriptors};
        }

        std::set<int> getSubmapIdsInRadius()
        {
            std::set<int> submap_ids;
            int current_submap_id = pose_to_submap_id_.back();
            for(int i = pose_to_submap_id_.size() - 1; (i >= 0) && (pose_to_submap_id_[i] == current_submap_id); i--)
            {
                for(int j = i-1; j > 0; j--)
                {
                    if(pose_to_submap_id_[j] < current_submap_id - 1)
                    {
                        double distance = distance_travelled_[i] - distance_travelled_[j];
                        distance = distance*max_drift_ + kNeighborRadius;

                        Vec7 pose_i = *time_and_pose_[times_in_order_[i]];
                        Vec3 pos = pose_i.head<3>();
                        Vec7 pose_j = *time_and_pose_[times_in_order_[j]];
                        Vec3 current_pos = pose_j.head<3>();
                        double euclidean_distance = (current_pos - pos).norm();
                        if(euclidean_distance < distance)
                        {
                            submap_ids.insert(pose_to_submap_id_[j]);
                        }
                    }
                }
            }
            return submap_ids;
        }


        std::vector<std::pair<int, Mat4> > attemptVisualRegistration(const std::set<int>& submap_candidates)
        {
            std::vector<std::pair<int, Mat4> > submap_id_and_coarse_poses;
            cv::BFMatcher matcher(cv::NORM_L2, false);
            for(int submap_id : submap_candidates)
            {
                RCLCPP_INFO(this->get_logger(), "Attempting visual registration between submap %d and %d.", pose_to_submap_id_.back(), submap_id);
                std::vector<cv::KeyPoint> keypoints1 = submap_keypoints_[pose_to_submap_id_.back()];
                cv::Mat descriptors1 = submap_descriptors_[pose_to_submap_id_.back()];
                std::vector<cv::KeyPoint> keypoints2 = submap_keypoints_[submap_id];
                cv::Mat descriptors2 = submap_descriptors_[submap_id];

                std::vector<std::vector<cv::DMatch> > knn_matches;
                matcher.knnMatch(descriptors1, descriptors2, knn_matches, 2);

                std::vector<cv::DMatch> good_matches;
                for(size_t i = 0; i < knn_matches.size(); i++)
                {
                    auto kp1 = keypoints1[knn_matches[i][0].queryIdx];
                    auto kp2 = keypoints2[knn_matches[i][0].trainIdx];
                    double scale_ratio = kp1.size / kp2.size;
                    if( (std::abs(scale_ratio - 1.0) < 0.05) && (knn_matches[i][0].distance < 0.75 * knn_matches[i][1].distance) )
                    {
                        good_matches.push_back(knn_matches[i][0]);
                    }

                }

                if (good_matches.size() < 4)
                {
                    RCLCPP_WARN(this->get_logger(), "Not enough good matches (%zu) between submap %d and %d for visual registration. Skipping this match.", good_matches.size(), pose_to_submap_id_.back(), submap_id);
                    continue;
                }

                // Extract the matched keypoints
                std::vector<cv::Point2f> dst_pts, src_pts;
                for (const auto& match : good_matches)
                {
                    src_pts.push_back(cv::Point2f(keypoints1[match.queryIdx].pt.y, keypoints1[match.queryIdx].pt.x)); // Note the swap of x and y
                    dst_pts.push_back(cv::Point2f(keypoints2[match.trainIdx].pt.y, keypoints2[match.trainIdx].pt.x)); // Note the swap of x and y
                }

                // Estimate the Affine partial 2D
                cv::Mat inliers;
                cv::Mat affine = cv::estimateAffinePartial2D(src_pts, dst_pts, inliers, cv::RANSAC, 2.0);

                if (affine.empty())
                {
                    RCLCPP_WARN(this->get_logger(), "Could not estimate a valid affine transformation between submap %d and %d.", pose_to_submap_id_.back(), submap_id);
                    continue;
                }
                // If less than 3 inliers, skip this match
                int inlier_count = cv::countNonZero(inliers);
                if(inlier_count < 3)
                {
                    RCLCPP_WARN(this->get_logger(), "Not enough inliers (%d) after RANSAC between submap %d and %d. Skipping this match.", inlier_count, pose_to_submap_id_.back(), submap_id);
                    continue;
                }


                // Convert the affine transformation to a 4x4 matrix
                auto [pose_3d, scale] = affineToPoseAndScale(affine, pose_to_submap_id_.back(), submap_id);
                if(std::isnan(scale) || std::isinf(scale))
                {
                    RCLCPP_WARN(this->get_logger(), "Estimated scale is invalid (scale: %f) between submap %d and %d. Skipping this match.", scale, pose_to_submap_id_.back(), submap_id);
                    continue;
                }
                if(std::abs(scale - 1.0) > 0.1)
                {
                    RCLCPP_WARN(this->get_logger(), "Estimated scale is too different from 1.0 (scale: %f) between submap %d and %d. Skipping this match.", scale, pose_to_submap_id_.back(), submap_id);
                    continue;
                }
                submap_id_and_coarse_poses.emplace_back(submap_id, pose_3d);

            }
            return submap_id_and_coarse_poses;
        }


        std::pair<Mat4, double> affineToPoseAndScale(const cv::Mat& affine, const int id_source, const int id_target)
        {
            double scale = std::sqrt(affine.at<double>(0,0)*affine.at<double>(0,0) + affine.at<double>(0,1)*affine.at<double>(0,1));
            double angle = std::atan2(affine.at<double>(1,0), affine.at<double>(0,0));
            Mat3 pose_2d = Mat3::Identity();
            pose_2d(0,0) = std::cos(angle);
            pose_2d(0,1) = -std::sin(angle);
            pose_2d(1,0) = std::sin(angle);
            pose_2d(1,1) = std::cos(angle);
            pose_2d(0,2) = affine.at<double>(0,2);
            pose_2d(1,2) = affine.at<double>(1,2);


            Mat3 cam_mat_source_3 = Mat3::Identity();
            cam_mat_source_3.block<2,3>(0,0) = cam_mats_[id_source];
            Mat3 cam_mat_target_3 = Mat3::Identity();
            cam_mat_target_3.block<2,3>(0,0) = cam_mats_[id_target];

            pose_2d = cam_mat_target_3.inverse() * pose_2d * cam_mat_source_3;

            Mat4 pose_3d = Mat4::Identity();
            pose_3d.block<2,2>(0,0) = pose_2d.block<2,2>(0,0);
            pose_3d(0,3) = pose_2d(0,2);
            pose_3d(1,3) = pose_2d(1,2);

                
            if ( std::abs(scale - 1.0) > 0.05)
            {
                return {pose_3d, scale};
            }

            auto pts_source = maps_[id_source]->getPts();
            auto pts_target = maps_[id_target]->getPts();
            ankerl::unordered_dense::map<std::pair<int, int>, std::pair<int, double>> target_cells;
            Mat4 target_transform = submap_init_poses_[id_target] * submap_original_poses_[id_target].inverse();
            for(const auto& pt : pts_target)
            {
                Vec3 pt_W = target_transform.block<3,3>(0,0) * pt.vec3() + target_transform.block<3,1>(0,3);
                int cell_x = static_cast<int>(std::floor(pt_W(0) / image_res_));
                int cell_y = static_cast<int>(std::floor(pt_W(1) / image_res_));
                std::pair<int, int> cell_idx = {cell_x, cell_y};
                double height = pt_W(2);
                if(target_cells.find(cell_idx) == target_cells.end())
                {
                    target_cells[cell_idx] = {1, height};
                }
                else
                {
                    target_cells[cell_idx].first += 1;
                    target_cells[cell_idx].second += height;
                }
            }
            ankerl::unordered_dense::map<std::pair<int, int>, std::pair<int, double>> source_cells;
            Mat4 source_transform = pose_3d * submap_init_poses_[id_source] * submap_original_poses_[id_source].inverse();
            for(const auto& pt : pts_source)
            {
                Vec3 pt_W = source_transform.block<3,3>(0,0) * pt.vec3() + source_transform.block<3,1>(0,3);
                int cell_x = static_cast<int>(std::floor(pt_W(0) / image_res_));
                int cell_y = static_cast<int>(std::floor(pt_W(1) / image_res_));
                std::pair<int, int> cell_idx = {cell_x, cell_y};
                double height = pt_W(2);
                if(source_cells.find(cell_idx) == source_cells.end())
                {
                    source_cells[cell_idx] = {1, height};
                }
                else
                {
                    source_cells[cell_idx].first += 1;
                    source_cells[cell_idx].second += height;
                }
            }

            std::vector<double> height_differences;
            for(const auto& [cell_idx, target_cell_data] : target_cells)
            {
                if(source_cells.find(cell_idx) != source_cells.end())
                {
                    double target_cell_height = target_cell_data.second / target_cell_data.first;
                    double source_cell_height = source_cells[cell_idx].second / source_cells[cell_idx].first;
                    height_differences.push_back(target_cell_height - source_cell_height);
                }
            }
            double median_height = 0.0;
            if(height_differences.size() > 2)            {
                std::sort(height_differences.begin(), height_differences.end());
                median_height = height_differences[height_differences.size()/2];
            }
            pose_3d(2,3) = median_height;

        
            pose_3d = submap_original_poses_[id_target] * submap_init_poses_[id_target].inverse() * pose_3d * submap_init_poses_[id_source] * submap_original_poses_[id_source].inverse(); 

            return {pose_3d, scale};
        }



        std::vector<std::tuple<int64_t, int64_t, Mat4> > attemptFineRegistration(const std::vector<std::pair<int, Mat4> >& submap_id_and_coarse_poses)
        {
            std::vector<std::tuple<int64_t, int64_t, Mat4> > pose_graph_edges;
            for(const auto& [submap_id, coarse_pose] : submap_id_and_coarse_poses)
            {
                Mat4 target_submap_pose = posQuatToTransform(*time_and_pose_[map_scans_poses_[submap_id].front().first]);
                Mat4 coarse_pose_inv = coarse_pose.inverse();
                // Get 4 poses / scans from the target submap (evenly spread throughout the submap time range)
                std::vector<std::pair<int64_t, Mat4> > target_poses;
                target_poses.push_back(map_scans_poses_[submap_id].front());
                target_poses.push_back(map_scans_poses_[submap_id][map_scans_poses_[submap_id].size()/3]);
                target_poses.push_back(map_scans_poses_[submap_id][2*map_scans_poses_[submap_id].size()/3]);
                target_poses.push_back(map_scans_poses_[submap_id].back());

                for(auto [target_timestamp, relative_pose] : target_poses)
                {
                    std::vector<Pointd> scan = loadPointCloudFromPly(getScanPath(target_timestamp));
                    if(scan.size() == 0)
                    {
                        RCLCPP_WARN(this->get_logger(), "Scan file is empty for timestamp: %ld. Skipping fine registration for this timestamp.", target_timestamp);
                        continue;
                    }
                    Mat4 target_scan_pose = posQuatToTransform(*time_and_pose_[target_timestamp]);
                    Mat4 initial_guess = coarse_pose_inv * submap_original_poses_[submap_id] * target_submap_pose.inverse() * target_scan_pose;


                    std::vector<Pointd> downsampled_scan = downsamplePointCloud(scan, voxel_size_factor_*map_res_, max_nb_points_,true);
                    if(downsampled_scan.size() == 0)                    {
                        RCLCPP_WARN(this->get_logger(), "Downsampled scan is empty for timestamp: %ld. Skipping fine registration for this timestamp.", target_timestamp);
                        continue;
                    }

                    Mat4 T_o_target = maps_.back()->registerPts(downsampled_scan, initial_guess, 1, false, 5.0, 10);
                    T_o_target = maps_.back()->registerPts(downsampled_scan, T_o_target, 1, false, 2.0, 10);
                    T_o_target = maps_.back()->registerPts(downsampled_scan, T_o_target, 1, false, loss_scale_);

                    // Align the scan to test the number of inliers
                    std::vector<Pointd> transformed_scan;
                    for(const auto& pt : scan)
                    {
                        Vec3 transformed_vec = T_o_target.block<3,3>(0,0) * pt.vec3() + T_o_target.block<3,1>(0,3);
                        transformed_scan.emplace_back(Vec3(transformed_vec(0), transformed_vec(1), transformed_vec(2)), 0);
                    }
                    int inlier_count = 0;
                    for(const auto& pt : transformed_scan)
                    {
                        double distance = maps_.back()->queryDistField(pt.vec3());
                        if(distance < map_res_)
                        {
                            inlier_count++;
                        }
                    }
                    double inlier_ratio = static_cast<double>(inlier_count) / static_cast<double>(scan.size());

                    if(inlier_ratio < 0.8)
                    {
                        RCLCPP_WARN(this->get_logger(), "Low inlier ratio (%f) after fine registration between submap %d and scan at timestamp %ld. Skipping this match.", inlier_ratio, submap_id, target_timestamp);
                        continue;
                    }

                    // Get the closest pose/timestamp in the current submap to the target pose
                    int64_t closest_timestamp = -1;
                    double closest_distance = std::numeric_limits<double>::max();
                    Mat4 T_o_source = Mat4::Identity();
                    for(const auto& [timestamp, rel_pose] : map_scans_poses_.back())
                    {
                        Mat4 current_pose = submap_original_poses_.back() * rel_pose;
                        double distance = (current_pose.block<3,1>(0,3) - T_o_target.block<3,1>(0,3)).norm();
                        if(distance < closest_distance)
                        {
                            closest_distance = distance;
                            closest_timestamp = timestamp;
                            T_o_source = current_pose;
                        }
                    }
                    if(closest_timestamp == -1)
                    {
                        RCLCPP_WARN(this->get_logger(), "Could not find a closest timestamp in the current submap for scan at timestamp %ld. Skipping this match.", target_timestamp);
                        continue;
                    }

                    Mat4 T_target_source = T_o_target.inverse() * T_o_source;
                    pose_graph_edges.emplace_back(target_timestamp, closest_timestamp, T_target_source);
                    

                    // Read both the target and closest scans, transform the closest scan with the estimated transformation and save both for visualization
                    std::vector<Pointd> source_scan = loadPointCloudFromPly(getScanPath(closest_timestamp));
                    if(source_scan.size() == 0)                    {
                        RCLCPP_WARN(this->get_logger(), "Scan file is empty for closest timestamp: %ld. Skipping saving transformed scan for visualization.", closest_timestamp);
                        continue;
                    }
                    std::vector<Pointd> transformed_closest_scan;
                    for(const auto& pt : source_scan)                    {
                        Vec3 transformed_vec = T_target_source.block<3,3>(0,0) * pt.vec3() + T_target_source.block<3,1>(0,3);
                        transformed_closest_scan.emplace_back(Vec3(transformed_vec(0), transformed_vec(1), transformed_vec(2)), 0);
                    }
                    std::vector<Pointd> target_scan = loadPointCloudFromPly(getScanPath(target_timestamp));
                    if(target_scan.size() == 0)                    {
                        RCLCPP_WARN(this->get_logger(), "Scan file is empty for target timestamp: %ld. Skipping saving target scan for visualization.", target_timestamp);
                        continue;
                    }
                    // Save the target scan, and the transformed closest scan for visualization
                    std::string target_scan_file = output_folder_ + "/loop_from_submap_" + std::to_string(submap_id) + "_timestamp_" + std::to_string(target_timestamp) + "_inlier_ratio_" + std::to_string(inlier_ratio) + "_target.ply";
                    savePointCloudToPly(target_scan_file, target_scan);
                    RCLCPP_INFO(this->get_logger(), "Saved target scan to %s", target_scan_file.c_str());
                    std::string transformed_closest_scan_file = output_folder_ + "/loop_from_submap_" + std::to_string(submap_id) + "_timestamp_" + std::to_string(target_timestamp) + "_inlier_ratio_" + std::to_string(inlier_ratio) + "_closest_transformed.ply";
                    savePointCloudToPly(transformed_closest_scan_file, transformed_closest_scan);
                    RCLCPP_INFO(this->get_logger(), "Saved transformed closest scan to %s", transformed_closest_scan_file.c_str());


                }
            }
            return pose_graph_edges;
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