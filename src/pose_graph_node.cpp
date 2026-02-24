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
const double kMaxDrift = 0.02;


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
        bool low_ram_mode_ = false;

        std::vector<std::string> map_paths_;
        std::vector<std::string> traj_files_;
        std::string scans_folder_ = "";
        std::string output_folder_ = "";

        std::vector<std::shared_ptr<MapDistField> > maps_;
        std::vector<std::vector<int64_t> > map_scans_times_;
        std::vector<std::pair<int64_t, int64_t> > map_time_ranges_;
        std::vector<Mat4> submap_original_poses_;
        std::vector<Mat4> submap_init_poses_;
        std::vector<Mat2_3> cam_mats_;

        MapDistFieldOptions map_options_;

        std::map<int64_t, std::shared_ptr<Vec7> > time_and_pose_;
        std::vector<std::shared_ptr<Vec7> > poses_in_order_;
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
            map_scans_times_.emplace_back();
            for(const auto& [timestamp, pose] : trajectory)
            {
                map_scans_times_.back().push_back(timestamp);
            }

            map_time_ranges_.emplace_back(trajectory.front().first, trajectory.back().first);
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
                    poses_in_order_.push_back(pose_ptr);
                    distance_travelled_.push_back(0.0);
                    pose_to_submap_id_.push_back(maps_.size() - 1);
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
                    poses_in_order_.push_back(pose);
                    double distance = (posQuatToTransform(*time_and_pose_[trajectory[i-1].first]).block<3,1>(0,3) - posQuatToTransform(*pose).block<3,1>(0,3)).norm();
                    distance_travelled_.push_back(distance_travelled_.back() + distance);
                    pose_to_submap_id_.push_back(maps_.size() - 1);
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
            std::cout << "Extracted " << keypoints.size() << " keypoints from the submap image." << std::endl;
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
                        distance = distance*kMaxDrift + kNeighborRadius;

                        Vec3 pos = poses_in_order_[j]->head<3>();
                        Vec3 current_pos = poses_in_order_.back()->head<3>();
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
                std::cout << "Attempting visual registration between " << pose_to_submap_id_.back() << " and " << submap_id << std::endl;
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
                    std::cout << "Not enough good matches (" << good_matches.size() << ") between submap " << pose_to_submap_id_.back() << " and " << submap_id << " for visual registration." << std::endl;
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
                std::cout << " ------------------------------------------------ " << std::endl;
                cv::Mat affine = cv::estimateAffinePartial2D(src_pts, dst_pts, inliers, cv::RANSAC, 2.0);
                std::cout << "/////////// Estimated affine transformation between submap " << pose_to_submap_id_.back() << " and " << submap_id << ":" << std::endl;

                if (affine.empty())
                {
                    std::cout << "Could not estimate a valid affine transformation between submap " << pose_to_submap_id_.back() << " and " << submap_id << "." << std::endl;
                    continue;
                }
                // If less than 3 inliers, skip this match
                int inlier_count = cv::countNonZero(inliers);
                if(inlier_count < 3)
                {
                    std::cout << "Not enough inliers (" << inlier_count << ") after RANSAC between submap " << pose_to_submap_id_.back() << " and " << submap_id << "." << std::endl;
                    continue;
                }


                // Convert the affine transformation to a 4x4 matrix
                auto [pose_3d, scale] = affineToPoseAndScale(affine, pose_to_submap_id_.back(), submap_id);
                if(std::isnan(scale) || std::isinf(scale))
                {
                    std::cout << "Estimated scale is invalid between submap " << pose_to_submap_id_.back() << " and " << submap_id << ". Skipping this match." << std::endl;
                    continue;
                }
                if(std::abs(scale - 1.0) > 0.1)
                {
                    std::cout << "Estimated scale is too different from 1.0 (scale: " << scale << ") between submap " << pose_to_submap_id_.back() << " and " << submap_id << ". Skipping this match." << std::endl;
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

            Mat3 image_trans = pose_2d;
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


};




int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PoseGraphNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}