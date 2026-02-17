#include <ceres/ceres.h>
#include <ceres/manifold.h>
#include "lice/submap_bundle_adjustment.h"
#include "lice/pointcloud_utils.h"
#include <fstream>






SubmapBundleAdjustment::SubmapBundleAdjustment(const std::string& data_folder, const SubmapBundleAdjustmentOptions& options)
    : options_(options)
{
    // Load the submap poses and loop closures
    std::string data_path = data_folder;
    if(data_path.back() != '/')
    {
        data_path += "/";
    }
    loop_folder_ = data_path;
    pcd_folder_ = data_path + "pcds/";
    loadSubmapPoses(data_path + "poses.txt");
    loadLoopClosures(data_path + "matches_and_transforms.txt");

}




void SubmapBundleAdjustment::poseGraphOptimization()
{
    ceres::Problem problem;

    // Add the poses with manifolds to the problem
    for(size_t i = 0; i < submap_poses_.size(); ++i)
    {
        ceres::ProductManifold<ceres::EuclideanManifold<3>, ceres::QuaternionManifold>* se3 =
            new ceres::ProductManifold<ceres::EuclideanManifold<3>, ceres::QuaternionManifold>();
        problem.AddParameterBlock(submap_poses_[i].data(), 7, se3);
    }


    // Add odometry constraints
    for(size_t i = 0; i < odom_transforms_.size(); ++i)
    {
        Mat4 measured_rel_pose = odom_transforms_[i];
        Mat6 cov = Mat6::Zero();
        cov.block<3,3>(0,0) = Mat3::Identity() * options_.odom_pos_std * options_.odom_pos_std;
        cov.block<3,3>(3,3) = Mat3::Identity() * options_.odom_rot_std * options_.odom_rot_std;

        ceres::CostFunction* cost_function =
            new ceres::AutoDiffCostFunction<RelativePoseCostFunctor, 6, 7, 7>(
                new RelativePoseCostFunctor(measured_rel_pose, cov));

        problem.AddResidualBlock(
            cost_function,
            nullptr,
            submap_poses_[i].data(),
            submap_poses_[i+1].data());
    }

    // Add loop closure constraints
    ceres::LossFunction* loss_function_pos = new ceres::CauchyLoss(options_.loop_loss_scale_pos);
    ceres::LossFunction* loss_function_rot = new ceres::CauchyLoss(options_.loop_loss_scale_rot);
    for(size_t i = 0; i < loop_closures_.size(); ++i)
    {
        int id_1 = loop_closures_[i].first;
        int id_2 = loop_closures_[i].second;
        Mat4 measured_rel_pose = loop_transforms_[i];
        Mat3 pos_cov = Mat3::Identity() * options_.loop_pos_std * options_.loop_pos_std;
        Mat3 rot_cov = Mat3::Identity() * options_.loop_rot_std * options_.loop_rot_std;
        
        // Position residual
        ceres::CostFunction* cost_function_pos =
            new ceres::AutoDiffCostFunction<RelativePositionCostFunctor, 3, 7, 7>(
                new RelativePositionCostFunctor(measured_rel_pose.block<3,1>(0,3), pos_cov));

        problem.AddResidualBlock(
            cost_function_pos,
            loss_function_pos,
            submap_poses_[id_1].data(),
            submap_poses_[id_2].data());

        // Rotation residual
        ceres::CostFunction* cost_function_rot =
            new ceres::AutoDiffCostFunction<RelativeRotationCostFunctor, 3, 7, 7>(
                new RelativeRotationCostFunctor(measured_rel_pose.block<3,3>(0,0), rot_cov));

        problem.AddResidualBlock(
            cost_function_rot,
            loss_function_rot,
            submap_poses_[id_1].data(),
            submap_poses_[id_2].data());
    }

    // Fix the first pose to anchor the map
    problem.SetParameterBlockConstant(submap_poses_[0].data());


    // Solve the problem
    ceres::Solver::Options solver_options;
    solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    solver_options.max_num_iterations = 100;
    solver_options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(solver_options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;

    // Write the optimized poses
    writeOptimizedPoses(loop_folder_ + "pose_graph_poses.txt");

}







void SubmapBundleAdjustment::loadSubmapPoses(const std::string& filename)
{
    submap_poses_.clear();
    odom_transforms_.clear();

    std::ifstream file(filename);
    if(!file.is_open())
    {
        std::cerr << "SubmapBundleAdjustment::loadSubmapPoses: Unable to open file: " << filename << std::endl;
        return;
    }

    std::string line;
    Mat4 prev_pose = Mat4::Identity();
    while(std::getline(file, line))
    {
        std::istringstream iss(line);
        Vec6 pose;
        for(int i = 0; i < 6; ++i)
        {
            iss >> pose[i];
        }
        Mat4 trans = posRotToTransform(pose);
        submap_poses_.push_back(transformToPosQuat<double>(trans));


        if(submap_poses_.size() > 1)
        {
            Mat4 odom  = prev_pose.inverse() * trans;
            odom_transforms_.push_back(odom);
        }
        prev_pose = trans;
    }

    file.close();
}


void SubmapBundleAdjustment::loadLoopClosures(const std::string& filename)
{
    loop_closures_.clear();
    loop_transforms_.clear();

    std::ifstream file(filename);
    if(!file.is_open())
    {
        std::cerr << "SubmapBundleAdjustment::loadLoopClosures: Unable to open file: " << filename << std::endl;
        return;
    }

    std::string line;
    while(std::getline(file, line))
    {
        std::istringstream iss(line);
        int id_1, id_2;
        iss >> id_1 >> id_2;
        loop_closures_.emplace_back(id_1, id_2);
        Vec6 loop_closure;
        for(int i = 0; i < 6; ++i)
        {
            iss >> loop_closure[i];
        }
        loop_transforms_.push_back(posRotToTransform(loop_closure));
    }

    file.close();
}



void SubmapBundleAdjustment::writeOptimizedPoses(const std::string& filename) const
{
    std::ofstream file(filename);
    if(!file.is_open())
    {
        std::cerr << "SubmapBundleAdjustment::writeOptimizedPoses: Unable to open file: " << filename << std::endl;
        return;
    }

    for(const auto& pose : submap_poses_)
    {
        Vec6 pose6 = posQuatToPosRot<double>(pose);
        for(int i = 0; i < 6; ++i)
        {
            file << pose6[i] << " ";
        }
        file << std::endl;
    }

    file.close();
}




void SubmapBundleAdjustment::refineLoopTransforms()
{
    MapDistFieldOptions map_options;
    for(size_t i = 0; i < loop_closures_.size(); ++i)
    {
        std::cout << "Refining loop transform " << i << std::endl;
        int id_1 = loop_closures_[i].first;
        int id_2 = loop_closures_[i].second;
        std::string pcd_1 = pcd_folder_ + "submap_" + std::to_string(id_1) + ".ply";
        std::string pcd_2 = pcd_folder_ + "submap_" + std::to_string(id_2) + ".ply";

        std::string res_1 = pcd_folder_ + "submap_" + std::to_string(id_1) + ".info";

        map_options.cell_size = readResolutions(res_1);


        MapDistField map_field(map_options, nullptr);
        map_field.loadMap(pcd_1);

        

        std::vector<Pointd> pts_to_register = loadPointCloudFromPly(pcd_2);

        std::vector<Pointd> downsampled_pts = downsamplePointCloud<double>(pts_to_register, 5*map_options.cell_size, 10000);

        Mat4 initial_guess = loop_transforms_[i];

        Mat4 refined_transform = map_field.registerPts(downsampled_pts, initial_guess, 0.0, false, 3*map_options.cell_size, 25);

        downsampled_pts = downsamplePointCloud<double>(pts_to_register, 2*map_options.cell_size, 100000);
        refined_transform = map_field.registerPts(downsampled_pts, refined_transform, 0.0, false, 1*map_options.cell_size, 50);

        loop_transforms_[i] = refined_transform;
    }
}


double SubmapBundleAdjustment::readResolutions(const std::string& filename)
{
    std::ifstream file(filename);
    if(!file.is_open())
    {
        std::cerr << "SubmapBundleAdjustment::readResolutions: Unable to open file: " << filename << std::endl;
        return -1.0;
    }

    // Find the line that starts with "cell_size"
    std::string line;
    double resolution = 0.3; // Default value
    while(std::getline(file, line))
    {
        if(line.find("cell_size") != std::string::npos)
        {
            std::istringstream iss(line);
            std::string key;
            iss >> key >> resolution;
            break;
        }
    }
    file.close();
    return resolution;
}