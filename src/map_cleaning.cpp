#include "cxxopts/include/cxxopts.hpp"
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <vector>
#include <cstdint>
#include "lice/map_distance_field.h"
#include "lice/math_utils.h"
#include "lice/pointcloud_utils.h"

int main(int argc, char** argv)
{
    // Define command-line options
    cxxopts::Options options("Map cleaning tool", "A tool to clean maps by removing dynamic objects");
    options.add_options()
        ("d,data_folder", "Path to the data folder with map.ply or ", cxxopts::value<std::string>())
        ("v,voxel_size", "Voxel size for the distance field", cxxopts::value<double>())
        ("r,radius", "Radius around the pose for free-space carving", cxxopts::value<double>()->default_value("20.0"))
        ("s,using_submaps", "Use submaps for cleaning")
        ("h,help", "Print help");

    auto result = options.parse(argc, argv);

    // Handle help option
    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }
    // Validate required arguments
    if (!result.count("data_folder") || !result.count("voxel_size")) {
        std::cerr << "Error: Missing required arguments." << std::endl;
        std::cout << options.help() << std::endl;
        return 1;
    }
    // Retrieve argument values
    std::string data_folder = result["data_folder"].as<std::string>();
    double carving_radius = result["radius"].as<double>();
    bool using_submaps = result.count("using_submaps");

    // Create the paths
    if(data_folder.back() != '/')
    {
        data_folder += "/";
    }
    std::string scan_folder = data_folder + "scans/";

    // Collect the map paths
    std::vector<std::string> map_paths;
    if (using_submaps) {
        // Load submap paths
        int map_ptr = 0;
        while (true) {
            std::string ply_path = data_folder + "submap_" + std::to_string(map_ptr) + ".ply";
            if (std::filesystem::exists(ply_path)) {
                map_paths.push_back(ply_path);
                map_ptr++;
            } else {
                break;
            }
        }
    } else {
        map_paths.push_back(data_folder + "map.ply");
    }

    // Read the scan folder
    std::cout << "Loading scans from: " << scan_folder << std::endl;
    if (!std::filesystem::exists(scan_folder)) {
        std::cerr << "Error: Scan folder does not exist." << std::endl;
        return 1;
    }
    std::vector<std::string> scan_files;
    std::vector<int64_t> scan_times;
    for (const auto& entry : std::filesystem::directory_iterator(scan_folder))
    {
        if (entry.path().extension() == ".ply") {
            scan_files.push_back(entry.path().string());
            // Convert the file name (without extension) to timestamp
            std::string filename = entry.path().stem().string();
            int64_t timestamp = std::stoll(filename);
            scan_times.push_back(timestamp);
        }
    }

    // Sort scans by timestamp
    std::vector<std::pair<int64_t, std::string>> scans;
    for (size_t i = 0; i < scan_files.size(); ++i) {
        scans.emplace_back(scan_times[i], scan_files[i]);
    }
    std::sort(scans.begin(), scans.end());


    // Loop through each map and clean
    for (size_t i = 0; i < map_paths.size(); ++i) {
        std::string map_path = map_paths[i];
        std::cout << "Cleaning map: " << map_path << std::endl;
        MapDistFieldOptions options;
        options.cell_size = result["voxel_size"].as<double>();
        options.free_space_carving = true;
        options.free_space_carving_radius = carving_radius;
        options.last_scan_carving = false;
        options.min_range = 0.0;
        options.max_range = std::numeric_limits<double>::max();

        // Copy the original map replacing .ply with _original.ply
        std::string original_map_path = map_path;
        original_map_path.replace(original_map_path.end() - 4, original_map_path.end(), "_original.ply");
        std::filesystem::copy_file(map_path, original_map_path, std::filesystem::copy_options::overwrite_existing);
        std::cout << "Copied original map to: " << original_map_path << std::endl;


        MapDistField map(options, nullptr);
        map.loadMap(map_path);


        // Load the corresponding trajectory
        std::string traj_file;
        if(using_submaps)
        {
            traj_file = data_folder + "trajectory_submap_" + std::to_string(i) + ".csv";
        }
        else
        {
            traj_file = data_folder + "trajectory_map.csv";
        }
        std::cout << "Loading trajectory from: " << traj_file << std::endl;
        std::vector<int64_t> traj_times;
        std::vector<Mat4> traj_poses;
        {
            std::ifstream file(traj_file);
            if (!file.is_open()) {
                std::cerr << "Error: Could not open trajectory file." << std::endl;
                return 1;
            }
            std::string line;
            std::getline(file, line); // Skip header
            while (std::getline(file, line)) {
                // Read the comma-separated values
                int64_t timestamp;
                std::array<double, 6> values;
                std::istringstream iss(line);
                if(std::getline(iss, line, ','))
                {
                    timestamp = std::stoll(line);
                    for(int j = 0; j < 6; j++)
                    {
                        if(!std::getline(iss, line, ','))
                        {
                            std::cerr << "Error: Malformed trajectory line." << std::endl;
                            continue;
                        }
                        values[j] = std::stod(line);
                    }
                }
                else
                {
                    std::cerr << "Error: Malformed trajectory line." << std::endl;
                    continue;
                }
                double tx = values[0];
                double ty = values[1];
                double tz = values[2];
                double rx = values[3];
                double ry = values[4];
                double rz = values[5];

                traj_times.push_back(timestamp);
                Mat4 pose = Mat4::Identity();
                pose.block<3,3>(0,0) = expMap(Vec3(rx, ry, rz));
                pose(0,3) = tx;
                pose(1,3) = ty;
                pose(2,3) = tz;
                traj_poses.push_back(pose);
            }
        }
        std::cout << "Loaded " << traj_times.size() << " trajectory poses." << std::endl;
        // For each pose in the trajectory, find the closest scan
        for(size_t j = 0; j < traj_times.size(); j++)
        {
            std::cout << "Processing trajectory pose at time: " << j << " / " << traj_times.size() << std::endl;
            int64_t traj_time = traj_times[j];
            Mat4 traj_pose = traj_poses[j];

            // Find closest scan
            int64_t min_time_diff = std::numeric_limits<int64_t>::max();
            std::string closest_scan;
            for (const auto& scan : scans) {
                int64_t time_diff = std::abs(scan.first - traj_time);
                if (time_diff < min_time_diff) {
                    min_time_diff = time_diff;
                    closest_scan = scan.second;
                }
            }
            if (min_time_diff > 1e6) { // 1ms threshold
                std::cout << "Warning: No close scan found for trajectory time " << traj_time << std::endl;
                continue;
            }

            

            // Carve free space using the closest scan and the trajectory pose
            std::vector<Pointd> scan_pts = loadPointCloudFromPly(closest_scan);
            map.freeSpaceCarving(scan_pts, traj_pose);
        }

        // Save cleaned map
        std::string output_path = map_path;
        // Replace .ply with _cleaned.ply
        map.writeMap(output_path);
        std::cout << "Saved cleaned map to: " << output_path << std::endl;

    }

    return 0;
}