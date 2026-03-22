from launch import LaunchDescription
from launch_ros.actions import Node
from launch.events import Shutdown
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
from ament_index_python.packages import get_package_prefix
import numpy as np


min_range = float(1.5)
max_range = float(60.0)
voxel_size = float(0.15)

key_framing = True
key_frame_dist_thr = float(1.5)
key_frame_rot_thr = float(15.0 * 3.14 / 180.0)
key_frame_time_thr = float(0.05)

over_reject = False # If true, also reject the neighborhood points when performing the dynamic filtering and free space carving.


def generate_launch_description():
    rviz_file = PathJoinSubstitution(
           [FindPackageShare("ffastllamaa"), "cfg", "rviz_config.rviz"])
    return LaunchDescription([
        Node(
            package='ffastllamaa', 
            executable='lidar_scan_odometry', 
            name='lidar_scan_odometry',
            remappings=[
                ('/imu/acc', '/os1_cloud_node/imu'),
                ('/imu/gyr', '/os1_cloud_node/imu'),
                ('/lidar_raw_points', '/os1_cloud_node/points')
            ],
            parameters=[
                {'dense_pc_output': False}, # Set to True to output dense point cloud
                {'min_range': float(min_range)},
                {'max_range': float(max_range)},
                {'feature_voxel_size': float(voxel_size)},
                {"state_freq": 200.0},
                {"max_associations_per_type": 1000},
                {"planar_only": True},
                {"g": 9.81},

                # Adapting IMU measurements for some weird IMUs
                {"acc_in_m_per_s2": True},
                {"invert_imu": False},

                # Calibration
                {"calib_px": -0.006253},
                {"calib_py": 0.011775},
                {"calib_pz": 0.028535},
                {"calib_rx": 0.0},
                {"calib_ry": 0.0},
                {"calib_rz": np.pi},

                # In case the point cloud is not sorted by time, set this to True
                {"unsorted_pc": True},
                {"point_cloud_scale": 0.997},

            ],
            output='screen',
        ),
        Node(
            package='ffastllamaa', 
            executable='gp_map', 
            name='gp_map',
            remappings=[
                ('/points_input', '/lidar_scan_undistorted'),
                ('/pose_input', '/undistortion_pose'),
                ],
            parameters=[
                {"localization_only": True},
                {"init_pose_x": 0.0},
                {"init_pose_y": 0.0},
                {"init_pose_z": 0.0},
                {"init_pose_rx": 0.0},
                {"init_pose_ry": 0.0},
                {"init_pose_rz": 0.0},

                {"point_cloud_internal_type": True},
                {"voxel_size": float(voxel_size)},
                {"neighbourhood_size": 2},
                {"register": True},
                {"register_with_approximate_field": False},
                {"voxel_size_factor_for_registration": 2.0},
                {"max_num_pts_for_registration": 1500},
                {"use_temporal_weights": True}, # If true, registration weight are 10 times bigger for voxels associated to the older scans than for the newer ones
                {"with_init_guess": True},
                {"map_publish_period": 0.5},
                {"key_framing": key_framing},
                {"key_framing_dist_thr": key_frame_dist_thr},
                {"key_framing_rot_thr": key_frame_rot_thr},
                {"key_framing_time_thr": key_frame_time_thr},

                # Free space carving (<= 0.0 to disable it)
                {"min_range": float(min_range)},
                {"free_space_carving_radius": float(-20)},
                {"over_reject": over_reject},
                {"last_scan_carving": False},

                # Path to where the map will be saved
                {"map_path": "/home/ced/Documents/code/2fast2lamaa-private/newer_college_output/01_short/"},
                {"map_type": "map"},
                {"using_submaps": False},

                {"submap_length": -200.0},
                {"submap_overlap": 0.2}

            ],
            output='screen',
            on_exit=Shutdown()
        ),
        Node(package = "tf2_ros", 
                       executable = "static_transform_publisher",
                       arguments = ["0", "0", "0", "0", "0", "0",  "map", "map_viz"]),
        Node(
            package='rviz2', 
            executable='rviz2', 
            name='rviz2',
            output='screen',
            arguments=['-d' , rviz_file],
        )
    ])
