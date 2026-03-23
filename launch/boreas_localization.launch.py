from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
from ament_index_python.packages import get_package_prefix


min_range = float(5.0)
max_range = float(200.0)
voxel_size = float(0.3)

key_framing = True
key_frame_dist_thr = float(10.0)
key_frame_rot_thr = float(5.0 * 3.14 / 180.0)
key_frame_time_thr = float(0.5)



def generate_launch_description():
    rviz_file = PathJoinSubstitution(
           [FindPackageShare("ffastllamaa"), "cfg", "rviz_config.rviz"])
    return LaunchDescription([
        Node(
            package='ffastllamaa', 
            executable='lidar_scan_odometry', 
            name='lidar_scan_odometry',
            remappings=[
                ('/imu/acc', '/imu/data'),
                ('/imu/gyr', '/imu/data'),
                ('/lidar_raw_points', '/velodyne_points')
            ],
            parameters=[
                {'dense_pc_output': False}, # Set to True to output dense point cloud
                {'min_range': float(min_range)},
                {'max_range': float(max_range)},
                {'feature_voxel_size': float(voxel_size)},
                {"max_associations_per_type": 1000},
                {"planar_only": False},
                {"minimum_intensity": 2.0},
                {"mode": "imu"},  # State representation mode: imu (acc and gyr preint), gyr (gyr preint and const vel), no_imu (const linear and angular vel)


                # Adapting IMU measurements for some weird IMUs
                {"acc_in_m_per_s2": True},
                {"invert_imu": False},

                # Calibration
                {"calib_px": 0.},
                {"calib_py": 0.},
                {"calib_pz": -0.28},
                {"calib_rx": 2.92077461},
                {"calib_ry": -1.15627809},
                {"calib_rz": -0.00226139},

                # In case the point cloud is not sorted by time, set this to True
                {"unsorted_pc": False},

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

                {"voxel_size": float(voxel_size)},
                {"max_num_pts_for_registration": 8000},
                
                {"key_framing": key_framing},
                {"key_framing_dist_thr": key_frame_dist_thr},
                {"key_framing_rot_thr": key_frame_rot_thr},
                {"key_framing_time_thr": key_frame_time_thr},

                {"min_range": float(min_range)},
                # Free space carving (<= 0.0 to disable it)
                {"free_space_carving_radius": float(-50)},

                # Path to where the map will be saved
                {"map_path": get_package_prefix('ffastllamaa') + "/share/ffastllamaa/maps/"},
                {"using_submaps": True},
                {"reverse_path": False},

                {"write_scans": False}
            ],
            output='screen',
        ),
        
        Node(package = "tf2_ros", 
                       executable = "static_transform_publisher",
                       arguments = ["0", "0", "0", "0", "0", "3.14",  "map", "map_viz"]),
        Node(
            package='rviz2', 
            executable='rviz2', 
            name='rviz2',
            output='screen',
            arguments=['-d' , rviz_file],
        )
    ])
