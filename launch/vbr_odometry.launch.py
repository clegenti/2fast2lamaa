from launch import LaunchDescription
from launch_ros.actions import Node
from launch.events import Shutdown
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
from ament_index_python.packages import get_package_prefix
from launch.actions import SetEnvironmentVariable
SetEnvironmentVariable(name='RCUTILS_COLORIZED_OUTPUT', value='1'),


min_range = float(1.0)
max_range = float(200.0)
voxel_size = float(0.3)

key_framing = False


def generate_launch_description():
    rviz_file = PathJoinSubstitution(
           [FindPackageShare("ffastllamaa"), "cfg", "rviz_config.rviz"])
    return LaunchDescription([
        Node(
            package='ffastllamaa', 
            executable='lidar_scan_odometry', 
            name='lidar_scan_odometry',
            remappings=[
                ('/imu/acc', '/ouster/imu'),
                ('/imu/gyr', '/ouster/imu'),
                ('/lidar_raw_points', '/ouster/points')
            ],
            parameters=[
                {'dense_pc_output': False}, # Set to True to output dense point cloud
                {'min_range': float(min_range)},
                {'max_range': float(max_range)},
                {'max_feature_range': float(max_range)},
                {'feature_voxel_size': float(voxel_size)},
                {"max_associations_per_type": 1000},
                {"planar_only": False},

                # Adapting IMU measurements for some weird IMUs
                {"acc_in_m_per_s2": True},
                {"invert_imu": False},

                ## Calibration
                {"calib_px": -0.0062},
                {"calib_py": 0.0118},
                {"calib_pz": -0.0076},
                {"calib_rx": 0.},
                {"calib_ry": 0.},
                {"calib_rz": 0.},
                
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
                ('/gp_map/acc', '/ouster/imu'),
                ('/gp_map/gyr', '/ouster/imu'),
                ('/twist', '/start_of_scan_twist')
                ],
            parameters=[
                {"voxel_size": float(voxel_size)},
                {"max_num_pts_for_registration": 2000},

                {"key_framing": key_framing},

                {"min_range": float(min_range)},
                # Free space carving (<= 0.0 to disable it)
                {"free_space_carving_radius": float(50)},

                # Path to where the map will be saved
                {"map_path": get_package_prefix('ffastllamaa') + "/share/ffastllamaa/maps/"},

                {"submap_length": float(50.0)},

                {"write_scans": True}

            ],
            output='screen',
            on_exit=Shutdown()
        ),
        Node(package = "ffastllamaa",
             executable = "pose_graph",
             name = "pose_graph",
             output = "screen",
             emulate_tty=True,
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
