from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import Shutdown
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
from ament_index_python.packages import get_package_prefix
import numpy as np


min_range = float(1.5)
max_range = float(150.0)
voxel_size = float(0.15)

key_framing = False

# IMU, shared by lidar_scan_odometry and gp_map (gravity factor and IMU-rate estimator)
imu_topic = '/os1_cloud_node/imu'
acc_in_m_per_s2 = True      # False if the accelerometer measures in g
invert_imu = False          # Flip the sign of the measurements (for some weird IMUs)
gravity = 9.81              # m/s^2
acc_std = 0.02              # m/s^2
gyr_std = 0.005             # rad/s
use_imu_estimator = True    # Pose at the IMU rate on /imu_rate_odom (see the README)


def generate_launch_description():
    rviz_file = PathJoinSubstitution(
           [FindPackageShare("ffastllamaa"), "cfg", "rviz_config.rviz"])
    return LaunchDescription([
        Node(
            package='ffastllamaa', 
            executable='lidar_scan_odometry', 
            name='lidar_scan_odometry',
            remappings=[
                ('/imu/acc', imu_topic),
                ('/imu/gyr', imu_topic),
                ('/lidar_raw_points', '/os1_cloud_node/points')
            ],
            parameters=[
                {'dense_pc_output': False}, # Set to True to output dense point cloud
                {'min_range': float(min_range)},
                {'max_range': float(max_range)},
                {'feature_voxel_size': float(voxel_size)},
                {"max_associations_per_type": 1000},
                {"planar_only": True},
                {"mode": "imu"},  # State representation mode: imu (acc and gyr preint), gyr (gyr preint and const vel), no_imu (const linear and angular vel)

                # IMU (shared with gp_map, see the top of the file)
                {"acc_in_m_per_s2": acc_in_m_per_s2},
                {"invert_imu": invert_imu},
                {"g": gravity},
                {"acc_std": acc_std},
                {"gyr_std": gyr_std},

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
                ('/gp_map/acc', imu_topic),
                ('/gp_map/gyr', imu_topic),
                ('/twist', '/start_of_scan_twist')
                ],
            parameters=[
                {"voxel_size": float(voxel_size)},
                {"max_num_pts_for_registration": 2000},

                {"key_framing": key_framing},

                {"min_range": float(min_range)},
                # Free space carving (<= 0.0 to disable it)
                {"free_space_carving_radius": float(-50)},

                # Path to where the map will be saved
                {"map_path": get_package_prefix('ffastllamaa') + "/share/ffastllamaa/maps/"},

                {"submap_length": -200.0},
                # IMU (shared with lidar_scan_odometry, see the top of the file)
                {"acc_in_m_per_s2": acc_in_m_per_s2},
                {"invert_imu": invert_imu},
                {"acc_std": acc_std},
                {"gyr_std": gyr_std},
                {"use_imu_estimator": use_imu_estimator},
                {"imu_estimator_gravity_norm": gravity}

            ],
            output='screen',
            on_exit=Shutdown()
        ),
        Node(package = "tf2_ros", 
                       executable = "static_transform_publisher",
                       arguments = ["--x", "0", "--y", "0", "--z", "0", "--yaw", "0", "--pitch", "0", "--roll", "0", "--frame-id", "map", "--child-frame-id", "map_viz"]),
        Node(
            package='rviz2', 
            executable='rviz2', 
            name='rviz2',
            output='screen',
            arguments=['-d' , rviz_file],
        )
    ])
