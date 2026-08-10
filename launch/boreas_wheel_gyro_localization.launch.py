from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
from ament_index_python.packages import get_package_prefix


# Localization in a prebuilt map, using the wheel-gyro odometry instead of the lidar-based one.
# Calibration of the boreas platform (2024 and later sequences, the IMU is the "dmu" one).
# T_imu_lidar     = inv(T_applanix_dmu) @ T_applanix_lidar
# T_imu_wheel     = inv(T_applanix_dmu) @ T_applanix_wheel
# The wheel frame is the one the encoder measures the travelled distance along its y axis.

# Travelled distance per encoder count (in meters), as estimated by the wheel-gyro calibration
wheel_scale = float(0.00052209)

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
            executable='wheel_gyro_odometry',
            name='wheel_gyro_odometry',
            remappings=[
                ('/imu/gyr', '/imu/data'),
                ('/lidar_raw_points', '/velodyne_points'),
                ('/wheel_encoder', '/wheel_encoder')
            ],
            parameters=[
                # Wheel encoder to travelled distance
                {"wheel_scale": float(wheel_scale)},
                {"encoder_wrap_max": 16777216},  # 2^24, the boreas dmi counter rolls over there

                # Calibration between the IMU and the lidar (T_imu_lidar)
                {"calib_px": 0.},
                {"calib_py": 0.},
                {"calib_pz": -0.28},
                {"calib_rx": 2.92077461},
                {"calib_ry": -1.15627809},
                {"calib_rz": -0.00226139},

                # Calibration between the IMU and the wheel encoder (T_wheel_imu)
                {"wheel_calib_px": 0.84743934},
                {"wheel_calib_py": 0.42288179},
                {"wheel_calib_pz": 1.46239841},
                {"wheel_calib_rx": -3.13609754},
                {"wheel_calib_ry": -0.0305978795},
                {"wheel_calib_rz": 0.00213185588},


                # Gyroscope bias (in the IMU frame), estimated online over the windows where the
                # wheel encoder does not tick
                {"estimate_gyr_bias": False},
                {"gyr_bias_init_x": 0.},
                {"gyr_bias_init_y": 0.},
                {"gyr_bias_init_z": 0.},
                {"zero_vel_required_time": 3.0},
                {"bias_use_time": 2.0},
                {"bias_max_use_time": 10.0},

                # Point cloud undistortion
                {"undistort_pc": True},
                {"point_time_multiplier": 1e-9},
                {"absolute_time": False},

                # Feature extraction / downsampling of the point clouds, same parameters as the
                # lidar_scan_odometry node
                {"extract_features": True},
                {"min_range": float(min_range)},
                {"max_range": float(max_range)},
                {"max_feature_range": float(max_range)},
                {"feature_voxel_size": float(voxel_size)},
                {"planar_only": False},
                {"minimum_intensity": 2.0},
                # In case the point cloud is not sorted by time, set this to True
                {"unsorted_pc": False},
                # Set to True to also output the undistorted dense point cloud
                {"dense_pc_output": False},
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

                # Path to where the map is loaded from
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
