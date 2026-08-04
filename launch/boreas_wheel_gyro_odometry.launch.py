from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


# Calibration of the boreas platform (2024 and later sequences, the IMU is the "dmu" one).
# T_imu_lidar     = inv(T_applanix_dmu) @ T_applanix_lidar
# T_imu_wheel     = inv(T_applanix_dmu) @ T_applanix_wheel
# The wheel frame is the one the encoder measures the travelled distance along its y axis.

# Travelled distance per encoder count (in meters), as estimated by the wheel-gyro calibration
wheel_scale = float(0.00052209)


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
            ],
            output='screen',
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
