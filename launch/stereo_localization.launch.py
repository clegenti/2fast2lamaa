from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import Shutdown
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
from ament_index_python.packages import get_package_prefix


min_range = float(0.5)
max_range = float(200.0)
voxel_size = float(0.1)

key_framing = False
key_frame_dist_thr = float(10.0)
key_frame_rot_thr = float(15.0 * 3.14 / 180.0)
key_frame_time_thr = float(0.5)


def generate_launch_description():
    rviz_file = PathJoinSubstitution(
           [FindPackageShare("ffastllamaa"), "cfg", "rviz_event_config.rviz"])
    return LaunchDescription([
        Node(
            package='ffastllamaa',
            executable='gp_map',
            name='gp_map',
            remappings=[
                ('/points_input', '/okvis_point_cloud'),
                ('/pose_input', '/okvis_pose'),
                ],
            parameters=[
                {"localization_only": True},
                # The initial pose comes from the lidar trajectory of the mapping run, so the first
                # scan does not need the coarse-to-fine registrations meant to recover a rough guess
                {"can_trust_init": True},
                # The front-end publishes on its own keyframes, not at a fixed rate, so the gap between
                # two inputs carries no information about a dropped frame
                {"use_frame_dropout_detection": False},
                {"init_pose_x": 0.0},
                {"init_pose_y": 0.0},
                {"init_pose_z": 0.0},
                {"init_pose_rx": 0.0},
                {"init_pose_ry": 0.0},
                {"init_pose_rz": 0.0},
                {"point_cloud_internal_type": False},

                {"voxel_size": float(voxel_size)},
                {"max_num_pts_for_registration": 2000},
                {"voxel_size_factor_for_registration": 2.0},

                {"use_odom_prior": True},
                {"odom_prior_weight_pos": 0.01},
                {"odom_prior_weight_rot": 0.001},

                # Estimate a scale of the scans alongside the pose. `scale_prior_weight` is the inverse
                # of the standard deviation of the scale change between two consecutive scans.
                {"use_scale_optimization": False},
                {"scale_prior_weight": 100.0},

                # Weight the registration with the per-point covariance of the input cloud (which then
                # needs to carry the 6 cov_xx..cov_zz fields). The robust loss then applies to the
                # mahalanobis distance, so `loss_function_scale` becomes a number of standard
                # deviations (default 1.0) instead of a distance in meters.
                {"use_point_covariances": True},
                {"point_covariances_per_iteration": False},
                {"point_covariances_min_std": 0.0},
                {"loss_function_scale": 0.2},

                {"key_framing": key_framing},
                {"key_framing_dist_thr": key_frame_dist_thr},
                {"key_framing_rot_thr": key_frame_rot_thr},
                {"key_framing_time_thr": key_frame_time_thr},

                {"min_range": float(min_range)},
                # Free space carving (<= 0.0 to disable it)
                {"free_space_carving_radius": float(-50)},

                # Path to the map to localize in
                {"map_path": "/map/path"},
                {"using_submaps": False},

                {"write_scans": False}
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
