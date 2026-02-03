from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="dqn_stage_nav_torch",
            executable="odom_reset_wrapper",
            name="odom_reset_wrapper",
            output="screen",
            parameters=[
                {"sub_odom_topic": "/odom"},
                {"pub_odom_topic": "/odom/sim"},
                {"stage_reset_service": "/reset_positions"},
                {"wrapper_reset_service": "/reset_sim"},
                {"zero_position_eps": 0.08},
            ],
        ),

        Node(
            package="dqn_stage_nav_torch",
            executable="train_node",
            name="train_node",
            output="screen",
            parameters=[
                {"scan_topic": "/base_scan"},
                {"odom_topic": "/odom/sim"},
                {"raw_odom_topic": "/ground_truth"},
                {"use_raw_odom_for_global": True},
                {"cmd_vel_topic": "/cmd_vel"},
                {"reset_service": "/reset_sim"},

                # Training
                {"episodes_total": 4000},
                {"episodes_per_goal": 50},
                {"max_steps_per_episode": 500},

                # Lidar
                {"lidar_bins": 20},
                {"lidar_max_range": 4.5},

                # Control (Stage)
                {"v_max": 1.25},
                {"w_max": 2.00},

                # Umbrales
                {"goal_radius": 0.40},
                {"collision_radius": 0.25},

                # Rewards base
                {"step_penalty": -0.02},
                {"progress_scale": 3.0},
                {"obstacle_near_dist": 0.50},
                {"obstacle_near_scale": 2.0},

                {"goal_reward": 200.0},
                {"collision_penalty": -80.0},
                {"timeout_penalty": -20.0},

                {"near_goal_radius": 1.0},
                {"near_goal_tau": 0.75},
                {"near_goal_max_frac": 0.33},
                {"far_start": 7.5},
                {"far_tau": 2.0},
                {"far_max": 100.0},
                {"far_terminate": 12.0},

                # Stuck
                {"stuck_window": 100},
                {"stuck_min_progress": 0.10},
                {"stuck_penalty": -20.0},

                # Saving
                {"save_every_episodes": 50},
                {"checkpoint_every_episodes": 100},
                {"max_checkpoints": 20},
            ],
        ),
    ])
