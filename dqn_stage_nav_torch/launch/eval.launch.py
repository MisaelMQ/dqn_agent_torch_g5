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

                # QoS
                {"qos_depth": 1},
                {"qos_best_effort": True},
                {"qos_transient_local": False},
                {"pub_reset_count_topic": "/reset_count"},
            ],
        ),

        Node(
            package="dqn_stage_nav_torch",
            executable="eval_node",
            name="dqn_eval_node",
            output="screen",
            parameters=[
                {"scan_topic": "/base_scan"},
                {"odom_topic": "/odom/sim"},
                {"raw_odom_topic": "/ground_truth"},
                {"use_raw_odom_for_global": True},
                {"cmd_vel_topic": "/cmd_vel"},
                {"reset_service": "/reset_sim"},

                # Model + goals 
                {"model_path": ""},  
                {"goals_csv_path": ""}, 
                {"results_csv_path": ""},  
                {"device": "cuda"},

                {"trials_per_goal": 1},
                {"shuffle_goals": False},
                {"max_goals": 0},

                # Control + termination
                {"control_dt": 0.01},
                {"max_steps": 800},
                {"collision_range": 0.25},

                {"goal_tolerance_enter": 0.45},
                {"goal_tolerance_exit": 1.00},
                {"goal_hold_steps": 8},

                # State
                {"lidar_bins": 20},
                {"lidar_max_range": 4.5},
                {"lidar_fov_deg": 270.0},
                {"front_sector_deg": 30.0},
                {"max_goal_dist": 15.0},

                # Actions
                {"v_max": 1.25},
                {"w_max": 2.00},

                # Stuck / oscillation
                {"stuck_window_steps_far": 30},
                {"stuck_move_eps_far": 0.08},
                {"stuck_window_steps_near": 60},
                {"stuck_move_eps_near": 0.05},

                {"osc_window_steps": 24},
                {"osc_allow_only_rot": True},
                {"osc_alt_ratio": 0.80},
                {"near_goal_max_steps_without_hold": 160},

                # Optional: action masking
                {"mask_stop_dist": 0.28},
                {"mask_slow_dist": 0.48},

                {"print_every_steps": 50},
            ],
        ),
    ])