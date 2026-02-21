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

                {"episodes_total": 10000},
                {"episodes_per_goal": 100},
                {"max_steps_per_episode": 600},

                # Eval for detecting Degradation
                {"eval_every_episodes": 200},
                {"eval_disable_training": True},
                {"rollback_on_eval_drop": True},
                {"eval_window": 10},
                {"eval_min_success_keep": 0.45},

                {"lidar_bins": 20},
                {"lidar_max_range": 4.5},
                {"lidar_fov_deg": 270.0},
                {"front_sector_deg": 30.0},

                {"v_max": 1.25},
                {"w_max": 2.00},

                {"goal_tolerance": 0.40},
                {"collision_range": 0.25},

                {"step_penalty": -0.02},
                {"progress_scale": 3.0},
                {"orient_scale": 0.10},

                {"obstacle_near_dist": 0.72},
                {"obstacle_near_scale": 2.6},
                {"obstacle_power": 2.0},

                {"front_gate_safe": 0.35},
                {"front_gate_open": 1.05},

                {"clearance_scale": 0.06},
                {"spin_penalty": 0.03},

                {"mask_stop_dist": 0.28},
                {"mask_slow_dist": 0.48},

                {"goal_reward": 200.0},
                {"collision_penalty": -90.0},
                {"timeout_extra_penalty": -20.0},
                {"stuck_extra_penalty": -25.0},

                {"near_goal_radius": 1.0},
                {"near_goal_tau": 0.75},
                {"near_goal_max_frac": 0.33},

                {"far_start": 9.0},
                {"far_tau": 4.0},
                {"far_max": 4.0},
                {"far_terminate": 12.0},

                {"min_steps_before_far": 80},
                {"far_margin": 2.0},
                {"far_terminate_override": 0.0},

                # Escape shaping 
                {"escape_enable": True},
                {"escape_sector_deg": 35.0},
                {"escape_narrow_dist": 0.40},
                {"escape_goal_bearing_deg": 35.0},
                {"escape_forward_penalty": 0.25},

                {"stuck_window": 120},
                {"stuck_min_move": 0.15},
                {"stuck_min_progress": 0.06},

                {"curriculum_window": 80},
                {"unlock_medium_success": 0.70},
                {"unlock_hard_success": 0.45},
                {"curriculum_mix_easy_prob": 0.10},

                {"epsilon_boost_on_goal_change": 0.15},

                {"agent_gamma": 0.99},
                {"agent_lr": 1e-4},
                {"agent_batch_size": 256},
                {"agent_memory_size": 100000},
                {"agent_min_memory": 5000},

                {"agent_epsilon_start": 1.0},
                {"agent_epsilon_min": 0.05},
                {"agent_epsilon_decay": 0.9999},

                # Late Stability
                {"lr_decay_every_episodes": 2000},
                {"lr_decay_factor": 0.5},
                {"success_replay_boost": 0},

                {"agent_soft_tau": 0.005},
                {"agent_hard_update_every": 0},
                {"agent_grad_norm": 10.0},

                {"agent_use_per": True},
                {"agent_per_alpha": 0.5},
                {"agent_per_beta_start": 0.4},
                {"agent_per_beta_frames": 250000},
                {"agent_per_eps": 1e-3},
                {"agent_per_prio_clip": 100.0},
                {"agent_n_step": 3},

                {"agent_device": "cuda"},
                {"agent_seed": 42},

                {"save_every_episodes": 50},
                {"keep_last_n_checkpoints": 8},
                {"print_every_steps": 50},

                {"control_dt": 0.001667},
                {"reset_timeout_sec": 2.5},
            ],
        ),
    ])