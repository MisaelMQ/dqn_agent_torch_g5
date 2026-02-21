#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import os
import random
import time
import traceback
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty

from dqn_stage_nav_torch.torch_dqn_agent import TorchDQNAgent, TorchDQNConfig
from dqn_stage_nav_torch.state_processor import StateProcessor, StateProcessorConfig


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quaternion(q) -> float:
    x, y, z, w = q.x, q.y, q.z, q.w
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def angle_to_goal(gx: float, gy: float, yaw: float, goalx: float, goaly: float) -> float:
    desired = math.atan2(goaly - gy, goalx - gx)
    return wrap_pi(desired - yaw)


@dataclass(frozen=True)
class Action:
    name: str
    v: float
    w: float


def build_action_set(v_max: float, w_max: float) -> List[Action]:
    return [
        Action("ROT_L", 0.0, +0.8 * w_max),
        Action("ROT_R", 0.0, -0.8 * w_max),
        Action("FWD_S", 0.35 * v_max, 0.0),
        Action("FWD_F", 1.00 * v_max, 0.0),
        Action("ARC_L", 0.70 * v_max, +0.45 * w_max),
        Action("ARC_R", 0.70 * v_max, -0.45 * w_max),
    ]


_GOALS_WITH_DIFF: List[Tuple[Tuple[float, float], str]] = [
    ((-7.0, 0.0), "easy"),
    ((0.0, -6.0), "easy"),
    ((-2.0, 2.0), "easy"),
    ((6.0, -3.0), "medium"),
    ((-3.0, -2.0), "medium"),
    ((-6.0, 5.0), "medium"),
    ((5.0, 5.0), "hard"),
    ((3.0, 7.0), "hard"),
    ((7.0, 3.0), "hard"),
    ((-2.0, 7.0), "hard"),
]


def build_curriculum_by_difficulty(goals_data, order=("easy", "medium", "hard")):
    buckets = {d: [] for d in order}
    for xy, diff in goals_data:
        if diff in buckets:
            buckets[diff].append((xy, diff))
    curriculum = []
    for d in order:
        curriculum.extend(buckets[d])
    return curriculum


class DQNTrainNode(Node):
    def __init__(self):
        super().__init__("train_node")

        # Topics / Services
        self.declare_parameter("scan_topic", "/base_scan")
        self.declare_parameter("odom_topic", "/odom/sim")
        self.declare_parameter("raw_odom_topic", "/ground_truth")
        self.declare_parameter("use_raw_odom_for_global", True)
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("reset_service", "/reset_sim")

        # Schedule
        self.declare_parameter("episodes_total", 10000)
        self.declare_parameter("episodes_per_goal", 100)
        self.declare_parameter("max_steps_per_episode", 600)

        # Eval (para detectar degradación)
        self.declare_parameter("eval_every_episodes", 200)
        self.declare_parameter("eval_disable_training", True)
        self.declare_parameter("rollback_on_eval_drop", True)
        self.declare_parameter("eval_window", 10)
        self.declare_parameter("eval_min_success_keep", 0.45)

        # Lidar
        self.declare_parameter("lidar_bins", 20)
        self.declare_parameter("lidar_max_range", 4.5)
        self.declare_parameter("lidar_fov_deg", 270.0)
        self.declare_parameter("front_sector_deg", 30.0)

        # Dynamics
        self.declare_parameter("v_max", 1.25)
        self.declare_parameter("w_max", 2.00)

        # Termination
        self.declare_parameter("goal_tolerance", 0.40)
        self.declare_parameter("collision_range", 0.25)

        # Stuck
        self.declare_parameter("stuck_window", 120)
        self.declare_parameter("stuck_min_move", 0.15)
        self.declare_parameter("stuck_min_progress", 0.06)  # AND + más permisivo

        # Exploration
        self.declare_parameter("epsilon_boost_on_goal_change", 0.15)

        # Reward shaping
        self.declare_parameter("step_penalty", -0.02)
        self.declare_parameter("progress_scale", 3.0)
        self.declare_parameter("orient_scale", 0.10)

        self.declare_parameter("obstacle_near_dist", 0.72)
        self.declare_parameter("obstacle_near_scale", 2.6)
        self.declare_parameter("obstacle_power", 2.0)

        self.declare_parameter("front_gate_safe", 0.35)
        self.declare_parameter("front_gate_open", 1.05)

        self.declare_parameter("clearance_scale", 0.06)
        self.declare_parameter("spin_penalty", 0.03)

        # Action masking (ligero)
        self.declare_parameter("mask_stop_dist", 0.28)
        self.declare_parameter("mask_slow_dist", 0.48)

        # Terminal rewards
        self.declare_parameter("goal_reward", 200.0)
        self.declare_parameter("collision_penalty", -90.0)
        self.declare_parameter("timeout_extra_penalty", -20.0)
        self.declare_parameter("stuck_extra_penalty", -25.0)

        # Near/Far shaping
        self.declare_parameter("near_goal_radius", 1.0)
        self.declare_parameter("near_goal_tau", 0.75)
        self.declare_parameter("near_goal_max_frac", 0.33)

        self.declare_parameter("far_start", 9.0)
        self.declare_parameter("far_tau", 4.0)
        self.declare_parameter("far_max", 4.0)
        self.declare_parameter("far_terminate", 12.0)

        self.declare_parameter("min_steps_before_far", 80)
        self.declare_parameter("far_margin", 2.0)
        self.declare_parameter("far_terminate_override", 0.0)

        # Escape shaping 
        self.declare_parameter("escape_enable", True)
        self.declare_parameter("escape_sector_deg", 35.0)        
        self.declare_parameter("escape_narrow_dist", 0.40)        
        self.declare_parameter("escape_goal_bearing_deg", 35.0)   
        self.declare_parameter("escape_forward_penalty", 0.25)    

        # Curriculum
        self.declare_parameter("curriculum_window", 80)
        self.declare_parameter("unlock_medium_success", 0.70)
        self.declare_parameter("unlock_hard_success", 0.45)
        self.declare_parameter("curriculum_mix_easy_prob", 0.10)

        # Agente
        self.declare_parameter("agent_gamma", 0.99)
        self.declare_parameter("agent_lr", 1e-4)
        self.declare_parameter("agent_batch_size", 256)
        self.declare_parameter("agent_memory_size", 100_000)
        self.declare_parameter("agent_min_memory", 5_000)

        self.declare_parameter("agent_epsilon_start", 1.0)
        self.declare_parameter("agent_epsilon_min", 0.05)
        self.declare_parameter("agent_epsilon_decay", 0.9999)

        # Extra Stability
        self.declare_parameter("lr_decay_every_episodes", 2000)
        self.declare_parameter("lr_decay_factor", 0.5)
        self.declare_parameter("success_replay_boost", 0) 

        self.declare_parameter("agent_soft_tau", 0.005)  
        self.declare_parameter("agent_hard_update_every", 0)  

        self.declare_parameter("agent_grad_norm", 10.0)

        self.declare_parameter("agent_use_per", True)
        self.declare_parameter("agent_per_alpha", 0.5)       
        self.declare_parameter("agent_per_beta_start", 0.4)
        self.declare_parameter("agent_per_beta_frames", 250_000)
        self.declare_parameter("agent_per_eps", 1e-3)
        self.declare_parameter("agent_per_prio_clip", 100.0)

        self.declare_parameter("agent_n_step", 3)
        self.declare_parameter("agent_device", "cuda")
        self.declare_parameter("agent_seed", 42)

        # Saving / Logging
        self.declare_parameter("save_every_episodes", 50)
        self.declare_parameter("keep_last_n_checkpoints", 8)
        self.declare_parameter("print_every_steps", 50)

        # Loop
        self.declare_parameter("control_dt", 0.0033)
        self.declare_parameter("reset_timeout_sec", 2.5)

        p = self.get_parameter
        self.scan_topic = str(p("scan_topic").value)
        self.odom_topic = str(p("odom_topic").value)
        self.raw_odom_topic = str(p("raw_odom_topic").value)
        self.use_raw_odom_for_global = bool(p("use_raw_odom_for_global").value)
        self.cmd_vel_topic = str(p("cmd_vel_topic").value)
        self.reset_service = str(p("reset_service").value)

        self.episodes_total = int(p("episodes_total").value)
        self.episodes_per_goal = int(p("episodes_per_goal").value)
        self.max_steps = int(p("max_steps_per_episode").value)

        self.eval_every = int(p("eval_every_episodes").value)
        self.eval_disable_training = bool(p("eval_disable_training").value)
        self.rollback_on_eval_drop = bool(p("rollback_on_eval_drop").value)
        self.eval_window = int(p("eval_window").value)
        self.eval_min_success_keep = float(p("eval_min_success_keep").value)

        self.lidar_bins = int(p("lidar_bins").value)
        self.lidar_max = float(p("lidar_max_range").value)
        self.lidar_fov_deg = float(p("lidar_fov_deg").value)
        self.front_sector_deg = float(p("front_sector_deg").value)

        self.v_max = float(p("v_max").value)
        self.w_max = float(p("w_max").value)

        self.goal_tol = float(p("goal_tolerance").value)
        self.collision_range = float(p("collision_range").value)

        self.stuck_window = int(p("stuck_window").value)
        self.stuck_min_move = float(p("stuck_min_move").value)
        self.stuck_min_progress = float(p("stuck_min_progress").value)

        self.eps_boost = float(p("epsilon_boost_on_goal_change").value)

        self.step_penalty = float(p("step_penalty").value)
        self.progress_scale = float(p("progress_scale").value)
        self.orient_scale = float(p("orient_scale").value)

        self.obstacle_near_dist = float(p("obstacle_near_dist").value)
        self.obstacle_near_scale = float(p("obstacle_near_scale").value)
        self.obstacle_power = float(p("obstacle_power").value)

        self.front_gate_safe = float(p("front_gate_safe").value)
        self.front_gate_open = float(p("front_gate_open").value)

        self.clearance_scale = float(p("clearance_scale").value)
        self.spin_penalty = float(p("spin_penalty").value)

        self.mask_stop_dist = float(p("mask_stop_dist").value)
        self.mask_slow_dist = float(p("mask_slow_dist").value)

        self.goal_reward = float(p("goal_reward").value)
        self.collision_penalty = float(p("collision_penalty").value)
        self.timeout_extra_penalty = float(p("timeout_extra_penalty").value)
        self.stuck_extra_penalty = float(p("stuck_extra_penalty").value)

        self.near_goal_radius = float(p("near_goal_radius").value)
        self.near_goal_tau = float(p("near_goal_tau").value)
        self.near_goal_max_frac = float(p("near_goal_max_frac").value)

        self.far_start = float(p("far_start").value)
        self.far_tau = float(p("far_tau").value)
        self.far_max = float(p("far_max").value)
        self.far_terminate_base = float(p("far_terminate").value)
        self.min_steps_before_far = int(p("min_steps_before_far").value)
        self.far_margin = float(p("far_margin").value)
        self.far_terminate_override = float(p("far_terminate_override").value)

        self.escape_enable = bool(p("escape_enable").value)
        self.escape_sector_deg = float(p("escape_sector_deg").value)
        self.escape_narrow_dist = float(p("escape_narrow_dist").value)
        self.escape_goal_bearing = math.radians(float(p("escape_goal_bearing_deg").value))
        self.escape_forward_penalty = float(p("escape_forward_penalty").value)

        self.curr_window = int(p("curriculum_window").value)
        self.unlock_medium = float(p("unlock_medium_success").value)
        self.unlock_hard = float(p("unlock_hard_success").value)
        self.mix_easy_prob = float(p("curriculum_mix_easy_prob").value)

        self.lr_decay_every = int(p("lr_decay_every_episodes").value)
        self.lr_decay_factor = float(p("lr_decay_factor").value)
        self.success_replay_boost = int(p("success_replay_boost").value)

        self.save_every = int(p("save_every_episodes").value)
        self.keep_last_n = int(p("keep_last_n_checkpoints").value)
        self.print_every_steps = int(p("print_every_steps").value)
        self.control_dt = float(p("control_dt").value)
        self.reset_timeout_sec = float(p("reset_timeout_sec").value)

        # Paths
        cwd = os.getcwd()
        self.save_dir = (
            os.path.join(cwd, "src", "dqn_stage_nav_torch", "models")
            if os.path.isdir(os.path.join(cwd, "src"))
            else os.path.expanduser("~/dqn_models")
        )
        os.makedirs(self.save_dir, exist_ok=True)
        self.path_best = os.path.join(self.save_dir, "best_model.pth")
        self.path_last = os.path.join(self.save_dir, "last_model.pth")
        self.csv_path = os.path.join(self.save_dir, "training_log.csv")

        # Actions
        self.actions = build_action_set(self.v_max, self.w_max)
        self.action_size = len(self.actions)
        self.idx_rot = np.array([i for i, a in enumerate(self.actions) if a.name.startswith("ROT")], dtype=np.int64)
        self.idx_fast = np.array([i for i, a in enumerate(self.actions) if a.name == "FWD_F"], dtype=np.int64)
        self.idx_forwardish = np.array([i for i, a in enumerate(self.actions) if a.name in ("FWD_S", "FWD_F", "ARC_L", "ARC_R")], dtype=np.int64)

        # State processor
        sp_cfg = StateProcessorConfig(
            n_lidar_bins=self.lidar_bins,
            max_goal_dist=15.0,
            range_max_fallback=self.lidar_max,
            fov_deg=self.lidar_fov_deg,
            front_sector_deg=self.front_sector_deg,
        )
        self.state_proc = StateProcessor(sp_cfg)

        # Agent config 
        agent_cfg = TorchDQNConfig(
            n_lidar_bins=self.lidar_bins,
            aux_dim=5,
            action_size=self.action_size,
            gamma=float(p("agent_gamma").value),
            lr=float(p("agent_lr").value),
            batch_size=int(p("agent_batch_size").value),
            memory_size=int(p("agent_memory_size").value),
            min_memory_to_train=int(p("agent_min_memory").value),
            epsilon=float(p("agent_epsilon_start").value),
            epsilon_min=float(p("agent_epsilon_min").value),
            epsilon_decay=float(p("agent_epsilon_decay").value),
            soft_tau=float(p("agent_soft_tau").value),
            hard_update_every=int(p("agent_hard_update_every").value),
            max_grad_norm=float(p("agent_grad_norm").value),
            n_step=int(p("agent_n_step").value),
            use_per=bool(p("agent_use_per").value),
            per_alpha=float(p("agent_per_alpha").value),
            per_beta_start=float(p("agent_per_beta_start").value),
            per_beta_frames=int(p("agent_per_beta_frames").value),
            per_eps=float(p("agent_per_eps").value),
            per_prio_clip=float(p("agent_per_prio_clip").value),
            device=str(p("agent_device").value),
            seed=int(p("agent_seed").value),
        )
        self.agent = TorchDQNAgent(agent_cfg)

        loaded, ep = self.agent.load_checkpoint(self.path_last)
        self.episode_idx = int(ep) if (loaded and ep is not None) else 0

        # QoS sensors (x30)
        sensor_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
        )
        self.sub_scan = self.create_subscription(LaserScan, self.scan_topic, self._scan_cb, sensor_qos)
        self.sub_odom_sim = self.create_subscription(Odometry, self.odom_topic, self._odom_sim_cb, sensor_qos)
        self.sub_odom_raw = self.create_subscription(Odometry, self.raw_odom_topic, self._odom_raw_cb, sensor_qos)

        self.pub_cmd = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.reset_client = self.create_client(Empty, self.reset_service)

        # Latest messages / counters
        self.scan_msg: Optional[LaserScan] = None
        self.odom_sim_msg: Optional[Odometry] = None
        self.odom_raw_msg: Optional[Odometry] = None
        self.scan_count = 0
        self.odom_sim_count = 0
        self.odom_raw_count = 0

        self._last_processed_scan_count = 0
        self._last_processed_odom_sim_count = 0
        self._last_processed_odom_raw_count = 0

        # Curriculum
        self.curriculum = build_curriculum_by_difficulty(_GOALS_WITH_DIFF)
        self.goals_by_diff = {"easy": [], "medium": [], "hard": []}
        for xy, d in self.curriculum:
            self.goals_by_diff[d].append((xy, d))
        self.unlocked_level = 0
        self.goal_global = np.array(self.goals_by_diff["easy"][0][0], dtype=np.float32)
        self.curr_goal_label = "easy"
        self.last_goal_key: Optional[Tuple[float, float]] = None
        self.success_hist = {d: deque(maxlen=self.curr_window) for d in ("easy", "medium", "hard")}

        # Eval tracking
        self.eval_mode = False
        self.eval_hist = deque(maxlen=max(1, self.eval_window))
        self.last_eval_rate = 0.0
        self._best_ckpt_loaded = False

        # Episode state
        self.step_idx = 0
        self.waiting_reset = True
        self.reset_ts = time.monotonic()
        self.reset_scan_mark = 0
        self.reset_odom_sim_mark = 0
        self.reset_odom_raw_mark = 0

        self.last_state: Optional[np.ndarray] = None
        self.last_action_idx: Optional[int] = None
        self.last_dist = 0.0
        self.prev_v = 0.0
        self.prev_w = 0.0

        self.episode_return = 0.0
        self.loss_accum = 0.0
        self.loss_count = 0

        self.pos_hist = deque(maxlen=self.stuck_window)
        self.dist_hist = deque(maxlen=self.stuck_window)

        self.dist_start = 0.0
        self.dist_min = 1e9
        self.min_range_min = 1e9
        self.front_min_min = 1e9
        self.last_reward_step = 0.0

        self.far_terminate_ep = self.far_terminate_base

        self.comp_sum = {
            "r_step": 0.0, "r_progress": 0.0, "r_orient": 0.0, "r_obstacle": 0.0,
            "r_spin": 0.0, "r_clear": 0.0, "r_near": 0.0, "r_far": 0.0,
            "r_escape": 0.0, "r_terminal_override": 0.0,
        }
        self.comp_count = 0
        self.best_return = -1e9

        self._init_csv()
        self.create_timer(self.control_dt, self.control_loop)
        self._request_reset()

    # ---------- CSV ----------
    def _init_csv(self) -> None:
        new_file = (not os.path.exists(self.csv_path)) or (os.path.getsize(self.csv_path) == 0)
        self.csv_f = open(self.csv_path, "a", newline="")
        self.csv_w = csv.DictWriter(
            self.csv_f,
            fieldnames=[
                "episode","mode","goal_x","goal_y","goal_label","reason","success",
                "return","steps","dist_start","dist_end","dist_min","ang_end",
                "min_range_end","min_range_min","front_min_end","front_min_min",
                "last_reward","mean_reward","avg_loss","epsilon","buffer_size","train_steps",
                "c_step","c_progress","c_orient","c_obstacle","c_spin","c_clear","c_near","c_far","c_escape","c_term_override",
                "far_terminate_ep",
            ],
        )
        if new_file:
            self.csv_w.writeheader()
            self.csv_f.flush()

    # ---------- Callbacks ----------
    def _scan_cb(self, msg: LaserScan) -> None:
        self.scan_msg = msg
        self.scan_count += 1

    def _odom_sim_cb(self, msg: Odometry) -> None:
        self.odom_sim_msg = msg
        self.odom_sim_count += 1

    def _odom_raw_cb(self, msg: Odometry) -> None:
        self.odom_raw_msg = msg
        self.odom_raw_count += 1

    # ---------- Pose ----------
    def _pose_from_odom(self, msg: Odometry) -> Tuple[float, float, float]:
        p = msg.pose.pose.position
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        return float(p.x), float(p.y), float(yaw)

    def _pose_global(self) -> Tuple[float, float, float]:
        if self.use_raw_odom_for_global and self.odom_raw_msg is not None:
            return self._pose_from_odom(self.odom_raw_msg)
        if self.odom_sim_msg is None:
            return 0.0, 0.0, 0.0
        return self._pose_from_odom(self.odom_sim_msg)

    def _publish_twist(self, v: float, w: float) -> None:
        t = Twist()
        t.linear.x = float(v)
        t.angular.z = float(w)
        self.pub_cmd.publish(t)

    # ---------- Curriculum ----------
    def _rate(self, d: str) -> float:
        h = self.success_hist[d]
        return float(sum(h) / len(h)) if len(h) > 0 else 0.0

    def _maybe_unlock(self) -> None:
        if self.unlocked_level < 1 and self._rate("easy") >= self.unlock_medium:
            self.unlocked_level = 1
            self.get_logger().info("Curriculum unlock -> medium enabled (monotonic)")
        if self.unlocked_level < 2 and self._rate("medium") >= self.unlock_hard:
            self.unlocked_level = 2
            self.get_logger().info("Curriculum unlock -> hard enabled (monotonic)")

    def _pool_for_stage(self) -> List[Tuple[Tuple[float, float], str]]:
        if self.unlocked_level <= 0:
            return list(self.goals_by_diff["easy"])
        if self.unlocked_level == 1:
            return list(self.goals_by_diff["medium"])
        return list(self.goals_by_diff["hard"])

    def _should_eval_episode(self) -> bool:
        if self.eval_every <= 0:
            return False
        return (self.episode_idx % self.eval_every) == 0

    def _set_goal_for_episode(self) -> None:
        self._maybe_unlock()

        self.eval_mode = self._should_eval_episode()
        if self.eval_mode:
            pool = self._pool_for_stage()
        else:
            if self.unlocked_level >= 1 and random.random() < max(0.0, min(1.0, self.mix_easy_prob)):
                pool = list(self.goals_by_diff["easy"])
            else:
                pool = self._pool_for_stage()

        if not pool:
            pool = self.curriculum

        goal_idx = (self.episode_idx // max(1, self.episodes_per_goal)) % len(pool)
        coords, label = pool[goal_idx]

        if self.last_goal_key != coords:
            self.agent.boost_exploration(self.eps_boost)
            self.last_goal_key = coords

        self.goal_global = np.array(coords, dtype=np.float32)
        self.curr_goal_label = label

        mode = "EVAL" if self.eval_mode else "TRAIN"
        self.get_logger().info(
            f"EP {self.episode_idx} INIT [{mode}] | Goal: {coords} [{label}] | eps={self.agent.epsilon:.3f} | unlocked={self.unlocked_level}"
        )

    # ---------- Reset ----------
    def _request_reset(self) -> None:
        self._publish_twist(0.0, 0.0)

        if not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Reset service not available")
            return

        self.reset_client.call_async(Empty.Request())
        self.waiting_reset = True
        self.reset_ts = time.monotonic()
        self.reset_scan_mark = self.scan_count
        self.reset_odom_sim_mark = self.odom_sim_count
        self.reset_odom_raw_mark = self.odom_raw_count

        self._set_goal_for_episode()

    def _reset_episode_state(self) -> None:
        self.step_idx = 0
        self.episode_return = 0.0
        self.loss_accum = 0.0
        self.loss_count = 0
        self.last_state = None
        self.last_action_idx = None
        self.prev_v = 0.0
        self.prev_w = 0.0
        self.pos_hist.clear()
        self.dist_hist.clear()
        self.dist_min = 1e9
        self.min_range_min = 1e9
        self.front_min_min = 1e9
        self.last_reward_step = 0.0
        for k in self.comp_sum:
            self.comp_sum[k] = 0.0
        self.comp_count = 0

        gx, gy, _ = self._pose_global()
        dist0 = math.hypot(float(self.goal_global[0]) - gx, float(self.goal_global[1]) - gy)
        self.last_dist = dist0
        self.dist_start = dist0
        self.dist_min = min(self.dist_min, dist0)

        if self.far_terminate_override > 0.0:
            self.far_terminate_ep = self.far_terminate_override
        else:
            self.far_terminate_ep = max(self.far_terminate_base, dist0 + self.far_margin)

    # ---------- Sector min helpers (escape shaping) ----------
    def _sector_min(self, ranges: np.ndarray, angle_min: float, angle_max: float, center: float, half_width: float) -> float:
        if ranges.size == 0:
            return float(self.lidar_max)
        total = float(angle_max - angle_min)
        if total <= 1e-9:
            return float(np.min(ranges))
        a0 = center - half_width
        a1 = center + half_width
        n = int(ranges.size)
        i0 = int(np.clip((a0 - angle_min) / total * n, 0, n))
        i1 = int(np.clip((a1 - angle_min) / total * n, 0, n))
        if i1 <= i0:
            return float(np.min(ranges))
        return float(np.min(ranges[i0:i1]))

    # ---------- Reward helpers ----------
    def _near_goal_bonus(self, dist: float) -> float:
        if dist > self.near_goal_radius:
            return 0.0
        near_max = self.goal_reward * self.near_goal_max_frac
        return float(near_max * math.exp(-dist / max(1e-6, self.near_goal_tau)))

    def _far_penalty(self, dist: float) -> float:
        if dist <= self.far_start:
            return 0.0
        x = (dist - self.far_start) / max(1e-6, self.far_tau)
        return float(-self.far_max * (1.0 - math.exp(-x)))

    def _progress_gate(self, front_min: float) -> float:
        a = float(self.front_gate_safe)
        b = float(self.front_gate_open)
        if b <= a:
            return 1.0
        if front_min <= a:
            return 0.0
        if front_min >= b:
            return 1.0
        return float((front_min - a) / (b - a))

    def _action_mask(self, front_min: float) -> Optional[List[bool]]:
        if front_min < self.mask_stop_dist:
            mask = [False] * self.action_size
            for i in self.idx_rot.tolist():
                mask[int(i)] = True
            return mask
        if front_min < self.mask_slow_dist and len(self.idx_fast) > 0:
            mask = [True] * self.action_size
            mask[int(self.idx_fast[0])] = False
            return mask
        return None

    def _obstacle_penalty(self, min_range: float) -> float:
        if min_range >= self.obstacle_near_dist:
            return 0.0
        x = (self.obstacle_near_dist - float(min_range)) / max(1e-6, self.obstacle_near_dist)
        x = max(0.0, min(1.0, x))
        return -float(self.obstacle_near_scale) * (x ** float(self.obstacle_power))

    def _escape_penalty(self, ang_err: float, left_min: float, right_min: float, a_idx: int) -> float:
        if not self.escape_enable:
            return 0.0

        narrow = (left_min < self.escape_narrow_dist) and (right_min < self.escape_narrow_dist)
        if not narrow:
            return 0.0
        if abs(ang_err) < self.escape_goal_bearing:
            return 0.0

        a = self.actions[a_idx]
        if ang_err > 0.0 and a.name in ("ARC_R", "FWD_S", "FWD_F"):
            return -self.escape_forward_penalty

        if ang_err < 0.0 and a.name in ("ARC_L", "FWD_S", "FWD_F"):
            return -self.escape_forward_penalty
        return 0.0

    def _compute_reward_and_components(
        self,
        last_dist: float,
        dist: float,
        ang_err: float,
        min_range: float,
        front_min: float,
        left_min: float,
        right_min: float,
        prev_action: Action,
        prev_action_idx: int,
        done: bool,
        reason: str,
    ) -> Tuple[float, Dict[str, float]]:
        comp = {k: 0.0 for k in self.comp_sum.keys()}
        comp["r_step"] = self.step_penalty
        gate = self._progress_gate(front_min)
        comp["r_progress"] = gate * (last_dist - dist) * self.progress_scale
        comp["r_orient"] = self.orient_scale * math.cos(ang_err)
        comp["r_obstacle"] = self._obstacle_penalty(min_range)
        if abs(prev_action.v) < 0.05 and abs(prev_action.w) > 0.1:
            comp["r_spin"] = -self.spin_penalty
        comp["r_clear"] = self.clearance_scale * (float(min_range) / max(1e-6, self.lidar_max))
        comp["r_near"] = self._near_goal_bonus(dist)
        comp["r_far"] = self._far_penalty(dist)
        comp["r_escape"] = self._escape_penalty(ang_err, left_min, right_min, prev_action_idx)

        r = float(sum(comp.values()))

        if done:
            base_r = r
            if reason == "success":
                r = self.goal_reward
            elif reason == "collision":
                r = self.collision_penalty
            elif reason == "timeout":
                r = r + self.timeout_extra_penalty
            elif reason == "stuck":
                r = r + self.stuck_extra_penalty
            elif reason == "far":
                r = self.collision_penalty
            comp["r_terminal_override"] = r - base_r
        return float(r), comp

    # ---------- Stuck ----------
    def _is_stuck(self) -> bool:
        if len(self.pos_hist) < self.pos_hist.maxlen:
            return False
        x0, y0 = self.pos_hist[0]
        x1, y1 = self.pos_hist[-1]
        moved = math.hypot(x1 - x0, y1 - y0)
        d0 = self.dist_hist[0]
        d1 = self.dist_hist[-1]
        progress = (d0 - d1)
        return (moved < self.stuck_min_move) and (progress < self.stuck_min_progress)

    # ---------- Training stability hooks ----------
    def _maybe_decay_lr(self) -> None:
        if self.lr_decay_every <= 0:
            return
        if self.episode_idx > 0 and (self.episode_idx % self.lr_decay_every) == 0:
            try:
                for g in self.agent.opt.param_groups:
                    g["lr"] = max(1e-6, float(g["lr"]) * float(self.lr_decay_factor))
                lr_now = float(self.agent.opt.param_groups[0]["lr"])
                self.get_logger().info(f"LR decayed -> {lr_now:.6g}")
            except Exception:
                self.get_logger().warn("Could not decay LR (agent.opt not accessible)")

    # ---------- Saving ----------
    def _save_checkpoints(self, episode: int, episode_return: float) -> None:
        self.agent.save_checkpoint(self.path_last, episode, extra={"return": float(episode_return)})

        if episode_return > self.best_return and not self.eval_mode:
            self.best_return = episode_return
            self.agent.save_checkpoint(self.path_best, episode, extra={"return": float(episode_return)})

        if (episode % self.save_every) != 0:
            return
        ckpt_path = os.path.join(self.save_dir, f"ckpt_ep{episode:04d}.pth")
        self.agent.save_checkpoint(ckpt_path, episode, extra={"return": float(episode_return)})

        files = [f for f in os.listdir(self.save_dir) if f.startswith("ckpt_ep") and f.endswith(".pth")]
        if len(files) <= self.keep_last_n:
            return
        files.sort()
        for f in files[:-self.keep_last_n]:
            try:
                os.remove(os.path.join(self.save_dir, f))
            except Exception:
                pass

    # ---------- Main loop ----------
    def _has_new_step_data(self) -> bool:
        if self.scan_msg is None or self.odom_sim_msg is None:
            return False
        if self.scan_count == self._last_processed_scan_count:
            return False
        if self.odom_sim_count == self._last_processed_odom_sim_count:
            return False
        if self.use_raw_odom_for_global and self.odom_raw_msg is not None:
            if self.odom_raw_count == self._last_processed_odom_raw_count:
                return False
        return True

    def control_loop(self) -> None:
        try:
            if self.scan_msg is None or self.odom_sim_msg is None:
                return

            if self.waiting_reset:
                new_data = (self.scan_count > self.reset_scan_mark) and (self.odom_sim_count > self.reset_odom_sim_mark)
                if self.use_raw_odom_for_global:
                    new_data = new_data and (self.odom_raw_count > self.reset_odom_raw_mark)

                if new_data:
                    self.waiting_reset = False
                    self._reset_episode_state()
                    self._last_processed_scan_count = self.scan_count
                    self._last_processed_odom_sim_count = self.odom_sim_count
                    self._last_processed_odom_raw_count = self.odom_raw_count
                elif (time.monotonic() - self.reset_ts) > self.reset_timeout_sec:
                    self.get_logger().warn("Reset timed out, retrying...")
                    self._request_reset()
                return

            if not self._has_new_step_data():
                return

            self._last_processed_scan_count = self.scan_count
            self._last_processed_odom_sim_count = self.odom_sim_count
            self._last_processed_odom_raw_count = self.odom_raw_count

            gx, gy, gyaw = self._pose_global()
            scan = self.scan_msg

            ranges = np.asarray(scan.ranges, dtype=np.float32)
            ranges = np.nan_to_num(ranges, nan=self.lidar_max, posinf=self.lidar_max, neginf=self.lidar_max)
            ranges = np.clip(ranges, 0.0, self.lidar_max)

            angle_min = float(getattr(scan, "angle_min", 0.0))
            angle_max = float(getattr(scan, "angle_max", 0.0))

            lidar_bins_norm, min_range = self.state_proc.bin_lidar_min(
                ranges=ranges.tolist(), range_max=self.lidar_max, angle_min=angle_min, angle_max=angle_max
            )
            front_min = self.state_proc.compute_front_min(
                ranges=ranges.tolist(), range_max=self.lidar_max, angle_min=angle_min, angle_max=angle_max, front_sector_deg=self.front_sector_deg
            )

            # laterales para escape shaping
            half = math.radians(max(1.0, self.escape_sector_deg) / 2.0)
            left_min = self._sector_min(ranges, angle_min, angle_max, center=+math.pi/2, half_width=half)
            right_min = self._sector_min(ranges, angle_min, angle_max, center=-math.pi/2, half_width=half)

            dist = math.hypot(float(self.goal_global[0]) - gx, float(self.goal_global[1]) - gy)
            ang_err = angle_to_goal(gx, gy, gyaw, float(self.goal_global[0]), float(self.goal_global[1]))

            self.dist_min = min(self.dist_min, dist)
            self.min_range_min = min(self.min_range_min, float(min_range))
            self.front_min_min = min(self.front_min_min, float(front_min))

            state = self.state_proc.build_state(
                lidar_bins_norm=lidar_bins_norm,
                pos_g=(gx, gy),
                yaw_g=gyaw,
                goal_g=(float(self.goal_global[0]), float(self.goal_global[1])),
                v_prev=self.prev_v,
                w_prev=self.prev_w,
                v_max=self.v_max,
                w_max=self.w_max,
                min_range=min_range,
                range_max=self.lidar_max,
            )

            self.pos_hist.append((gx, gy))
            self.dist_hist.append(dist)

            done = False
            reason = ""

            if dist < self.goal_tol:
                done = True
                reason = "success"
            elif min_range < self.collision_range:
                done = True
                reason = "collision"
            elif self.step_idx >= self.max_steps:
                done = True
                reason = "timeout"
            elif self._is_stuck() and self.step_idx > max(30, self.stuck_window):
                done = True
                reason = "stuck"
            elif (self.step_idx >= self.min_steps_before_far) and (dist >= self.far_terminate_ep):
                done = True
                reason = "far"

            # Choosing Action (eval => training=False) + action_mask
            if not done:
                mask = self._action_mask(front_min)
                training_flag = (not self.eval_mode)
                a_idx = int(self.agent.act(state, training=training_flag, action_mask=mask))
                action = self.actions[a_idx]
                self._publish_twist(action.v, action.w)
            else:
                a_idx = None
                action = None
                self._publish_twist(0.0, 0.0)

            # Learning
            if (not self.eval_mode) and self.last_state is not None and self.last_action_idx is not None:
                prev_action = self.actions[self.last_action_idx]
                r, comp = self._compute_reward_and_components(
                    last_dist=self.last_dist,
                    dist=dist,
                    ang_err=ang_err,
                    min_range=min_range,
                    front_min=front_min,
                    left_min=left_min,
                    right_min=right_min,
                    prev_action=prev_action,
                    prev_action_idx=self.last_action_idx,
                    done=done,
                    reason=reason,
                )
                self.last_reward_step = r
                self.agent.remember(self.last_state, self.last_action_idx, r, state, done)

                loss = self.agent.train_step()
                if loss > 0.0:
                    self.loss_accum += loss
                    self.loss_count += 1

                self.episode_return += r
                for k in self.comp_sum:
                    self.comp_sum[k] += float(comp.get(k, 0.0))
                self.comp_count += 1

            # Step
            if not done and action is not None and a_idx is not None:
                self.last_state = state
                self.last_action_idx = int(a_idx)
                self.last_dist = dist
                self.prev_v = action.v
                self.prev_w = action.w
                self.step_idx += 1

                if self.print_every_steps > 0 and (self.step_idx % self.print_every_steps) == 0:
                    mode = "EVAL" if self.eval_mode else "TRAIN"
                    self.get_logger().info(
                        f"{mode} Step {self.step_idx:03d} | Dist={dist:5.2f} | Ang={ang_err:5.2f} | "
                        f"MinR={float(min_range):4.2f} | Front={float(front_min):4.2f} | "
                        f"L={left_min:4.2f} R={right_min:4.2f} | A={self.actions[self.last_action_idx].name} | "
                        f"r={self.last_reward_step:7.2f} | Ret={self.episode_return:8.1f} | Eps={self.agent.epsilon:5.3f}"
                    )
            else:
                self._finish_episode(reason, gx, gy, gyaw, dist, ang_err, float(min_range), float(front_min))

        except Exception:
            self.get_logger().error(traceback.format_exc())

    def _finish_episode(
        self,
        reason: str,
        gx: float,
        gy: float,
        gyaw: float,
        dist_end: float,
        ang_end: float,
        min_range_end: float,
        front_min_end: float,
    ) -> None:
        self._publish_twist(0.0, 0.0)

        avg_loss = (self.loss_accum / self.loss_count) if self.loss_count > 0 else 0.0
        stats = self.agent.get_stats()

        mean_reward = (self.episode_return / max(1, self.comp_count)) if self.comp_count > 0 else 0.0
        success = 1 if (reason == "success") else 0

        mode = "eval" if self.eval_mode else "train"
        if self.eval_mode:
            self.eval_hist.append(int(success))
            self.last_eval_rate = float(sum(self.eval_hist) / len(self.eval_hist))
            if self.rollback_on_eval_drop and len(self.eval_hist) >= max(3, self.eval_window):
                if self.last_eval_rate < self.eval_min_success_keep and os.path.exists(self.path_best):
                    ok, ep = self.agent.load_checkpoint(self.path_best)
                    if ok:
                        self.get_logger().warn(
                            f"EVAL rolling success={self.last_eval_rate:.2f} < {self.eval_min_success_keep:.2f}. "
                            f"Rollback -> best checkpoint (ep={ep})."
                        )
                        self._best_ckpt_loaded = True
        else:
            self.success_hist[self.curr_goal_label].append(int(success))
            self._maybe_decay_lr()

        def cavg(k: str) -> float:
            return (self.comp_sum[k] / max(1, self.comp_count)) if self.comp_count > 0 else 0.0

        self.csv_w.writerow({
            "episode": self.episode_idx,
            "mode": mode,
            "goal_x": float(self.goal_global[0]),
            "goal_y": float(self.goal_global[1]),
            "goal_label": self.curr_goal_label,
            "reason": reason,
            "success": int(success),
            "return": float(self.episode_return),
            "steps": int(self.step_idx),
            "dist_start": float(self.dist_start),
            "dist_end": float(dist_end),
            "dist_min": float(self.dist_min),
            "ang_end": float(ang_end),
            "min_range_end": float(min_range_end),
            "min_range_min": float(self.min_range_min),
            "front_min_end": float(front_min_end),
            "front_min_min": float(self.front_min_min),
            "last_reward": float(self.last_reward_step),
            "mean_reward": float(mean_reward),
            "avg_loss": float(avg_loss),
            "epsilon": float(stats.get("epsilon", 0.0)),
            "buffer_size": int(stats.get("buffer_size", 0)),
            "train_steps": int(stats.get("train_steps", 0)),
            "c_step": float(cavg("r_step")),
            "c_progress": float(cavg("r_progress")),
            "c_orient": float(cavg("r_orient")),
            "c_obstacle": float(cavg("r_obstacle")),
            "c_spin": float(cavg("r_spin")),
            "c_clear": float(cavg("r_clear")),
            "c_near": float(cavg("r_near")),
            "c_far": float(cavg("r_far")),
            "c_escape": float(cavg("r_escape")),
            "c_term_override": float(cavg("r_terminal_override")),
            "far_terminate_ep": float(self.far_terminate_ep),
        })
        self.csv_f.flush()

        if not self.eval_mode:
            self._save_checkpoints(self.episode_idx, self.episode_return)

        self.episode_idx += 1
        if self.episode_idx < self.episodes_total:
            self.waiting_reset = True
            self._request_reset()
        else:
            self.get_logger().info("Training Finished")
            try:
                self.csv_f.close()
            except Exception:
                pass
            if rclpy.ok():
                rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = DQNTrainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    main()