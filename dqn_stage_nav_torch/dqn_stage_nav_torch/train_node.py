#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import os
import time
import traceback
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import rclpy
from rclpy.node import Node

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


def rot2d(x: float, y: float, yaw: float) -> Tuple[float, float]:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return (c * x - s * y, s * x + c * y)


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
    ((5.0, -2.0), "medium"),
    ((5.0, 4.0), "hard"),
    ((3.0, 6.0), "hard"),
    ((7.0, 3.0), "hard"),
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

        # Topics/Services
        self.declare_parameter("scan_topic", "/base_scan")
        self.declare_parameter("odom_topic", "/odom/sim")
        self.declare_parameter("raw_odom_topic", "/ground_truth")
        self.declare_parameter("use_raw_odom_for_global", True)
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("reset_service", "/reset_sim")

        # Spawn Fallback 
        self.declare_parameter("spawn_global_x", -7.0)
        self.declare_parameter("spawn_global_y", -7.0)
        self.declare_parameter("spawn_global_yaw_deg", 45.0)

        # Training Schedule
        self.declare_parameter("episodes_total", 4000)
        self.declare_parameter("episodes_per_goal", 25)
        self.declare_parameter("max_steps_per_episode", 500)

        # Lidar
        self.declare_parameter("lidar_bins", 20)
        self.declare_parameter("lidar_max_range", 4.5)

        # Dynamics / Actions
        self.declare_parameter("v_max", 1.2)
        self.declare_parameter("w_max", 1.8)

        # Safety / Termination
        self.declare_parameter("goal_tolerance", 0.40)
        self.declare_parameter("collision_range", 0.25)

        # Stuck Detection
        self.declare_parameter("stuck_window", 100)      
        self.declare_parameter("stuck_min_move", 0.15)    
        self.declare_parameter("stuck_min_progress", 0.10)

        # Exploration Curriculum Helper
        self.declare_parameter("epsilon_boost_on_goal_change", 0.35)

        # Reward Base
        self.declare_parameter("step_penalty", -0.02)
        self.declare_parameter("progress_scale", 3.0)
        self.declare_parameter("orient_scale", 0.15)
        self.declare_parameter("obstacle_near_dist", 0.60)
        self.declare_parameter("obstacle_near_scale", 2.0)
        self.declare_parameter("spin_penalty", 0.05)

        # Terminal Rewards
        self.declare_parameter("goal_reward", 160.0)
        self.declare_parameter("collision_penalty", -120.0)
        self.declare_parameter("timeout_extra_penalty", -25.0)
        self.declare_parameter("stuck_extra_penalty", -35.0)

        # Near-goal Exponential Bonus 
        self.declare_parameter("near_goal_radius", 1.0)
        self.declare_parameter("near_goal_tau", 0.8)
        self.declare_parameter("near_goal_max_frac", 0.3333333333)  

        # Far Exponential Penalty + Terminate
        self.declare_parameter("far_start", 10.0)
        self.declare_parameter("far_tau", 2.0)
        self.declare_parameter("far_max", 120.0)          
        self.declare_parameter("far_terminate", 15.0)    

        # Saving
        self.declare_parameter("save_every_episodes", 50)
        self.declare_parameter("keep_last_n_checkpoints", 8)

        # Rate
        self.declare_parameter("control_dt", 0.10)

        # ---- Read Params ----
        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.raw_odom_topic = str(self.get_parameter("raw_odom_topic").value)
        self.use_raw_odom_for_global = bool(self.get_parameter("use_raw_odom_for_global").value)
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.reset_service = str(self.get_parameter("reset_service").value)

        self.spawn_x_fallback = float(self.get_parameter("spawn_global_x").value)
        self.spawn_y_fallback = float(self.get_parameter("spawn_global_y").value)
        self.spawn_yaw_fallback = math.radians(float(self.get_parameter("spawn_global_yaw_deg").value))

        self.episodes_total = int(self.get_parameter("episodes_total").value)
        self.episodes_per_goal = int(self.get_parameter("episodes_per_goal").value)
        self.max_steps = int(self.get_parameter("max_steps_per_episode").value)

        self.lidar_bins = int(self.get_parameter("lidar_bins").value)
        self.lidar_max = float(self.get_parameter("lidar_max_range").value)

        self.v_max = float(self.get_parameter("v_max").value)
        self.w_max = float(self.get_parameter("w_max").value)

        self.goal_tol = float(self.get_parameter("goal_tolerance").value)
        self.collision_range = float(self.get_parameter("collision_range").value)

        self.stuck_window = int(self.get_parameter("stuck_window").value)
        self.stuck_min_move = float(self.get_parameter("stuck_min_move").value)
        self.stuck_min_progress = float(self.get_parameter("stuck_min_progress").value)

        self.eps_boost = float(self.get_parameter("epsilon_boost_on_goal_change").value)

        self.step_penalty = float(self.get_parameter("step_penalty").value)
        self.progress_scale = float(self.get_parameter("progress_scale").value)
        self.orient_scale = float(self.get_parameter("orient_scale").value)
        self.obstacle_near_dist = float(self.get_parameter("obstacle_near_dist").value)
        self.obstacle_near_scale = float(self.get_parameter("obstacle_near_scale").value)
        self.spin_penalty = float(self.get_parameter("spin_penalty").value)

        self.goal_reward = float(self.get_parameter("goal_reward").value)
        self.collision_penalty = float(self.get_parameter("collision_penalty").value)
        self.timeout_extra_penalty = float(self.get_parameter("timeout_extra_penalty").value)
        self.stuck_extra_penalty = float(self.get_parameter("stuck_extra_penalty").value)

        self.near_goal_radius = float(self.get_parameter("near_goal_radius").value)
        self.near_goal_tau = float(self.get_parameter("near_goal_tau").value)
        self.near_goal_max_frac = float(self.get_parameter("near_goal_max_frac").value)

        self.far_start = float(self.get_parameter("far_start").value)
        self.far_tau = float(self.get_parameter("far_tau").value)
        self.far_max = float(self.get_parameter("far_max").value)
        self.far_terminate = float(self.get_parameter("far_terminate").value)

        self.save_every = int(self.get_parameter("save_every_episodes").value)
        self.keep_last_n = int(self.get_parameter("keep_last_n_checkpoints").value)

        self.control_dt = float(self.get_parameter("control_dt").value)

        # Save Dir
        cwd = os.getcwd()
        if os.path.isdir(os.path.join(cwd, "src")):
            self.save_dir = os.path.join(cwd, "src", "dqn_stage_nav_torch", "models")
        else:
            self.save_dir = os.path.expanduser("~/dqn_models")

        os.makedirs(self.save_dir, exist_ok=True)
        self.path_best = os.path.join(self.save_dir, "best_model.pth")
        self.path_last = os.path.join(self.save_dir, "last_model.pth")
        self.csv_path = os.path.join(self.save_dir, "training_log.csv")

        self.get_logger().info(f"Saving models to: {self.save_dir}")
        self.get_logger().info(f"CSV: {self.csv_path}")

        # Actions
        self.actions = build_action_set(self.v_max, self.w_max)
        self.action_size = len(self.actions)

        # State Processor
        sp_cfg = StateProcessorConfig(
            n_lidar_bins=self.lidar_bins,
            max_goal_dist=15.0,
            range_max_fallback=self.lidar_max,
        )
        self.state_proc = StateProcessor(sp_cfg)

        # Agent
        agent_cfg = TorchDQNConfig(
            n_lidar_bins=self.lidar_bins,
            aux_dim=5,
            action_size=self.action_size,
            batch_size=256,
            device="cuda",
        )
        self.agent = TorchDQNAgent(agent_cfg)

        loaded, ep = self.agent.load_checkpoint(self.path_last)
        if loaded:
            self.get_logger().info(f"Resuming from last checkpoint at episode {ep} | eps={self.agent.epsilon:.3f}")

        # ROS interfaces
        self.sub_scan = self.create_subscription(LaserScan, self.scan_topic, self._scan_cb, 10)
        self.sub_odom_sim = self.create_subscription(Odometry, self.odom_topic, self._odom_sim_cb, 10)
        self.sub_odom_raw = self.create_subscription(Odometry, self.raw_odom_topic, self._odom_raw_cb, 10)

        self.pub_cmd = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.reset_client = self.create_client(Empty, self.reset_service)

        # Latest messages
        self.scan_msg: Optional[LaserScan] = None
        self.odom_sim_msg: Optional[Odometry] = None
        self.odom_raw_msg: Optional[Odometry] = None
        self.scan_count = 0
        self.odom_sim_count = 0
        self.odom_raw_count = 0

        # Curriculum
        self.curriculum = build_curriculum_by_difficulty(_GOALS_WITH_DIFF)
        self.curr_goal_idx = 0
        self.goal_hold_count = 0
        self.goal_global = np.array(self.curriculum[0][0], dtype=np.float32)
        self.curr_goal_label = self.curriculum[0][1]

        # Episode State
        self.episode_idx = int(ep) if loaded and ep is not None else 0
        self.step_idx = 0
        self.waiting_reset = True
        self.reset_ts = time.monotonic()
        self.reset_scan_mark = 0
        self.reset_odom_sim_mark = 0
        self.reset_odom_raw_mark = 0

        # Spawn Calibration
        self.spawn_x = self.spawn_x_fallback
        self.spawn_y = self.spawn_y_fallback
        self.spawn_yaw = self.spawn_yaw_fallback
        self.spawn_is_calibrated = False

        # RL Buffers
        self.last_state: Optional[np.ndarray] = None
        self.last_action_idx: Optional[int] = None
        self.last_dist = 0.0

        self.episode_return = 0.0
        self.loss_accum = 0.0
        self.loss_count = 0

        self.prev_v = 0.0
        self.prev_w = 0.0

        # Stuck Windows
        self.pos_hist = deque(maxlen=self.stuck_window)
        self.dist_hist = deque(maxlen=self.stuck_window)

        # Episode Metrics Extras
        self.dist_start = 0.0
        self.dist_min = 1e9
        self.min_range_min = 1e9
        self.last_reward_step = 0.0

        # Reward Component Accumulators
        self.comp_sum = {
            "r_step": 0.0,
            "r_progress": 0.0,
            "r_orient": 0.0,
            "r_obstacle": 0.0,
            "r_spin": 0.0,
            "r_near": 0.0,
            "r_far": 0.0,
            "r_terminal_override": 0.0,
        }
        self.comp_count = 0

        # Best Tracking
        self.best_return = -1e9

        # CSV
        self._init_csv()

        # Timer and Reset
        self.create_timer(self.control_dt, self.control_loop)
        self._request_reset()

    # ---------------- CSV ----------------
    def _init_csv(self):
        new_file = not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0
        self.csv_f = open(self.csv_path, "a", newline="")
        self.csv_w = csv.DictWriter(self.csv_f, fieldnames=[
            "episode", "goal_x", "goal_y", "goal_label", "reason", "success",
            "return", "steps",
            "dist_start", "dist_end", "dist_min",
            "ang_end",
            "min_range_end", "min_range_min",
            "last_reward", "mean_reward",
            "avg_loss", "epsilon", "buffer_size", "train_steps",
            "c_step", "c_progress", "c_orient", "c_obstacle", "c_spin", "c_near", "c_far", "c_term_override",
        ])
        if new_file:
            self.csv_w.writeheader()
            self.csv_f.flush()

    # ---------------- Callbacks ----------------
    def _scan_cb(self, msg: LaserScan):
        self.scan_msg = msg
        self.scan_count += 1

    def _odom_sim_cb(self, msg: Odometry):
        self.odom_sim_msg = msg
        self.odom_sim_count += 1

    def _odom_raw_cb(self, msg: Odometry):
        self.odom_raw_msg = msg
        self.odom_raw_count += 1

    # ---------------- Pose ----------------
    def _pose_from_odom(self, msg: Odometry) -> Tuple[float, float, float]:
        p = msg.pose.pose.position
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        return float(p.x), float(p.y), float(yaw)

    def _pose_sim(self) -> Tuple[float, float, float]:
        if self.odom_sim_msg is None:
            return 0.0, 0.0, 0.0
        return self._pose_from_odom(self.odom_sim_msg)

    def _pose_global(self) -> Tuple[float, float, float]:
        if self.use_raw_odom_for_global and self.odom_raw_msg is not None:
            return self._pose_from_odom(self.odom_raw_msg)

        sx, sy, syaw = self._pose_sim()
        dx, dy = rot2d(sx, sy, self.spawn_yaw)
        return self.spawn_x + dx, self.spawn_y + dy, wrap_pi(self.spawn_yaw + syaw)

    def _publish_twist(self, v: float, w: float):
        t = Twist()
        t.linear.x = float(v)
        t.angular.z = float(w)
        self.pub_cmd.publish(t)

    # ---------------- Curriculum ----------------
    def _update_goal(self):
        if self.goal_hold_count >= self.episodes_per_goal:
            self.curr_goal_idx = (self.curr_goal_idx + 1) % len(self.curriculum)
            self.goal_hold_count = 0
            self.agent.boost_exploration(self.eps_boost)

        coords, label = self.curriculum[self.curr_goal_idx]
        self.goal_global = np.array(coords, dtype=np.float32)
        self.curr_goal_label = label

        self.goal_hold_count += 1
        self.get_logger().info(f"EP {self.episode_idx} INIT | Goal: {coords} [{label}] | eps={self.agent.epsilon:.3f}")

    # ---------------- Reset ----------------
    def _request_reset(self):
        self._publish_twist(0.0, 0.0)

        if self.reset_client.wait_for_service(1.0):
            self.reset_client.call_async(Empty.Request())
            self.waiting_reset = True
            self.reset_ts = time.monotonic()
            self.reset_scan_mark = self.scan_count
            self.reset_odom_sim_mark = self.odom_sim_count
            self.reset_odom_raw_mark = self.odom_raw_count
            self.spawn_is_calibrated = False

            self._update_goal()
        else:
            self.get_logger().warn("Reset service not available")

    def _try_calibrate_spawn_from_raw(self):
        if self.spawn_is_calibrated:
            return
        if self.odom_raw_msg is None:
            return
        if self.odom_raw_count <= self.reset_odom_raw_mark:
            return
        self.spawn_x, self.spawn_y, self.spawn_yaw = self._pose_from_odom(self.odom_raw_msg)
        self.spawn_is_calibrated = True

    # ---------------- Reward shaping helpers ----------------
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

    def _compute_reward_and_components(
        self,
        last_dist: float,
        dist: float,
        ang_err: float,
        min_range: float,
        action: Action,
        done: bool,
        reason: str
    ) -> Tuple[float, Dict[str, float]]:
        comp = {k: 0.0 for k in self.comp_sum.keys()}

        # 1) time step penalty
        comp["r_step"] = self.step_penalty

        # 2) progress
        comp["r_progress"] = (last_dist - dist) * self.progress_scale

        # 3) orientation shaping 
        comp["r_orient"] = self.orient_scale * math.cos(ang_err)

        # 4) obstacle proximity
        if min_range < self.obstacle_near_dist:
            comp["r_obstacle"] = - (self.obstacle_near_dist - min_range) * self.obstacle_near_scale

        # 5) spinning penalty
        if abs(action.v) < 0.05 and abs(action.w) > 0.1:
            comp["r_spin"] = -self.spin_penalty

        # 6) near-goal bonus exponential
        comp["r_near"] = self._near_goal_bonus(dist)

        # 7) far penalty exponential
        comp["r_far"] = self._far_penalty(dist)

        r = sum(comp.values())

        # Terminal Override
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

    # ---------------- Stuck ----------------
    def _is_stuck(self) -> bool:
        if len(self.pos_hist) < self.pos_hist.maxlen:
            return False

        (x0, y0) = self.pos_hist[0]
        (x1, y1) = self.pos_hist[-1]
        moved = math.hypot(x1 - x0, y1 - y0)

        d0 = self.dist_hist[0]
        d1 = self.dist_hist[-1]
        progress = (d0 - d1)

        return (moved < self.stuck_min_move) or (progress < self.stuck_min_progress)

    # ---------------- Saving ----------------
    def _save_checkpoints(self, episode: int, episode_return: float):
        self.agent.save_checkpoint(self.path_last, episode, extra={"return": float(episode_return)})

        if episode_return > self.best_return:
            self.best_return = episode_return
            self.agent.save_checkpoint(self.path_best, episode, extra={"return": float(episode_return)})

        if (episode % self.save_every) == 0:
            ckpt_path = os.path.join(self.save_dir, f"ckpt_ep{episode:04d}.pth")
            self.agent.save_checkpoint(ckpt_path, episode, extra={"return": float(episode_return)})
            self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        files = [f for f in os.listdir(self.save_dir) if f.startswith("ckpt_ep") and f.endswith(".pth")]
        if len(files) <= self.keep_last_n:
            return
        files.sort()
        to_remove = files[:-self.keep_last_n]
        for f in to_remove:
            try:
                os.remove(os.path.join(self.save_dir, f))
            except Exception:
                pass

    # ---------------- Main loop ----------------
    def control_loop(self):
        try:
            if self.scan_msg is None or self.odom_sim_msg is None:
                return

            if self.waiting_reset:
                new_data = (self.scan_count > self.reset_scan_mark) and (self.odom_sim_count > self.reset_odom_sim_mark)

                if self.use_raw_odom_for_global:
                    self._try_calibrate_spawn_from_raw()

                if new_data:
                    self.waiting_reset = False
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
                    self.last_reward_step = 0.0
                    for k in self.comp_sum:
                        self.comp_sum[k] = 0.0
                    self.comp_count = 0

                    gx, gy, gyaw = self._pose_global()
                    dist0 = math.hypot(self.goal_global[0] - gx, self.goal_global[1] - gy)
                    self.last_dist = dist0
                    self.dist_start = dist0
                    self.dist_min = min(self.dist_min, dist0)

                    self.get_logger().info("Reset complete. Starting episode.")
                elif (time.monotonic() - self.reset_ts) > 2.5:
                    self.get_logger().warn("Reset timed out, retrying...")
                    self._request_reset()
                return

            # Pose and lidar
            gx, gy, gyaw = self._pose_global()

            ranges = np.array(self.scan_msg.ranges, dtype=np.float32)
            ranges = np.nan_to_num(ranges, posinf=self.lidar_max)
            ranges = np.clip(ranges, 0.0, self.lidar_max)

            lidar_bins_norm, min_range = self.state_proc.bin_lidar_min(ranges.tolist(), self.lidar_max)

            dist = math.hypot(self.goal_global[0] - gx, self.goal_global[1] - gy)
            ang_err = angle_to_goal(gx, gy, gyaw, float(self.goal_global[0]), float(self.goal_global[1]))

            self.dist_min = min(self.dist_min, dist)
            self.min_range_min = min(self.min_range_min, float(min_range))

            # Build state
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

            # Stuck windows
            self.pos_hist.append((gx, gy))
            self.dist_hist.append(dist)

            # Done Conditions
            done = False
            reason = ""

            if dist < self.goal_tol:
                done = True
                reason = "success"
            elif min_range < self.collision_range:
                done = True
                reason = "collision"
            elif dist >= self.far_terminate:
                done = True
                reason = "far"
            elif self.step_idx >= self.max_steps:
                done = True
                reason = "timeout"
            elif self._is_stuck() and self.step_idx > max(30, self.stuck_window):
                done = True
                reason = "stuck"

            # Select Action
            action = None
            a_idx = None
            if not done:
                a_idx = self.agent.act(state, training=True)
                action = self.actions[a_idx]
                self._publish_twist(action.v, action.w)
            else:
                self._publish_twist(0.0, 0.0)

            # RL transition: (last_state, last_action) -> state
            if self.last_state is not None and self.last_action_idx is not None:
                prev_action = self.actions[self.last_action_idx]

                r, comp = self._compute_reward_and_components(
                    last_dist=self.last_dist,
                    dist=dist,
                    ang_err=ang_err,
                    min_range=min_range,
                    action=prev_action,
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

            if not done and action is not None and a_idx is not None:
                self.last_state = state
                self.last_action_idx = a_idx
                self.last_dist = dist
                self.prev_v = action.v
                self.prev_w = action.w
                self.step_idx += 1

                if (self.step_idx % 50) == 0:
                    self.get_logger().info(
                        f"Step {self.step_idx:03d} | "
                        f"Global=({gx:6.2f},{gy:6.2f}) | "
                        f"Dist={dist:5.2f} | Ang={ang_err:5.2f} | MinR={min_range:4.2f} | "
                        f"A={self.actions[self.last_action_idx].name:5s} | "
                        f"r={self.last_reward_step:7.2f} | Ret={self.episode_return:8.1f} | "
                        f"Eps={self.agent.epsilon:5.3f}"
                    )
            else:
                self._finish_episode(reason, gx, gy, gyaw, dist, ang_err, min_range)

        except Exception:
            self.get_logger().error(traceback.format_exc())

    def _finish_episode(self, reason: str, gx: float, gy: float, gyaw: float, dist_end: float, ang_end: float, min_range_end: float):
        self._publish_twist(0.0, 0.0)

        avg_loss = (self.loss_accum / self.loss_count) if self.loss_count > 0 else 0.0
        stats = self.agent.get_stats()

        mean_reward = (self.episode_return / max(1, self.comp_count)) if self.comp_count > 0 else 0.0
        success = 1 if (reason == "success") else 0

        def cavg(k: str) -> float:
            return (self.comp_sum[k] / max(1, self.comp_count)) if self.comp_count > 0 else 0.0

        self.get_logger().info(
            f"EP {self.episode_idx} DONE ({reason}) | "
            f"Return={self.episode_return:7.1f} | meanR={mean_reward:6.2f} | "
            f"Loss={avg_loss:7.4f} | Pos=({gx:.2f},{gy:.2f}) | DistEnd={dist_end:.2f} | MinRmin={self.min_range_min:.2f}"
        )

        # CSV log
        self.csv_w.writerow({
            "episode": self.episode_idx,
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

            "last_reward": float(self.last_reward_step),
            "mean_reward": float(mean_reward),

            "avg_loss": float(avg_loss),
            "epsilon": float(stats["epsilon"]),
            "buffer_size": int(stats["buffer_size"]),
            "train_steps": int(stats["train_steps"]),

            "c_step": float(cavg("r_step")),
            "c_progress": float(cavg("r_progress")),
            "c_orient": float(cavg("r_orient")),
            "c_obstacle": float(cavg("r_obstacle")),
            "c_spin": float(cavg("r_spin")),
            "c_near": float(cavg("r_near")),
            "c_far": float(cavg("r_far")),
            "c_term_override": float(cavg("r_terminal_override")),
        })
        self.csv_f.flush()

        # Save Checkpoints
        self._save_checkpoints(self.episode_idx, self.episode_return)

        # Next
        self.episode_idx += 1
        if self.episode_idx < self.episodes_total:
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
