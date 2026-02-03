#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import time
import traceback
from dataclasses import dataclass
from typing import List, Tuple, Optional
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty

from dqn_stage_nav_torch.torch_dqn_agent import TorchDQNAgent, TorchDQNConfig
from dqn_stage_nav_torch.state_processor import StateProcessor, StateProcessorConfig


# --------------------- Math Helpers ---------------------
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


# --------------------- Action Set ---------------------
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


# --------------------- Node ---------------------
class DQNEvalNode(Node):
    def __init__(self):
        super().__init__("dqn_eval_node")

        # ---------------- Topics/Services ----------------
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

        # Lidar/State
        self.declare_parameter("lidar_bins", 20)
        self.declare_parameter("lidar_max_range", 4.5)
        self.declare_parameter("max_goal_dist", 15.0)

        # Actions/Dynamics
        self.declare_parameter("v_max", 1.2)
        self.declare_parameter("w_max", 1.8)

        # ---------------- Termination / Timing ----------------
        self.declare_parameter("goal_tolerance_enter", 0.45)
        self.declare_parameter("goal_tolerance_exit", 1.00)
        self.declare_parameter("goal_hold_steps", 8) 

        self.declare_parameter("collision_range", 0.25)
        self.declare_parameter("max_steps", 800)
        self.declare_parameter("control_dt", 0.10)

        # ---------------- Evaluation Plan ----------------
        self.declare_parameter("trials_per_goal", 1)
        self.declare_parameter(
            "eval_goals_xy",
            [
                -2.58, -5.73,  -2.77, -5.55,  -2.55, -4.90,   7.69,  3.27,
                 5.18, -2.66,   7.82, -3.14,   5.18,  5.75,   4.91, -1.86,
                 7.04, -2.44,   6.82,  3.49,   7.82,  4.79,   4.11,  5.75,
                 6.53,  4.93,   6.82,  5.66,   7.64,  5.66,   7.85,  4.79,
                 7.69,  4.00,   7.85,  3.80,   7.24,  5.08,   6.37,  5.41,
                 6.68,  5.08,   7.24,  4.93,   6.82,  4.64,   7.24,  3.64,
                 6.53,  3.49,   7.64,  3.49,   6.18,  3.80,   6.37,  3.64,
                 5.84, -2.81,   5.73, -3.42,   6.37, -3.14,   7.51, -3.38,
                 7.69, -3.55,   6.37, -3.64,   7.40, -3.55,   6.82, -3.27,
                 5.52, -3.38,   5.96, -3.14,   6.18, -3.27,   7.24, -3.27,
                 6.68, -3.42,   7.04, -3.64,   5.73, -2.97,   6.18, -2.97,
                 0.43, -5.96,   0.64, -6.11,  -5.89, -6.57,  -6.20, -6.11,
                -6.05, -6.73,  -5.56, -6.57,  -5.73, -6.11,  -4.94, -6.11,
                -4.88, -5.96,  -4.72, -6.25,  -4.09, -6.41,  -3.73, -6.11,
                -3.64, -5.96,  -3.42, -5.96,  -3.27, -6.11,  -5.18, -5.80,
                -4.39, -5.80,  -4.94, -5.80,  -6.05, -5.80,  -5.56, -5.96,
            ],
        )

        # ---------------- Stuck detection ----------------
        # Far from Goal
        self.declare_parameter("stuck_window_steps_far", 30)
        self.declare_parameter("stuck_move_eps_far", 0.08)

        # Close to Goal
        self.declare_parameter("stuck_window_steps_near", 60)
        self.declare_parameter("stuck_move_eps_near", 0.05)

        # Oscilation
        self.declare_parameter("osc_window_steps", 24)
        self.declare_parameter("osc_allow_only_rot", True)
        self.declare_parameter("osc_alt_ratio", 0.80) 
        self.declare_parameter("near_goal_max_steps_without_hold", 160)  

        # ---------------- Model ----------------
        self.declare_parameter(
            "model_path",
            os.path.expanduser("~/ros2_workspaces/diplomado_ws/src/dqn_stage_nav_torch/models/best_model.pth"),
        )
        self.declare_parameter("device", "cuda") 

        # ---------------- Logging/Outputs ----------------
        self.declare_parameter("print_every_steps", 25)
        self.declare_parameter("results_csv_path", "")  

        # ---------------- Read Params ----------------
        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.raw_odom_topic = str(self.get_parameter("raw_odom_topic").value)
        self.use_raw_odom_for_global = bool(self.get_parameter("use_raw_odom_for_global").value)
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.reset_service = str(self.get_parameter("reset_service").value)

        self.spawn_x_fallback = float(self.get_parameter("spawn_global_x").value)
        self.spawn_y_fallback = float(self.get_parameter("spawn_global_y").value)
        self.spawn_yaw_fallback = math.radians(float(self.get_parameter("spawn_global_yaw_deg").value))

        self.lidar_bins = int(self.get_parameter("lidar_bins").value)
        self.lidar_max = float(self.get_parameter("lidar_max_range").value)
        self.max_goal_dist = float(self.get_parameter("max_goal_dist").value)

        self.v_max = float(self.get_parameter("v_max").value)
        self.w_max = float(self.get_parameter("w_max").value)

        self.goal_tol_enter = float(self.get_parameter("goal_tolerance_enter").value)
        self.goal_tol_exit = float(self.get_parameter("goal_tolerance_exit").value)
        self.goal_hold_steps = int(self.get_parameter("goal_hold_steps").value)
        if self.goal_tol_exit < self.goal_tol_enter:
            self.goal_tol_exit = self.goal_tol_enter

        self.collision_range = float(self.get_parameter("collision_range").value)
        self.max_steps = int(self.get_parameter("max_steps").value)
        self.control_dt = float(self.get_parameter("control_dt").value)

        self.trials_per_goal = int(self.get_parameter("trials_per_goal").value)
        self.eval_goals = self._parse_goals(self.get_parameter("eval_goals_xy").value)
        if not self.eval_goals:
            raise RuntimeError("eval_goals_xy is empty or invalid (must be [x1,y1,x2,y2,...]).")

        self.stuck_window_far = int(self.get_parameter("stuck_window_steps_far").value)
        self.stuck_eps_far = float(self.get_parameter("stuck_move_eps_far").value)
        self.stuck_window_near = int(self.get_parameter("stuck_window_steps_near").value)
        self.stuck_eps_near = float(self.get_parameter("stuck_move_eps_near").value)

        self.osc_window_steps = int(self.get_parameter("osc_window_steps").value)
        self.osc_allow_only_rot = bool(self.get_parameter("osc_allow_only_rot").value)
        self.osc_alt_ratio = float(self.get_parameter("osc_alt_ratio").value)
        self.near_goal_max_steps_without_hold = int(self.get_parameter("near_goal_max_steps_without_hold").value)

        self.model_path = os.path.expanduser(str(self.get_parameter("model_path").value))
        self.device = str(self.get_parameter("device").value).strip()

        self.print_every_steps = int(self.get_parameter("print_every_steps").value)

        csv_param = str(self.get_parameter("results_csv_path").value).strip()
        if csv_param:
            self.results_csv_path = os.path.expanduser(csv_param)
        else:
            model_dir = os.path.dirname(os.path.abspath(self.model_path))
            self.results_csv_path = os.path.join(model_dir, "eval_results.csv")

        # ---------------- ROS I/O ----------------
        self.sub_scan = self.create_subscription(LaserScan, self.scan_topic, self._scan_cb, 10)
        self.sub_odom_sim = self.create_subscription(Odometry, self.odom_topic, self._odom_sim_cb, 10)
        self.sub_odom_raw = self.create_subscription(Odometry, self.raw_odom_topic, self._odom_raw_cb, 10)
        self.pub_cmd = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.reset_client = self.create_client(Empty, self.reset_service)

        # ---------------- Action Set + Agent/State ----------------
        self.actions = build_action_set(self.v_max, self.w_max)
        self.action_size = len(self.actions)

        sp_cfg = StateProcessorConfig(
            n_lidar_bins=self.lidar_bins,
            max_goal_dist=self.max_goal_dist,
            range_max_fallback=self.lidar_max,
        )
        self.state_proc = StateProcessor(sp_cfg)

        agent_cfg = TorchDQNConfig(
            n_lidar_bins=self.lidar_bins,
            aux_dim=5,
            action_size=self.action_size,
            epsilon=0.0,
            device=self.device,
        )
        self.agent = TorchDQNAgent(agent_cfg)

        if not os.path.exists(self.model_path):
            raise RuntimeError(f"Model not found: {self.model_path}")

        loaded, ep = self.agent.load_checkpoint(self.model_path)
        if not loaded:
            raise RuntimeError(f"Failed to load checkpoint: {self.model_path}")
        self.agent.epsilon = 0.0 

        # ---------------- Latest Messages ----------------
        self.scan_msg: Optional[LaserScan] = None
        self.odom_sim_msg: Optional[Odometry] = None
        self.odom_raw_msg: Optional[Odometry] = None

        # Spawn Calibration Fallback
        self.spawn_x = self.spawn_x_fallback
        self.spawn_y = self.spawn_y_fallback
        self.spawn_yaw = self.spawn_yaw_fallback

        # ---------------- Eval Bookkeeping ----------------
        self.total_episodes = len(self.eval_goals) * self.trials_per_goal
        self.goal_idx = 0
        self.trial_idx_for_goal = 0
        self.episode_idx = 0

        # Episode State
        self.waiting_reset = True
        self.reset_ts = time.monotonic()
        self.episode_start_ts = time.monotonic()

        self.step_idx = 0
        self.prev_v = 0.0
        self.prev_w = 0.0
        self.last_min_range = float("inf")
        self.last_dist = float("inf")
        self.best_dist = float("inf")

        # Goal Hold State
        self.goal_hold_count = 0
        self.near_goal_steps = 0  

        # Stuck Trackers
        self.pos_hist = deque(maxlen=max(2, self.stuck_window_near))
        self.action_name_hist = deque(maxlen=max(2, self.osc_window_steps))
        self.action_v_hist = deque(maxlen=max(2, self.osc_window_steps))

        # Results
        self.results = []
        self._init_csv()

        # Start
        self.current_goal = self._current_goal()
        self.get_logger().info(f"Loaded model: {self.model_path} (episode={ep})")
        self.get_logger().info(
            f"Eval plan: {len(self.eval_goals)} goals x {self.trials_per_goal} trials = {self.total_episodes} episodes"
        )
        self.get_logger().info(f"Saving results to: {self.results_csv_path}")
        self.get_logger().info(
            f"SUCCESS practical: enter={self.goal_tol_enter:.2f} exit={self.goal_tol_exit:.2f} hold_steps={self.goal_hold_steps} "
            f"| stuck_far: win={self.stuck_window_far} eps={self.stuck_eps_far:.2f} "
            f"| stuck_near: win={self.stuck_window_near} eps={self.stuck_eps_near:.2f} "
            f"| osc_win={self.osc_window_steps} alt_ratio={self.osc_alt_ratio:.2f}"
        )

        self._request_reset()
        self.timer = self.create_timer(self.control_dt, self.control_loop)

    # ---------- CSV ----------
    def _init_csv(self):
        os.makedirs(os.path.dirname(self.results_csv_path), exist_ok=True)
        if not os.path.exists(self.results_csv_path):
            with open(self.results_csv_path, "w", encoding="utf-8") as f:
                f.write(
                    "episode,goal_idx,trial_idx,goal_x,goal_y,outcome,steps,"
                    "dist_final,best_dist,success_dist_used,min_range_final,elapsed_s,stuck_reason,stamp\n"
                )

    def _append_csv(self, row: dict):
        with open(self.results_csv_path, "a", encoding="utf-8") as f:
            f.write(
                f"{row['episode']},{row['goal_idx']},{row['trial_idx']},"
                f"{row['goal_x']:.3f},{row['goal_y']:.3f},"
                f"{row['outcome']},{row['steps']},"
                f"{row['dist_final']:.3f},{row['best_dist']:.3f},{row['success_dist_used']:.3f},"
                f"{row['min_range_final']:.3f},{row['elapsed_s']:.3f},"
                f"{row['stuck_reason']},{row['stamp']}\n"
            )

    # ---------- Goals ----------
    def _parse_goals(self, goals_xy) -> List[Tuple[float, float]]:
        if goals_xy is None or not isinstance(goals_xy, (list, tuple)):
            return []
        vals = [float(v) for v in goals_xy]
        if len(vals) < 2:
            return []
        if len(vals) % 2 != 0:
            vals = vals[:-1]
        return [(vals[i], vals[i + 1]) for i in range(0, len(vals), 2)]

    def _current_goal(self) -> Tuple[float, float]:
        return self.eval_goals[self.goal_idx % len(self.eval_goals)]

    def _advance_episode_plan(self):
        self.trial_idx_for_goal += 1
        if self.trial_idx_for_goal >= self.trials_per_goal:
            self.trial_idx_for_goal = 0
            self.goal_idx += 1
        self.episode_idx += 1

    # ---------- Callbacks ----------
    def _scan_cb(self, msg: LaserScan):
        self.scan_msg = msg

    def _odom_sim_cb(self, msg: Odometry):
        self.odom_sim_msg = msg

    def _odom_raw_cb(self, msg: Odometry):
        self.odom_raw_msg = msg

    # ---------- Pose ----------
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

    # ---------- cmd ----------
    def _publish_twist(self, v: float, w: float):
        t = Twist()
        t.linear.x = float(v)
        t.angular.z = float(w)
        self.pub_cmd.publish(t)

    # ---------- Stuck Detection ----------
    def _effective_stuck_params(self, dist: float) -> Tuple[int, float]:
        # Near Goal
        if dist <= self.goal_tol_exit:
            return self.stuck_window_near, self.stuck_eps_near
        return self.stuck_window_far, self.stuck_eps_far

    def _is_stuck_by_motion(self, dist: float) -> bool:
        win, eps = self._effective_stuck_params(dist)

        if len(self.pos_hist) < win:
            return False

        x0, y0 = self.pos_hist[-win]
        x1, y1 = self.pos_hist[-1]
        moved = math.hypot(x1 - x0, y1 - y0)
        return moved < eps

    def _is_stuck_by_oscillation(self) -> bool:
        if not self.osc_allow_only_rot:
            return False
        if len(self.action_name_hist) < self.osc_window_steps:
            return False

        names = list(self.action_name_hist)[-self.osc_window_steps :]
        vs = list(self.action_v_hist)[-self.osc_window_steps :]

        if any(abs(v) > 1e-6 for v in vs):
            return False

        allowed = {"ROT_L", "ROT_R"}
        if any(n not in allowed for n in names):
            return False

        alt = 0
        for i in range(1, len(names)):
            if names[i] != names[i - 1]:
                alt += 1

        return alt >= int(self.osc_alt_ratio * (len(names) - 1))

    # ---------- Episode Finalize ----------
    def _finalize_episode(self, outcome: str, stuck_reason: str = ""):
        elapsed_s = time.monotonic() - self.episode_start_ts
        success_dist_used = self.goal_tol_exit

        row = {
            "episode": self.episode_idx,
            "goal_idx": self.goal_idx,
            "trial_idx": self.trial_idx_for_goal,
            "goal_x": float(self.current_goal[0]),
            "goal_y": float(self.current_goal[1]),
            "outcome": outcome,
            "steps": int(self.step_idx),
            "dist_final": float(self.last_dist),
            "best_dist": float(self.best_dist),
            "success_dist_used": float(success_dist_used),
            "min_range_final": float(self.last_min_range),
            "elapsed_s": float(elapsed_s),
            "stuck_reason": stuck_reason.replace(",", ";"), 
            "stamp": int(time.time()),
        }

        self.results.append(row)
        self._append_csv(row)

        self._publish_twist(0.0, 0.0)
        self._advance_episode_plan()

        if self.episode_idx >= self.total_episodes:
            self._print_summary()
            self.get_logger().info("[EVAL] Completed all episodes. Shutting down.")
            try:
                self.timer.cancel()
            except Exception:
                pass
            if rclpy.ok():
                rclpy.shutdown()
            return

        self.current_goal = self._current_goal()
        self._request_reset()

    def _print_summary(self):
        total = len(self.results)
        succ = sum(1 for r in self.results if r["outcome"] == "SUCCESS")
        col = sum(1 for r in self.results if r["outcome"] == "COLLISION")
        tout = sum(1 for r in self.results if r["outcome"] == "TIMEOUT")
        stk = sum(1 for r in self.results if r["outcome"] == "STUCK")

        def pct(n: int) -> float:
            return (100.0 * n / total) if total > 0 else 0.0

        self.get_logger().info(
            f"[SUMMARY] total={total} | SUCCESS={succ} ({pct(succ):.1f}%) | "
            f"COLLISION={col} ({pct(col):.1f}%) | TIMEOUT={tout} ({pct(tout):.1f}%) | STUCK={stk} ({pct(stk):.1f}%)"
        )

        for gi, g in enumerate(self.eval_goals):
            rows = [r for r in self.results if r["goal_idx"] == gi]
            if not rows:
                continue
            s = sum(1 for r in rows if r["outcome"] == "SUCCESS")
            self.get_logger().info(
                f"[GOAL {gi:02d}] goal=({g[0]:.2f},{g[1]:.2f}) trials={len(rows)} success={s} ({100.0*s/len(rows):.1f}%)"
            )

    # ---------- reset ----------
    def _request_reset(self):
        self._publish_twist(0.0, 0.0)

        self.waiting_reset = True
        self.reset_ts = time.monotonic()
        self.episode_start_ts = time.monotonic()

        self.step_idx = 0
        self.prev_v = 0.0
        self.prev_w = 0.0
        self.last_min_range = float("inf")
        self.last_dist = float("inf")
        self.best_dist = float("inf")

        self.goal_hold_count = 0
        self.near_goal_steps = 0

        self.pos_hist.clear()
        self.action_name_hist.clear()
        self.action_v_hist.clear()

        if self.reset_client.wait_for_service(1.0):
            self.reset_client.call_async(Empty.Request())
            self.get_logger().info(
                f"[EVAL] Reset requested | ep={self.episode_idx}/{self.total_episodes} "
                f"goal_idx={self.goal_idx} trial={self.trial_idx_for_goal} goal=({self.current_goal[0]:.2f},{self.current_goal[1]:.2f})"
            )
        else:
            self.get_logger().warn("[EVAL] Reset service not available (continuing without reset)")

    # ---------- Main Loop ----------
    def control_loop(self):
        try:
            if self.scan_msg is None or self.odom_sim_msg is None:
                return

            if self.waiting_reset:
                dt = time.monotonic() - self.reset_ts
                if dt > 2.5:
                    self.get_logger().warn("[EVAL] Reset timeout -> retry")
                    self._request_reset()
                    return
                if dt > 0.35:
                    self.waiting_reset = False
                    gx, gy, _ = self._pose_global()
                    dist0 = math.hypot(self.current_goal[0] - gx, self.current_goal[1] - gy)
                    self.get_logger().info(f"[EVAL] START pos=({gx:.2f},{gy:.2f}) dist0={dist0:.2f}")
                return

            gx, gy, gyaw = self._pose_global()
            self.pos_hist.append((gx, gy))

            ranges = np.array(self.scan_msg.ranges, dtype=np.float32)
            ranges = np.nan_to_num(ranges, nan=self.lidar_max, posinf=self.lidar_max, neginf=0.0)
            ranges = np.clip(ranges, 0.0, self.lidar_max)

            lidar_bins_norm, min_range = self.state_proc.bin_lidar_min(ranges.tolist(), self.lidar_max)

            dist = math.hypot(self.current_goal[0] - gx, self.current_goal[1] - gy)
            ang_err = angle_to_goal(gx, gy, gyaw, float(self.current_goal[0]), float(self.current_goal[1]))

            self.last_min_range = float(min_range)
            self.last_dist = float(dist)
            if dist < self.best_dist:
                self.best_dist = dist

            # -------- SUCCESS Practical with Hysteresis + Hold --------
            if dist <= self.goal_tol_exit:
                self.near_goal_steps += 1
                self.goal_hold_count += 2 if dist <= self.goal_tol_enter else 1
            else:
                self.near_goal_steps = 0
                self.goal_hold_count = 0

            if self.goal_hold_count >= self.goal_hold_steps:
                self.get_logger().info(
                    f"[EVAL] SUCCESS | ep={self.episode_idx} dist={dist:.2f} "
                    f"(enter={self.goal_tol_enter:.2f}, exit={self.goal_tol_exit:.2f}, hold={self.goal_hold_steps}) "
                    f"steps={self.step_idx} best={self.best_dist:.2f}"
                )
                self._finalize_episode("SUCCESS")
                return

            # Collision
            if min_range < self.collision_range:
                self.get_logger().warn(
                    f"[EVAL] COLLISION | ep={self.episode_idx} minR={min_range:.2f} steps={self.step_idx} best={self.best_dist:.2f}"
                )
                self._finalize_episode("COLLISION")
                return

            # Timeout
            if self.step_idx >= self.max_steps:
                self.get_logger().warn(
                    f"[EVAL] TIMEOUT | ep={self.episode_idx} dist={dist:.2f} steps={self.step_idx} best={self.best_dist:.2f}"
                )
                self._finalize_episode("TIMEOUT")
                return

            # Build State
            state = self.state_proc.build_state(
                lidar_bins_norm=lidar_bins_norm,
                pos_g=(gx, gy),
                yaw_g=gyaw,
                goal_g=(float(self.current_goal[0]), float(self.current_goal[1])),
                v_prev=self.prev_v,
                w_prev=self.prev_w,
                v_max=self.v_max,
                w_max=self.w_max,
                min_range=min_range,
                range_max=self.lidar_max,
            )

            # Act
            a_idx = self.agent.act(state, training=False)
            if not isinstance(a_idx, int):
                a_idx = int(a_idx)
            if a_idx < 0:
                a_idx = 0
            elif a_idx >= self.action_size:
                a_idx = self.action_size - 1

            act = self.actions[a_idx]
            self._publish_twist(act.v, act.w)

            self.prev_v = act.v
            self.prev_w = act.w

            self.action_name_hist.append(act.name)
            self.action_v_hist.append(act.v)

            self.step_idx += 1

            # -------- STUCK detection --------
            stuck_reason = ""
            if self._is_stuck_by_oscillation():
                stuck_reason = "oscillation_rot_only"
            elif self._is_stuck_by_motion(dist):
                win, eps = self._effective_stuck_params(dist)
                stuck_reason = f"no_progress(win={win},eps={eps:.3f})"

            if not stuck_reason and dist <= self.goal_tol_exit and self.near_goal_steps >= self.near_goal_max_steps_without_hold:
                stuck_reason = f"near_goal_no_hold(steps={self.near_goal_steps})"

            if stuck_reason:
                self.get_logger().warn(
                    f"[EVAL] STUCK | ep={self.episode_idx} dist={dist:.2f} steps={self.step_idx} "
                    f"best={self.best_dist:.2f} reason={stuck_reason}"
                )
                self._finalize_episode("STUCK", stuck_reason=stuck_reason)
                return

            if (self.step_idx % max(1, self.print_every_steps)) == 0:
                self.get_logger().info(
                    f"[EVAL] ep={self.episode_idx:03d} g={self.goal_idx} t={self.trial_idx_for_goal} step={self.step_idx:04d} "
                    f"pos=({gx:6.2f},{gy:6.2f}) dist={dist:5.2f} best={self.best_dist:5.2f} "
                    f"ang={ang_err:5.2f} minR={min_range:4.2f} "
                    f"A={act.name} v={act.v:.2f} w={act.w:.2f} hold={self.goal_hold_count}/{self.goal_hold_steps}"
                )

        except Exception:
            self.get_logger().error(traceback.format_exc())
            self._publish_twist(0.0, 0.0)


def main(args=None):
    rclpy.init(args=args)
    node = DQNEvalNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node._publish_twist(0.0, 0.0)
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
