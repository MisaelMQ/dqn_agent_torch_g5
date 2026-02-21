#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import os
import time
import traceback
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple
from collections import deque

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


# --------------------- Math Helpers ---------------------
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


# --------------------- Goal CSV Loader ---------------------
def _try_float(v) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def load_goals_from_csv(path: str) -> List[Tuple[float, float]]:
    """Loads goals from a CSV with header. Accepts columns:
    - x,y
    - goal_x,goal_y
    - X,Y
    Otherwise falls back to first two numeric columns per row.
    """
    if not path or (not os.path.exists(path)):
        return []

    goals: List[Tuple[float, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = [h.strip() for h in (reader.fieldnames or [])]

        def pick(row: dict, key: str) -> Optional[float]:
            if key in row:
                return _try_float(row[key])
            return None

        for row in reader:
            x = pick(row, "x") or pick(row, "X") or pick(row, "goal_x") or pick(row, "gx")
            y = pick(row, "y") or pick(row, "Y") or pick(row, "goal_y") or pick(row, "gy")

            if x is None or y is None:
                # fallback: first two numeric fields
                vals = []
                for h in headers:
                    fv = _try_float(row.get(h, None))
                    if fv is not None:
                        vals.append(fv)
                    if len(vals) >= 2:
                        break
                if len(vals) >= 2:
                    x, y = vals[0], vals[1]

            if x is None or y is None:
                continue
            goals.append((float(x), float(y)))

    return goals


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

        # Lidar/State
        self.declare_parameter("lidar_bins", 20)
        self.declare_parameter("lidar_max_range", 4.5)
        self.declare_parameter("lidar_fov_deg", 270.0)
        self.declare_parameter("front_sector_deg", 30.0)
        self.declare_parameter("max_goal_dist", 15.0)

        # Actions/Dynamics
        self.declare_parameter("v_max", 1.25)
        self.declare_parameter("w_max", 2.00)

        # ---------------- Termination / Timing ----------------
        self.declare_parameter("goal_tolerance_enter", 0.45)
        self.declare_parameter("goal_tolerance_exit", 1.00)
        self.declare_parameter("goal_hold_steps", 8)

        self.declare_parameter("collision_range", 0.25)
        self.declare_parameter("max_steps", 800)
        self.declare_parameter("control_dt", 0.10)

        # ---------------- Evaluation Plan ----------------
        self.declare_parameter("trials_per_goal", 1)

        # Goal list from CSV
        self.declare_parameter("goals_csv_path", "")
        self.declare_parameter("shuffle_goals", False)
        self.declare_parameter("max_goals", 0)  # 0 = all

        # Legacy fallback (kept for compatibility)
        self.declare_parameter("eval_goals_xy", [])

        # ---------------- Stuck detection ----------------
        self.declare_parameter("stuck_window_steps_far", 30)
        self.declare_parameter("stuck_move_eps_far", 0.08)

        self.declare_parameter("stuck_window_steps_near", 60)
        self.declare_parameter("stuck_move_eps_near", 0.05)

        # Oscillation
        self.declare_parameter("osc_window_steps", 24)
        self.declare_parameter("osc_allow_only_rot", True)
        self.declare_parameter("osc_alt_ratio", 0.80)
        self.declare_parameter("near_goal_max_steps_without_hold", 160)

        # Action masking
        self.declare_parameter("mask_stop_dist", 0.28)
        self.declare_parameter("mask_slow_dist", 0.48)

        # ---------------- Model ----------------
        self.declare_parameter("model_path", "")
        self.declare_parameter("device", "cuda")

        # ---------------- Logging/Outputs ----------------
        self.declare_parameter("print_every_steps", 25)
        self.declare_parameter("results_csv_path", "")

        # ---------------- Read Params ----------------
        gp = self.get_parameter

        self.scan_topic = str(gp("scan_topic").value)
        self.odom_topic = str(gp("odom_topic").value)
        self.raw_odom_topic = str(gp("raw_odom_topic").value)
        self.use_raw_odom_for_global = bool(gp("use_raw_odom_for_global").value)
        self.cmd_vel_topic = str(gp("cmd_vel_topic").value)
        self.reset_service = str(gp("reset_service").value)

        self.lidar_bins = int(gp("lidar_bins").value)
        self.lidar_max = float(gp("lidar_max_range").value)
        self.lidar_fov_deg = float(gp("lidar_fov_deg").value)
        self.front_sector_deg = float(gp("front_sector_deg").value)
        self.max_goal_dist = float(gp("max_goal_dist").value)

        self.v_max = float(gp("v_max").value)
        self.w_max = float(gp("w_max").value)

        self.goal_tol_enter = float(gp("goal_tolerance_enter").value)
        self.goal_tol_exit = float(gp("goal_tolerance_exit").value)
        self.goal_hold_steps = int(gp("goal_hold_steps").value)
        if self.goal_tol_exit < self.goal_tol_enter:
            self.goal_tol_exit = self.goal_tol_enter

        self.collision_range = float(gp("collision_range").value)
        self.max_steps = int(gp("max_steps").value)
        self.control_dt = float(gp("control_dt").value)

        self.trials_per_goal = int(gp("trials_per_goal").value)

        self.goals_csv_path = os.path.expanduser(str(gp("goals_csv_path").value)).strip()
        self.shuffle_goals = bool(gp("shuffle_goals").value)
        self.max_goals = int(gp("max_goals").value)

        self.stuck_window_far = int(gp("stuck_window_steps_far").value)
        self.stuck_eps_far = float(gp("stuck_move_eps_far").value)
        self.stuck_window_near = int(gp("stuck_window_steps_near").value)
        self.stuck_eps_near = float(gp("stuck_move_eps_near").value)

        self.osc_window_steps = int(gp("osc_window_steps").value)
        self.osc_allow_only_rot = bool(gp("osc_allow_only_rot").value)
        self.osc_alt_ratio = float(gp("osc_alt_ratio").value)
        self.near_goal_max_steps_without_hold = int(gp("near_goal_max_steps_without_hold").value)

        self.mask_stop_dist = float(gp("mask_stop_dist").value)
        self.mask_slow_dist = float(gp("mask_slow_dist").value)

        model_path_param = str(gp("model_path").value).strip()
        self.device = str(gp("device").value).strip()

        # Paths => Default to ~/.../models/best_model.pth
        if model_path_param:
            self.model_path = os.path.expanduser(model_path_param)
        else:
            self.model_path = os.path.expanduser("~/ros2_workspaces/diplomado_ws/src/dqn_stage_nav_torch/models/last_model.pth")

        # Goals: Default CSV alongside model (models/distributed_points.csv)
        if not self.goals_csv_path:
            model_dir = os.path.dirname(os.path.abspath(self.model_path))
            self.goals_csv_path = os.path.join(model_dir, "distributed_points.csv")

        self.print_every_steps = int(gp("print_every_steps").value)

        csv_param = str(gp("results_csv_path").value).strip()
        if csv_param:
            self.results_csv_path = os.path.expanduser(csv_param)
        else:
            model_dir = os.path.dirname(os.path.abspath(self.model_path))
            self.results_csv_path = os.path.join(model_dir, "eval_results_last.csv")

        # ---------------- Load goals ----------------
        goals = load_goals_from_csv(self.goals_csv_path)
        if not goals:
            # fallback legacy list
            goals = self._parse_goals(gp("eval_goals_xy").value)

        if not goals:
            raise RuntimeError("No evaluation goals loaded (goals_csv_path invalid/empty AND eval_goals_xy empty).")

        if self.shuffle_goals:
            rng = np.random.default_rng(42)
            rng.shuffle(goals)

        if self.max_goals > 0:
            goals = goals[: self.max_goals]

        self.eval_goals: List[Tuple[float, float]] = goals

        # ---------------- ROS I/O ----------------
        # Sensor QoS: x10/x30 safe
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

        # ---------------- Action Set + Agent/State ----------------
        self.actions = build_action_set(self.v_max, self.w_max)
        self.action_size = len(self.actions)
        self.idx_rot = np.array([i for i, a in enumerate(self.actions) if a.name.startswith("ROT")], dtype=np.int64)
        self.idx_fast = np.array([i for i, a in enumerate(self.actions) if a.name == "FWD_F"], dtype=np.int64)

        sp_cfg = StateProcessorConfig(
            n_lidar_bins=self.lidar_bins,
            max_goal_dist=self.max_goal_dist,
            range_max_fallback=self.lidar_max,
            fov_deg=self.lidar_fov_deg,
            front_sector_deg=self.front_sector_deg,
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

        # ---------------- Latest Messages + counters ----------------
        self.scan_msg: Optional[LaserScan] = None
        self.odom_sim_msg: Optional[Odometry] = None
        self.odom_raw_msg: Optional[Odometry] = None
        self.scan_count = 0
        self.odom_sim_count = 0
        self.odom_raw_count = 0
        self._last_processed_scan_count = 0
        self._last_processed_odom_sim_count = 0
        self._last_processed_odom_raw_count = 0

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
        self.min_range_min = float("inf")
        self.front_min_end = float("inf")
        self.front_min_min = float("inf")

        self.last_dist = float("inf")
        self.best_dist = float("inf")
        self.dist0 = float("inf")

        self.path_len = 0.0
        self.last_pos: Optional[Tuple[float, float]] = None

        self.abs_w_sum = 0.0
        self.v_sum = 0.0
        self.step_count_for_means = 0

        # Goal Hold State
        self.goal_hold_count = 0
        self.near_goal_steps = 0
        self.time_to_success_s: Optional[float] = None

        # Stuck Trackers
        self.pos_hist: Deque[Tuple[float, float]] = deque(maxlen=max(2, self.stuck_window_near))
        self.action_name_hist: Deque[str] = deque(maxlen=max(2, self.osc_window_steps))
        self.action_v_hist: Deque[float] = deque(maxlen=max(2, self.osc_window_steps))

        # Results
        self.results: List[dict] = []
        self._init_csv()

        # Start
        self.current_goal = self._current_goal()
        self.get_logger().info(f"Loaded model: {self.model_path} (ckpt_ep={ep})")
        self.get_logger().info(
            f"Goals: {len(self.eval_goals)} (from {self.goals_csv_path}) x trials={self.trials_per_goal} => {self.total_episodes} episodes"
        )
        self.get_logger().info(f"Saving results to: {self.results_csv_path}")

        self._request_reset()
        self.timer = self.create_timer(self.control_dt, self.control_loop)

    # ---------- CSV ----------
    def _init_csv(self):
        os.makedirs(os.path.dirname(self.results_csv_path), exist_ok=True)
        if not os.path.exists(self.results_csv_path):
            with open(self.results_csv_path, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "episode","goal_idx","trial_idx","goal_x","goal_y",
                    "success","outcome","steps","elapsed_s","time_to_success_s",
                    "dist0","dist_final","best_dist","progress_best","ang_final",
                    "path_len","mean_v","mean_abs_w",
                    "min_range_end","min_range_min","front_min_end","front_min_min",
                    "stuck_reason","oscillation_flag",
                    "model_path","goals_csv_path","stamp",
                ])

    def _append_csv(self, row: dict):
        with open(self.results_csv_path, "a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                row["episode"], row["goal_idx"], row["trial_idx"], f"{row['goal_x']:.3f}", f"{row['goal_y']:.3f}",
                int(row["success"]), row["outcome"], row["steps"], f"{row['elapsed_s']:.3f}",
                "" if row["time_to_success_s"] is None else f"{row['time_to_success_s']:.3f}",
                f"{row['dist0']:.3f}", f"{row['dist_final']:.3f}", f"{row['best_dist']:.3f}",
                f"{row['progress_best']:.3f}", f"{row['ang_final']:.3f}",
                f"{row['path_len']:.3f}", f"{row['mean_v']:.3f}", f"{row['mean_abs_w']:.3f}",
                f"{row['min_range_end']:.3f}", f"{row['min_range_min']:.3f}",
                f"{row['front_min_end']:.3f}", f"{row['front_min_min']:.3f}",
                row["stuck_reason"].replace(",", ";"),
                int(row["oscillation_flag"]),
                row["model_path"], row["goals_csv_path"], row["stamp"],
            ])

    # ---------- Goals ----------
    def _parse_goals(self, goals_xy) -> List[Tuple[float, float]]:
        if goals_xy is None or not isinstance(goals_xy, (list, tuple)):
            return []
        vals = []
        for v in goals_xy:
            fv = _try_float(v)
            if fv is not None:
                vals.append(float(fv))
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
        self.scan_count += 1

    def _odom_sim_cb(self, msg: Odometry):
        self.odom_sim_msg = msg
        self.odom_sim_count += 1

    def _odom_raw_cb(self, msg: Odometry):
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

    # ---------- cmd ----------
    def _publish_twist(self, v: float, w: float):
        t = Twist()
        t.linear.x = float(v)
        t.angular.z = float(w)
        self.pub_cmd.publish(t)

    # ---------- Stuck Detection ----------
    def _effective_stuck_params(self, dist: float) -> Tuple[int, float]:
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

    # ---------- Action masking (eval consistency) ----------
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

    # ---------- Episode Finalize ----------
    def _finalize_episode(self, outcome: str, stuck_reason: str = "", oscillation_flag: bool = False, ang_final: float = 0.0):
        elapsed_s = time.monotonic() - self.episode_start_ts
        success = (outcome == "SUCCESS")

        if success and self.time_to_success_s is None:
            self.time_to_success_s = elapsed_s

        mean_v = (self.v_sum / self.step_count_for_means) if self.step_count_for_means > 0 else 0.0
        mean_abs_w = (self.abs_w_sum / self.step_count_for_means) if self.step_count_for_means > 0 else 0.0

        row = {
            "episode": self.episode_idx,
            "goal_idx": self.goal_idx,
            "trial_idx": self.trial_idx_for_goal,
            "goal_x": float(self.current_goal[0]),
            "goal_y": float(self.current_goal[1]),
            "success": bool(success),
            "outcome": outcome,
            "steps": int(self.step_idx),
            "elapsed_s": float(elapsed_s),
            "time_to_success_s": self.time_to_success_s,
            "dist0": float(self.dist0),
            "dist_final": float(self.last_dist),
            "best_dist": float(self.best_dist),
            "progress_best": float(self.dist0 - self.best_dist),
            "ang_final": float(ang_final),
            "path_len": float(self.path_len),
            "mean_v": float(mean_v),
            "mean_abs_w": float(mean_abs_w),
            "min_range_end": float(self.last_min_range),
            "min_range_min": float(self.min_range_min),
            "front_min_end": float(self.front_min_end),
            "front_min_min": float(self.front_min_min),
            "stuck_reason": stuck_reason,
            "oscillation_flag": bool(oscillation_flag),
            "model_path": self.model_path,
            "goals_csv_path": self.goals_csv_path,
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
        self.min_range_min = float("inf")
        self.front_min_end = float("inf")
        self.front_min_min = float("inf")

        self.last_dist = float("inf")
        self.best_dist = float("inf")
        self.dist0 = float("inf")

        self.path_len = 0.0
        self.last_pos = None

        self.abs_w_sum = 0.0
        self.v_sum = 0.0
        self.step_count_for_means = 0

        self.goal_hold_count = 0
        self.near_goal_steps = 0
        self.time_to_success_s = None

        self.pos_hist.clear()
        self.action_name_hist.clear()
        self.action_v_hist.clear()

        self._last_processed_scan_count = self.scan_count
        self._last_processed_odom_sim_count = self.odom_sim_count
        self._last_processed_odom_raw_count = self.odom_raw_count

        if self.reset_client.wait_for_service(1.0):
            self.reset_client.call_async(Empty.Request())
            self.get_logger().info(
                f"[EVAL] Reset requested | ep={self.episode_idx}/{self.total_episodes} "
                f"goal_idx={self.goal_idx} trial={self.trial_idx_for_goal} goal=({self.current_goal[0]:.2f},{self.current_goal[1]:.2f})"
            )
        else:
            self.get_logger().warn("[EVAL] Reset service not available (continuing without reset)")

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

                # Wait for fresh sensor/odom after reset
                have_fresh = (self.scan_count > self._last_processed_scan_count) and (self.odom_sim_count > self._last_processed_odom_sim_count)
                if self.use_raw_odom_for_global:
                    have_fresh = have_fresh and (self.odom_raw_count > self._last_processed_odom_raw_count)

                if have_fresh and dt > 0.25:
                    self.waiting_reset = False
                    gx, gy, _ = self._pose_global()
                    self.dist0 = math.hypot(self.current_goal[0] - gx, self.current_goal[1] - gy)
                    self.last_dist = self.dist0
                    self.best_dist = self.dist0
                    self.last_pos = (gx, gy)
                    self.pos_hist.append((gx, gy))
                    self.get_logger().info(f"[EVAL] START pos=({gx:.2f},{gy:.2f}) dist0={self.dist0:.2f}")
                return

            if not self._has_new_step_data():
                return

            self._last_processed_scan_count = self.scan_count
            self._last_processed_odom_sim_count = self.odom_sim_count
            self._last_processed_odom_raw_count = self.odom_raw_count

            gx, gy, gyaw = self._pose_global()
            self.pos_hist.append((gx, gy))

            if self.last_pos is not None:
                self.path_len += math.hypot(gx - self.last_pos[0], gy - self.last_pos[1])
            self.last_pos = (gx, gy)

            scan = self.scan_msg
            ranges = np.asarray(scan.ranges, dtype=np.float32)
            ranges = np.nan_to_num(ranges, nan=self.lidar_max, posinf=self.lidar_max, neginf=self.lidar_max)
            ranges = np.clip(ranges, 0.0, self.lidar_max)

            angle_min = float(getattr(scan, "angle_min", 0.0))
            angle_max = float(getattr(scan, "angle_max", 0.0))

            lidar_bins_norm, min_range = self.state_proc.bin_lidar_min(ranges.tolist(), self.lidar_max, angle_min=angle_min, angle_max=angle_max)
            front_min = self.state_proc.compute_front_min(
                ranges=ranges.tolist(),
                range_max=self.lidar_max,
                angle_min=angle_min,
                angle_max=angle_max,
                front_sector_deg=self.front_sector_deg,
            )

            self.last_min_range = float(min_range)
            self.min_range_min = min(self.min_range_min, float(min_range))
            self.front_min_end = float(front_min)
            self.front_min_min = min(self.front_min_min, float(front_min))

            dist = math.hypot(self.current_goal[0] - gx, self.current_goal[1] - gy)
            ang_err = angle_to_goal(gx, gy, gyaw, float(self.current_goal[0]), float(self.current_goal[1]))

            self.last_dist = float(dist)
            if dist < self.best_dist:
                self.best_dist = float(dist)

            # -------- SUCCESS Practical with Hysteresis + Hold --------
            if dist <= self.goal_tol_exit:
                self.near_goal_steps += 1
                self.goal_hold_count += 2 if dist <= self.goal_tol_enter else 1
            else:
                self.near_goal_steps = 0
                self.goal_hold_count = 0

            if self.goal_hold_count >= self.goal_hold_steps:
                self.time_to_success_s = time.monotonic() - self.episode_start_ts
                self.get_logger().info(
                    f"[EVAL] SUCCESS | ep={self.episode_idx} dist={dist:.2f} steps={self.step_idx} best={self.best_dist:.2f}"
                )
                self._finalize_episode("SUCCESS", ang_final=ang_err)
                return

            # Collision
            if min_range < self.collision_range:
                self.get_logger().warn(
                    f"[EVAL] COLLISION | ep={self.episode_idx} minR={min_range:.2f} steps={self.step_idx} best={self.best_dist:.2f}"
                )
                self._finalize_episode("COLLISION", ang_final=ang_err)
                return

            # Timeout
            if self.step_idx >= self.max_steps:
                self.get_logger().warn(
                    f"[EVAL] TIMEOUT | ep={self.episode_idx} dist={dist:.2f} steps={self.step_idx} best={self.best_dist:.2f}"
                )
                self._finalize_episode("TIMEOUT", ang_final=ang_err)
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
            mask = self._action_mask(front_min)
            a_idx = self.agent.act(state, training=False, action_mask=mask)
            if not isinstance(a_idx, int):
                a_idx = int(a_idx)
            a_idx = int(np.clip(a_idx, 0, self.action_size - 1))

            act = self.actions[a_idx]
            self._publish_twist(act.v, act.w)

            self.prev_v = act.v
            self.prev_w = act.w

            self.action_name_hist.append(act.name)
            self.action_v_hist.append(act.v)

            self.v_sum += abs(act.v)
            self.abs_w_sum += abs(act.w)
            self.step_count_for_means += 1

            self.step_idx += 1

            # -------- STUCK detection --------
            stuck_reason = ""
            osc_flag = False
            if self._is_stuck_by_oscillation():
                stuck_reason = "oscillation_rot_only"
                osc_flag = True
            elif self._is_stuck_by_motion(dist):
                win, eps = self._effective_stuck_params(dist)
                stuck_reason = f"no_motion(win={win},eps={eps:.3f})"

            if not stuck_reason and dist <= self.goal_tol_exit and self.near_goal_steps >= self.near_goal_max_steps_without_hold:
                stuck_reason = f"near_goal_no_hold(steps={self.near_goal_steps})"

            if stuck_reason:
                self.get_logger().warn(
                    f"[EVAL] STUCK | ep={self.episode_idx} dist={dist:.2f} steps={self.step_idx} best={self.best_dist:.2f} reason={stuck_reason}"
                )
                self._finalize_episode("STUCK", stuck_reason=stuck_reason, oscillation_flag=osc_flag, ang_final=ang_err)
                return

            if (self.step_idx % max(1, self.print_every_steps)) == 0:
                self.get_logger().info(
                    f"[EVAL] ep={self.episode_idx:03d} g={self.goal_idx} t={self.trial_idx_for_goal} step={self.step_idx:04d} "
                    f"pos=({gx:6.2f},{gy:6.2f}) dist={dist:5.2f} best={self.best_dist:5.2f} "
                    f"ang={ang_err:5.2f} minR={min_range:4.2f} front={front_min:4.2f} "
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