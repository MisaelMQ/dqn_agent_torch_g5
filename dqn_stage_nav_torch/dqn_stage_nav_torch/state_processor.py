from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class StateProcessorConfig:
    n_lidar_bins: int = 72
    max_goal_dist: float = 20.0
    range_max_fallback: float = 5.0


class StateProcessor:
    """
    state = [lidar_bins_norm (n_bins), aux]
    aux = [dx_r_norm, dy_r_norm, v_prev_norm, w_prev_norm, min_range_norm]
    """

    def __init__(self, cfg: StateProcessorConfig):
        self.cfg = cfg

    def bin_lidar_min(self, ranges: List[float], range_max: float) -> Tuple[np.ndarray, float]:
        n = len(ranges)
        if n == 0:
            bins = np.ones((self.cfg.n_lidar_bins,), dtype=np.float32)
            return bins, float(range_max)

        rmax = float(range_max) if (range_max and range_max > 0.0) else float(self.cfg.range_max_fallback)
        arr = np.asarray(ranges, dtype=np.float32)

        arr = np.where(np.isfinite(arr), arr, rmax)
        arr = np.where(arr > 0.0, arr, rmax)
        arr = np.clip(arr, 0.0, rmax)

        min_range = float(arr.min()) if arr.size > 0 else rmax

        n_bins = int(self.cfg.n_lidar_bins)
        edges = np.linspace(0, n, n_bins + 1, dtype=np.int32)
        b = np.zeros((n_bins,), dtype=np.float32)
        for i in range(n_bins):
            s = edges[i]
            e = edges[i + 1]
            if e <= s:
                b[i] = rmax
            else:
                b[i] = float(arr[s:e].min())

        b_norm = np.clip(b / rmax, 0.0, 1.0).astype(np.float32)
        return b_norm, min_range

    @staticmethod
    def goal_in_robot_frame(pos_g: Tuple[float, float], yaw_g: float, goal_g: Tuple[float, float]) -> Tuple[float, float, float]:
        dx = float(goal_g[0] - pos_g[0])
        dy = float(goal_g[1] - pos_g[1])
        dist = math.hypot(dx, dy)

        c = math.cos(yaw_g)
        s = math.sin(yaw_g)
        dx_r = c * dx + s * dy
        dy_r = -s * dx + c * dy
        return dx_r, dy_r, dist

    def build_state(
        self,
        lidar_bins_norm: np.ndarray,
        pos_g: Tuple[float, float],
        yaw_g: float,
        goal_g: Tuple[float, float],
        v_prev: float,
        w_prev: float,
        v_max: float,
        w_max: float,
        min_range: float,
        range_max: float,
    ) -> np.ndarray:
        dx_r, dy_r, _ = self.goal_in_robot_frame(pos_g, yaw_g, goal_g)

        mgd = float(self.cfg.max_goal_dist)
        dxn = float(np.clip(dx_r / mgd, -1.0, 1.0))
        dyn = float(np.clip(dy_r / mgd, -1.0, 1.0))

        vdn = 0.0 if v_max <= 1e-9 else float(np.clip(v_prev / v_max, -1.0, 1.0))
        wdn = 0.0 if w_max <= 1e-9 else float(np.clip(w_prev / w_max, -1.0, 1.0))

        rmax = float(range_max) if (range_max and range_max > 0.0) else float(self.cfg.range_max_fallback)
        min_norm = float(np.clip(min_range / rmax, 0.0, 1.0))

        aux = np.array([dxn, dyn, vdn, wdn, min_norm], dtype=np.float32)
        return np.concatenate([lidar_bins_norm.astype(np.float32), aux], axis=0)
