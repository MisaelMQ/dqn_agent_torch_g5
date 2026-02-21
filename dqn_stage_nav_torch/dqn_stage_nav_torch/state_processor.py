from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np


@dataclass
class StateProcessorConfig:
    # Lidar -> bins
    n_lidar_bins: int = 72
    range_max_fallback: float = 5.0

    # Goal normalization
    max_goal_dist: float = 20.0
    fov_deg: float = 0.0
    front_sector_deg: float = 30.0


class StateProcessor:
    AUX_DIM: int = 5

    def __init__(self, cfg: StateProcessorConfig):
        self.cfg = cfg

    def _sanitize_ranges(self, ranges: Iterable[float], range_max: float) -> Tuple[np.ndarray, float]:
        rmax = float(range_max) if (range_max and range_max > 0.0) else float(self.cfg.range_max_fallback)
        arr = np.asarray(list(ranges), dtype=np.float32)
        if arr.size == 0:
            return np.empty((0,), dtype=np.float32), rmax

        arr = np.where(np.isfinite(arr), arr, rmax)
        arr = np.where(arr > 0.0, arr, rmax)
        arr = np.clip(arr, 0.0, rmax)
        return arr, rmax

    def _maybe_crop_fov(self, arr: np.ndarray, angle_min: float | None, angle_max: float | None) -> np.ndarray:
        fov = float(self.cfg.fov_deg)
        if fov <= 0.0 or arr.size == 0:
            return arr
        if angle_min is None or angle_max is None:
            return arr

        total = float(angle_max - angle_min)
        if total <= 1e-9:
            return arr

        desired = math.radians(fov)
        if desired >= total:
            return arr

        # FOV centered
        center = 0.0
        a0 = center - desired / 2.0
        a1 = center + desired / 2.0

        n = arr.size
        i0 = int(np.clip((a0 - angle_min) / total * n, 0, n))
        i1 = int(np.clip((a1 - angle_min) / total * n, 0, n))
        if i1 <= i0:
            return arr
        return arr[i0:i1]

    def bin_lidar_min(
        self,
        ranges: Iterable[float],
        range_max: float,
        angle_min: float | None = None,
        angle_max: float | None = None,
    ) -> Tuple[np.ndarray, float]:
        arr, rmax = self._sanitize_ranges(ranges, range_max)
        arr = self._maybe_crop_fov(arr, angle_min, angle_max)

        if arr.size == 0:
            bins = np.ones((int(self.cfg.n_lidar_bins),), dtype=np.float32)
            return bins, rmax

        min_range = float(arr.min())
        n_bins = int(self.cfg.n_lidar_bins)

        edges = np.linspace(0, arr.size, n_bins + 1, dtype=np.int32)
        out = np.empty((n_bins,), dtype=np.float32)

        for i in range(n_bins):
            s = int(edges[i])
            e = int(edges[i + 1])
            out[i] = float(arr[s:e].min()) if e > s else rmax

        out = np.clip(out / rmax, 0.0, 1.0).astype(np.float32)
        return out, min_range

    def compute_front_min(
        self,
        ranges: Iterable[float],
        range_max: float,
        angle_min: float | None = None,
        angle_max: float | None = None,
        front_sector_deg: float | None = None,
    ) -> float:
        arr, rmax = self._sanitize_ranges(ranges, range_max)
        arr = self._maybe_crop_fov(arr, angle_min, angle_max)

        if arr.size == 0:
            return float(rmax)

        fdeg = float(self.cfg.front_sector_deg if front_sector_deg is None else front_sector_deg)
        if fdeg <= 0.0:
            return float(arr.min())

        n = int(arr.size)
        if angle_min is None or angle_max is None:
            # fallback: sector central proporcional (aprox)
            frac = min(1.0, max(0.01, fdeg / 180.0))  # 30° ~ 0.166 de 180°
            w = max(1, int(round(frac * n)))
            c = n // 2
            i0 = max(0, c - w // 2)
            i1 = min(n, i0 + w)
            return float(arr[i0:i1].min()) if i1 > i0 else float(arr.min())

        total = float(angle_max - angle_min)
        if total <= 1e-9:
            return float(arr.min())

        desired = math.radians(fdeg)
        desired = min(desired, total)
        a0 = -desired / 2.0
        a1 = +desired / 2.0

        i0 = int(np.clip((a0 - angle_min) / total * n, 0, n))
        i1 = int(np.clip((a1 - angle_min) / total * n, 0, n))
        if i1 <= i0:
            return float(arr.min())
        return float(arr[i0:i1].min())

    @staticmethod
    def goal_in_robot_frame(
        pos_g: Tuple[float, float],
        yaw_g: float,
        goal_g: Tuple[float, float],
    ) -> Tuple[float, float, float]:
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

    def build_state_and_meta(
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
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        dx_r, dy_r, dist = self.goal_in_robot_frame(pos_g, yaw_g, goal_g)
        state = self.build_state(
            lidar_bins_norm=lidar_bins_norm,
            pos_g=pos_g,
            yaw_g=yaw_g,
            goal_g=goal_g,
            v_prev=v_prev,
            w_prev=w_prev,
            v_max=v_max,
            w_max=w_max,
            min_range=min_range,
            range_max=range_max,
        )
        meta = {
            "dx_r": float(dx_r),
            "dy_r": float(dy_r),
            "dist": float(dist),
            "min_range": float(min_range),
        }
        return state, meta