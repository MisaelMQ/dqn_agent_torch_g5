from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TorchDQNConfig:
    n_lidar_bins: int
    aux_dim: int
    action_size: int

    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = 256
    memory_size: int = 150_000
    min_memory_to_train: int = 5_000

    epsilon: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.99995

    use_double_dqn: bool = True
    use_dueling: bool = True

    # Target update: prefer hard updates for stability with PER + n-step
    soft_tau: float = 0.0
    hard_update_every: int = 2_000

    max_grad_norm: float = 10.0
    seed: int = 42
    device: str = "cuda"

    # N-step returns
    n_step: int = 3

    # PER
    use_per: bool = True
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = 200_000
    per_eps: float = 1e-6

    # --- PER Stability ---
    per_prio_clip: float = 100.0 


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _seed_everything(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


class QNetwork(nn.Module):
    def __init__(self, n_lidar_bins: int, aux_dim: int, action_size: int, dueling: bool):
        super().__init__()
        self.n_lidar_bins = int(n_lidar_bins)
        self.aux_dim = int(aux_dim)
        self.action_size = int(action_size)
        self.dueling = bool(dueling)

        self.lidar_net = nn.Sequential(
            nn.Linear(self.n_lidar_bins, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.aux_net = nn.Sequential(
            nn.Linear(self.aux_dim, 64),
            nn.ReLU(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(128 + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        if self.dueling:
            self.v_head = nn.Linear(256, 1)
            self.a_head = nn.Linear(256, self.action_size)
        else:
            self.head = nn.Linear(256, self.action_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        lidar = state[:, : self.n_lidar_bins]
        aux = state[:, self.n_lidar_bins : self.n_lidar_bins + self.aux_dim]
        z = self.trunk(torch.cat([self.lidar_net(lidar), self.aux_net(aux)], dim=1))
        if self.dueling:
            v = self.v_head(z)
            a = self.a_head(z)
            return v + (a - a.mean(dim=1, keepdim=True))
        return self.head(z)


class _SumTree:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        size = 1
        while size < self.capacity:
            size <<= 1
        self._size_pow2 = size
        self.tree = np.zeros(2 * self._size_pow2, dtype=np.float32)

    @property
    def total(self) -> float:
        return float(self.tree[1])

    def update(self, idx: int, value: float) -> None:
        i = int(idx) + self._size_pow2
        self.tree[i] = np.float32(value)
        i //= 2
        while i >= 1:
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]
            i //= 2

    def find_prefixsum_idx(self, mass: float) -> int:
        i = 1
        while i < self._size_pow2:
            left = 2 * i
            if mass <= float(self.tree[left]):
                i = left
            else:
                mass -= float(self.tree[left])
                i = left + 1
        return int(i - self._size_pow2)


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, seed: int, alpha: float, eps: float, prio_clip: float):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.prio_clip = float(max(prio_clip, eps))

        self.s = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.a = np.zeros((self.capacity, 1), dtype=np.int64)
        self.r = np.zeros((self.capacity, 1), dtype=np.float32)
        self.s2 = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.d = np.zeros((self.capacity, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0
        self.max_prio = 1.0
        self.tree = _SumTree(self.capacity)
        self.rng = np.random.default_rng(seed)

    def _sanitize_prio(self, p: float) -> float:
        p = float(p)
        if not np.isfinite(p) or p < 0.0:
            p = self.max_prio
        p = max(p, self.eps)
        return min(p, self.prio_clip)

    def push(
        self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool, prio: Optional[float] = None
    ) -> None:
        i = self.ptr
        self.s[i] = s
        self.a[i] = int(a)
        self.r[i] = float(r)
        self.s2[i] = s2
        self.d[i] = 1.0 if done else 0.0

        p = self._sanitize_prio(self.max_prio if prio is None else prio)
        p_alpha = float(p ** self.alpha)
        if not np.isfinite(p_alpha) or p_alpha <= 0.0:
            p_alpha = float((self.eps) ** self.alpha)
        self.tree.update(i, p_alpha)
        self.max_prio = max(self.max_prio, p)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float):
        n = int(batch_size)
        total = float(self.tree.total)

        if (self.size <= 0) or (not np.isfinite(total)) or (total <= 0.0):
            idx = self.rng.integers(0, self.size, size=n, dtype=np.int64)
            w = np.ones((n, 1), dtype=np.float32)
            return idx, self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx], w

        seg = total / n
        if (not np.isfinite(seg)) or (seg <= 0.0):
            idx = self.rng.integers(0, self.size, size=n, dtype=np.int64)
            w = np.ones((n, 1), dtype=np.float32)
            return idx, self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx], w

        idx = np.empty((n,), dtype=np.int64)
        for k in range(n):
            mass = (k * seg) + (float(self.rng.random()) * seg)
            mass = min(mass, total)
            idx[k] = self.tree.find_prefixsum_idx(mass)

        leaf = self.tree.tree[idx + self.tree._size_pow2]
        p = leaf / max(total, 1e-12)
        p = np.clip(p, 1e-12, 1.0)

        w = (self.size * p) ** (-float(beta))
        w = w / max(float(w.max()), 1e-12)
        w = w.astype(np.float32).reshape(-1, 1)

        return idx, self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx], w

    def update_priorities(self, idx: np.ndarray, prio: np.ndarray) -> None:
        idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        prio = np.asarray(prio, dtype=np.float32).reshape(-1)
        for i, p in zip(idx, prio):
            p = self._sanitize_prio(float(p))
            p_alpha = float(p ** self.alpha)
            if not np.isfinite(p_alpha) or p_alpha <= 0.0:
                p_alpha = float((self.eps) ** self.alpha)
            self.tree.update(int(i), p_alpha)
            self.max_prio = max(self.max_prio, p)


class UniformReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, seed: int):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)

        self.s = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.a = np.zeros((self.capacity, 1), dtype=np.int64)
        self.r = np.zeros((self.capacity, 1), dtype=np.float32)
        self.s2 = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.d = np.zeros((self.capacity, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def push(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool) -> None:
        i = self.ptr
        self.s[i] = s
        self.a[i] = int(a)
        self.r[i] = float(r)
        self.s2[i] = s2
        self.d[i] = 1.0 if done else 0.0

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, self.size, size=int(batch_size))
        w = np.ones((len(idx), 1), dtype=np.float32)
        return idx, self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx], w

    def update_priorities(self, idx: np.ndarray, prio: np.ndarray) -> None:
        return


class _NStepAccumulator:
    def __init__(self, n: int, gamma: float):
        self.n = max(1, int(n))
        self.gamma = float(gamma)
        self._s: Optional[np.ndarray] = None
        self._a: Optional[int] = None
        self._buf_r = []
        self._buf_done = []

    def reset(self) -> None:
        self._s = None
        self._a = None
        self._buf_r.clear()
        self._buf_done.clear()

    def push(self, s: np.ndarray, a: int, r: float, done: bool) -> Tuple[Optional[np.ndarray], Optional[int], Optional[float], bool]:
        if self._s is None:
            self._s = s
            self._a = int(a)

        self._buf_r.append(float(r))
        self._buf_done.append(bool(done))

        if len(self._buf_r) < self.n and not done:
            return None, None, None, False

        R = 0.0
        for i, ri in enumerate(self._buf_r[: self.n]):
            R += (self.gamma**i) * float(ri)

        done_n = any(self._buf_done[: self.n])
        s0, a0 = self._s, self._a

        if done or len(self._buf_r) >= self.n:
            self._buf_r.pop(0)
            self._buf_done.pop(0)
            if len(self._buf_r) == 0:
                self._s = None
                self._a = None
            else:
                self._s = s
                self._a = int(a)

        return s0, a0, float(R), done_n


class TorchDQNAgent:
    def __init__(self, cfg: TorchDQNConfig):
        self.cfg = cfg
        self.device = _resolve_device(cfg.device)
        _seed_everything(cfg.seed, self.device)

        self.state_dim = int(cfg.n_lidar_bins + cfg.aux_dim)

        self.q_online = QNetwork(cfg.n_lidar_bins, cfg.aux_dim, cfg.action_size, cfg.use_dueling).to(self.device)
        self.q_target = QNetwork(cfg.n_lidar_bins, cfg.aux_dim, cfg.action_size, cfg.use_dueling).to(self.device)
        self.q_target.load_state_dict(self.q_online.state_dict())
        self.q_target.eval()

        self.opt = torch.optim.Adam(self.q_online.parameters(), lr=float(cfg.lr))

        if cfg.use_per:
            self.buf = PrioritizedReplayBuffer(
                capacity=cfg.memory_size,
                state_dim=self.state_dim,
                seed=cfg.seed,
                alpha=cfg.per_alpha,
                eps=cfg.per_eps,
                prio_clip=cfg.per_prio_clip,
            )
        else:
            self.buf = UniformReplayBuffer(cfg.memory_size, self.state_dim, cfg.seed)

        self.epsilon = float(cfg.epsilon)
        self.train_steps = 0
        self.frame = 0

        # RNG 
        self.rng = np.random.default_rng(cfg.seed)

        self.nstep = _NStepAccumulator(cfg.n_step, cfg.gamma)

    def act(
        self,
        state: np.ndarray,
        training: bool = True,
        action_mask: Optional[Sequence[bool]] = None,
        return_q: bool = False,
    ) -> int | Tuple[int, np.ndarray]:
        mask = None
        if action_mask is not None:
            mask = np.asarray(list(action_mask), dtype=bool)
            if mask.size != int(self.cfg.action_size):
                raise ValueError(f"action_mask size {mask.size} != action_size {int(self.cfg.action_size)}")
            if not mask.any():
                mask = None  

        # Epsilon-greedy Exploration
        if training and random.random() < self.epsilon:
            if mask is None:
                a = random.randrange(self.cfg.action_size)
                return (a, np.full((self.cfg.action_size,), np.nan, dtype=np.float32)) if return_q else a
            allowed = np.flatnonzero(mask)
            a = int(self.rng.choice(allowed)) if hasattr(self, "rng") else int(np.random.choice(allowed))
            return (a, np.full((self.cfg.action_size,), np.nan, dtype=np.float32)) if return_q else a

        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.q_online(s).squeeze(0)  # [A]
            q_np = q.detach().cpu().numpy().astype(np.float32)

            if mask is not None:
                q_masked = q.clone()
                q_masked[~torch.as_tensor(mask, device=self.device)] = -1.0e9
                a = int(torch.argmax(q_masked).item())
            else:
                a = int(torch.argmax(q).item())

            return (a, q_np) if return_q else a


    def remember(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool) -> None:
        self.frame += 1

        s0, a0, rN, doneN = self.nstep.push(s, a, r, done)
        if s0 is None:
            if done:
                self.nstep.reset()
            return

        self.buf.push(s0, a0, rN, s2, doneN)

        if done:
            self.nstep.reset()

    def train_step(self) -> float:
        if self.buf.size < int(self.cfg.min_memory_to_train) or self.cfg.batch_size <= 0:
            return 0.0

        beta = self._per_beta()
        idx, s, a, r, s2, d, w = self.buf.sample(self.cfg.batch_size, beta)

        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(a, dtype=torch.int64, device=self.device)
        r = torch.as_tensor(r, dtype=torch.float32, device=self.device)
        s2 = torch.as_tensor(s2, dtype=torch.float32, device=self.device)
        d = torch.as_tensor(d, dtype=torch.float32, device=self.device)
        w = torch.as_tensor(w, dtype=torch.float32, device=self.device)

        q_sa = self.q_online(s).gather(1, a)

        with torch.no_grad():
            if self.cfg.use_double_dqn:
                a2 = self.q_online(s2).argmax(dim=1, keepdim=True)
                q_next = self.q_target(s2).gather(1, a2)
            else:
                q_next = self.q_target(s2).max(dim=1, keepdim=True)[0]

            gamma_n = float(self.cfg.gamma) ** max(1, int(self.cfg.n_step))
            target = r + gamma_n * q_next * (1.0 - d)

        td = target - q_sa
        loss = (w * F.smooth_l1_loss(q_sa, target, reduction="none")).mean()

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_online.parameters(), float(self.cfg.max_grad_norm))
        self.opt.step()

        # PER priority update 
        prio = td.detach().abs().cpu().numpy().reshape(-1)
        prio = np.clip(prio + float(self.cfg.per_eps), 0.0, float(self.cfg.per_prio_clip))
        self.buf.update_priorities(idx, prio)

        self.train_steps += 1
        self._update_target()
        self._decay_epsilon()
        return float(loss.item())

    def _per_beta(self) -> float:
        if not self.cfg.use_per:
            return 1.0
        t = min(1.0, float(self.frame) / max(1, int(self.cfg.per_beta_frames)))
        return float(self.cfg.per_beta_start + t * (1.0 - self.cfg.per_beta_start))

    def _update_target(self) -> None:
        if self.cfg.soft_tau and self.cfg.soft_tau > 0.0:
            tau = float(self.cfg.soft_tau)
            with torch.no_grad():
                for tp, op in zip(self.q_target.parameters(), self.q_online.parameters()):
                    tp.data.lerp_(op.data, tau)
            return

        k = int(self.cfg.hard_update_every)
        if k > 0 and (self.train_steps % k) == 0:
            self.q_target.load_state_dict(self.q_online.state_dict())

    def _decay_epsilon(self) -> None:
        if self.epsilon > float(self.cfg.epsilon_min):
            self.epsilon *= float(self.cfg.epsilon_decay)
            if self.epsilon < float(self.cfg.epsilon_min):
                self.epsilon = float(self.cfg.epsilon_min)

    def boost_exploration(self, eps_floor: float, eps_cap: float = 1.0) -> None:
        self.epsilon = float(min(float(eps_cap), max(self.epsilon, float(eps_floor))))

    
    def get_state_dim(self) -> int:
        return int(self.state_dim)

    def get_action_size(self) -> int:
        return int(self.cfg.action_size)

    def q_values(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.q_online(s).squeeze(0)
            return q.detach().cpu().numpy().astype(np.float32)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "epsilon": float(self.epsilon),
            "buffer_size": int(self.buf.size),
            "train_steps": int(self.train_steps),
            "device": str(self.device),
            "n_step": int(self.cfg.n_step),
            "use_per": bool(self.cfg.use_per),
        }

    def save_checkpoint(self, path: str, episode_idx: int, extra: Optional[dict] = None) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "model": self.q_online.state_dict(),
            "target": self.q_target.state_dict(),
            "optimizer": self.opt.state_dict(),
            "epsilon": float(self.epsilon),
            "episode": int(episode_idx),
            "train_steps": int(self.train_steps),
            "frame": int(self.frame),
            "cfg": asdict(self.cfg),
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    def load_checkpoint(self, path: str) -> Tuple[bool, Optional[int]]:
        if not os.path.exists(path):
            return False, None
        try:
            ckpt = torch.load(path, map_location=self.device)
            self.q_online.load_state_dict(ckpt["model"])
            if "target" in ckpt:
                self.q_target.load_state_dict(ckpt["target"])
            else:
                self.q_target.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt:
                self.opt.load_state_dict(ckpt["optimizer"])
            self.epsilon = float(ckpt.get("epsilon", self.cfg.epsilon))
            self.train_steps = int(ckpt.get("train_steps", 0))
            self.frame = int(ckpt.get("frame", 0))
            ep = ckpt.get("episode", None)
            return True, int(ep) if ep is not None else None
        except Exception:
            return False, None