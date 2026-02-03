from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

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
    lr: float = 3e-4
    batch_size: int = 256
    memory_size: int = 100_000
    min_memory_to_train: int = 1_000

    epsilon: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.99995

    # Target Update (Soft)
    tau: float = 0.005

    max_grad_norm: float = 10.0
    use_double_dqn: bool = True
    use_dueling: bool = True

    seed: int = 42
    device: str = "cuda"


class QNetwork(nn.Module):
    def __init__(self, n_lidar_bins: int, aux_dim: int, action_size: int, dueling: bool = True):
        super().__init__()
        self.n_lidar_bins = int(n_lidar_bins)
        self.aux_dim = int(aux_dim)
        self.action_size = int(action_size)
        self.dueling = bool(dueling)

        # MLP
        self.use_cnn = self.n_lidar_bins >= 30

        if self.use_cnn:
            self.feature_extractor = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv1d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            with torch.no_grad():
                dummy = torch.zeros(1, 1, self.n_lidar_bins)
                out_dim = int(self.feature_extractor(dummy).shape[1])
        else:
            self.feature_extractor = nn.Sequential(
                nn.Linear(self.n_lidar_bins, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
            )
            out_dim = 128

        self.aux_fc = nn.Sequential(
            nn.Linear(self.aux_dim, 64),
            nn.ReLU(),
        )

        self.trunk = nn.Sequential(
            nn.Linear(out_dim + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        if self.dueling:
            self.value_head = nn.Linear(256, 1)
            self.adv_head = nn.Linear(256, self.action_size)
        else:
            self.head = nn.Linear(256, self.action_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        lidar = state[:, : self.n_lidar_bins]
        aux = state[:, self.n_lidar_bins : self.n_lidar_bins + self.aux_dim]

        if self.use_cnn:
            lidar = lidar.unsqueeze(1)  # (B, 1, N)

        x = self.feature_extractor(lidar)
        y = self.aux_fc(aux)

        z = self.trunk(torch.cat([x, y], dim=1))

        if self.dueling:
            val = self.value_head(z)  # (B, 1)
            adv = self.adv_head(z)    # (B, A)
            return val + (adv - adv.mean(dim=1, keepdim=True))
        return self.head(z)


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, seed: int = 42):
        self.capacity = int(capacity)
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((capacity, 1), dtype=np.int64)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def push(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, d: float) -> None:
        self.state[self.ptr] = s
        self.action[self.ptr] = a
        self.reward[self.ptr] = r
        self.next_state[self.ptr] = s2
        self.done[self.ptr] = d

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = self.rng.integers(0, self.size, size=batch_size)
        return (
            self.state[idx],
            self.action[idx],
            self.reward[idx],
            self.next_state[idx],
            self.done[idx],
        )


class TorchDQNAgent:
    def __init__(self, cfg: TorchDQNConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu")

        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(cfg.seed)

        self.state_dim = int(cfg.n_lidar_bins + cfg.aux_dim)

        self.q_online = QNetwork(cfg.n_lidar_bins, cfg.aux_dim, cfg.action_size, cfg.use_dueling).to(self.device)
        self.q_target = QNetwork(cfg.n_lidar_bins, cfg.aux_dim, cfg.action_size, cfg.use_dueling).to(self.device)
        self.q_target.load_state_dict(self.q_online.state_dict())
        self.q_target.eval()

        self.optimizer = torch.optim.Adam(self.q_online.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.memory_size, self.state_dim, cfg.seed)

        self.epsilon = float(cfg.epsilon)
        self.train_steps = 0

    def act(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randrange(self.cfg.action_size)

        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.q_online(s)
            return int(q.argmax(dim=1).item())

    def remember(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool) -> None:
        self.buffer.push(s, int(a), float(r), s2, 1.0 if done else 0.0)

    def train_step(self) -> float:
        if self.buffer.size < self.cfg.min_memory_to_train:
            return 0.0

        s, a, r, s2, d = self.buffer.sample(self.cfg.batch_size)

        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(a, dtype=torch.int64, device=self.device)
        r = torch.as_tensor(r, dtype=torch.float32, device=self.device)
        s2 = torch.as_tensor(s2, dtype=torch.float32, device=self.device)
        d = torch.as_tensor(d, dtype=torch.float32, device=self.device)

        q_curr = self.q_online(s).gather(1, a)

        with torch.no_grad():
            if self.cfg.use_double_dqn:
                a_next = self.q_online(s2).argmax(dim=1, keepdim=True)
                q_next = self.q_target(s2).gather(1, a_next)
            else:
                q_next = self.q_target(s2).max(dim=1, keepdim=True)[0]

            target = r + (self.cfg.gamma * q_next * (1.0 - d))

        loss = F.smooth_l1_loss(q_curr, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_online.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()

        self.train_steps += 1

        # Soft Update Target
        tau = float(self.cfg.tau)
        with torch.no_grad():
            for tp, op in zip(self.q_target.parameters(), self.q_online.parameters()):
                tp.data.mul_(1.0 - tau).add_(op.data, alpha=tau)

        # Epsilon Decay
        if self.epsilon > self.cfg.epsilon_min:
            self.epsilon *= self.cfg.epsilon_decay
            if self.epsilon < self.cfg.epsilon_min:
                self.epsilon = self.cfg.epsilon_min

        return float(loss.item())

    def boost_exploration(self, eps_floor: float, eps_cap: float = 1.0) -> None:
        eps_floor = float(eps_floor)
        eps_cap = float(eps_cap)
        self.epsilon = float(min(eps_cap, max(self.epsilon, eps_floor)))

    def get_stats(self) -> Dict[str, Any]:
        return {
            "epsilon": float(self.epsilon),
            "buffer_size": int(self.buffer.size),
            "train_steps": int(self.train_steps),
            "device": str(self.device),
        }

    def save_checkpoint(self, path: str, episode_idx: int, extra: Optional[dict] = None) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "model": self.q_online.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": float(self.epsilon),
            "episode": int(episode_idx),
            "cfg": self.cfg.__dict__,
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
            self.q_target.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.epsilon = float(ckpt.get("epsilon", self.cfg.epsilon))
            ep = ckpt.get("episode", None)
            return True, int(ep) if ep is not None else None
        except Exception as e:
            print(f"Error loading model: {e}")
            return False, None
