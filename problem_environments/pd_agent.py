"""
Provable Defense main (RAW 0–255). Agent normalizes internally (x/self.norm).

SCRIPTS
1) pd
python main_pd.py --env BreakoutNoFrameskip-v4 --samples 12000 --d 1024 --save-projector ckpts/projector_84x84_d1024.pt

2) test pd (with trigger)
python main_pd.py --env BreakoutNoFrameskip-v4 --load-projector ckpts/projector_84x84_d1024.pt --eval-episodes 20 --do-trigger-eval

python main_pd.py --env BreakoutNoFrameskip-v4 --samples 12000 --d 5000 --save-projector ckpts/projector_84x84_d5000.pt
python main_pd.py --env BreakoutNoFrameskip-v4 --load-projector ckpts/projector_84x84_d5000.pt --eval-episodes 20 --do-trigger-eval

---
python main_pd.py --env BreakoutNoFrameskip-v4 --samples 32768 --d 20000 --save-projector ckpts/pj_stack4_d20K.pt --eval-episodes 20 --do-trigger-eval --include-stack
---

"""
from __future__ import annotations
import argparse
import os
import random
import json
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Categorical
from typing import Optional
from problem_environments.atari_environment import Agent
#
# from provable_defense import (
#     SafeSubspaceEstimator,
#     StateProjector,
#     SafeSubspace,  # for save/load
#     # SanitizedPolicyWrapper,  # not used; we provide an adapter matching your Agent API
# )

# provable_defense.py
# PyTorch implementation of "subspace sanitization" (Provable Defense in RL)
# Works with vector or image observations (e.g., Atari 4x84x84).
import math
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------- Utilities ----------

def _flatten_obs(obs: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """
    obs: [..., D...] tensor. Returns (flattened 2D tensor [N, D], original_tail_shape).
    Supports single obs [D...] or batch [N, D...].
    """
    if obs.dim() == 1:
        return obs.unsqueeze(0), tuple(obs.shape)
    elif obs.dim() >= 2:
        n = obs.shape[0]
        tail = tuple(obs.shape[1:])
        return obs.reshape(n, -1), tail
    else:
        raise ValueError(f"Unsupported obs dim: {obs.shape}")


def _unflatten_obs(flat: torch.Tensor, tail_shape: Tuple[int, ...]) -> torch.Tensor:
    if flat.dim() != 2:
        raise ValueError("flat must be [N, D]")
    n = flat.shape[0]
    return flat.reshape((n,) + tail_shape)


# ---------- Safe subspace estimator ----------

@dataclass
class SafeSubspace:
    mean: torch.Tensor  # [D]
    components: torch.Tensor  # [D, d] principal axes (column-wise)
    d: int  # kept dimension
    D: int  # original dimension
    clamp_min: Optional[float] = None
    clamp_max: Optional[float] = None

    @torch.no_grad()
    def project(self, obs: torch.Tensor) -> torch.Tensor:
        """Project obs onto safe subspace. obs can be [D...] or [N, D...]."""
        device = self.mean.device
        is_single = (obs.dim() == 1)
        if is_single:
            obs = obs.unsqueeze(0)
        flat, tail = _flatten_obs(obs)
        flat = flat.to(device, dtype=self.mean.dtype)

        centered = flat - self.mean  # [N, D]
        # y = U (U^T centered) + mean
        U = self.components  # [D, d]
        y = centered @ U  # [N, d]
        proj = y @ U.T  # [N, D]
        proj = proj + self.mean

        if (self.clamp_min is not None) or (self.clamp_max is not None):
            lo = self.clamp_min if self.clamp_min is not None else -float("inf")
            hi = self.clamp_max if self.clamp_max is not None else float("inf")
            proj = proj.clamp(min=lo, max=hi)

        out = _unflatten_obs(proj, tail)
        return out.squeeze(0) if is_single else out


class SafeSubspaceEstimator:
    """
    Estimate safe subspace E from clean observations using low-rank PCA.
    Uses torch.pca_lowrank for scalability (works on GPU).
    """

    def __init__(self,
                 device: Optional[torch.device] = None,
                 dtype: torch.dtype = torch.float32):
        self.device = device or torch.device("cpu")
        self.dtype = dtype

    @torch.no_grad()
    def fit(self,
            clean_obs_iter: Iterable[torch.Tensor],
            max_components: Optional[int] = None,
            d: Optional[int] = None,
            eig_drop_ratio: float = 1e-3,
            eig_abs_floor: float = 1e-10,
            clamp_range: Optional[Tuple[float, float]] = None) -> SafeSubspace:
        """
        clean_obs_iter: yields obs tensors (either [D...] or [N, D...]) *after the same preprocessing*
                        your policy expects (e.g., stacked frames, normalization).
        max_components: upper bound on computed PCs (for big D). If None, compute up to D.
        d: if provided, keep exactly d components. Otherwise choose via spectrum drop & floor.
        eig_drop_ratio: keep eigvals >= eig_drop_ratio * eigval_max.
        eig_abs_floor: also drop eigvals < eig_abs_floor (paper used 1e-10 as a heuristic).
        clamp_range: optional (min, max) to clamp projected pixels back to valid range, e.g. (0, 1).

        Returns: SafeSubspace
        """
        # 1) stack to [N, D]
        flats = []
        for ob in clean_obs_iter:
            if isinstance(ob, np.ndarray):
                ob = torch.from_numpy(ob)
            if ob.dim() == 1:
                flat = ob.unsqueeze(0)
            elif ob.dim() >= 2:
                flat, _ = _flatten_obs(ob)
            else:
                raise ValueError("Invalid obs shape")
            flats.append(flat)
        X = torch.cat(flats, dim=0).to(self.device, self.dtype)  # [N, D]
        N, D = X.shape

        # 2) center
        mean = X.mean(dim=0, keepdim=True)  # [1, D]
        Xc = X - mean

        # 3) PCA (low-rank)
        if max_components is None:
            q = min(D, N)  # full rank (might be heavy for big Atari)
        else:
            q = min(int(max_components), D, N)

        # U: [N,k], S: [k], V: [D,k]  (Xc ≈ U diag(S) V^T)
        U, S, V = torch.pca_lowrank(Xc, q=q, center=False)

        # 4) choose d
        if d is None:
            eig = (S ** 2) / (N - 1)  # eigenvalues of covariance
            eig_max = torch.max(eig) if eig.numel() > 0 else torch.tensor(0., device=self.device)
            keep = (eig >= eig_abs_floor) & (eig >= eig_drop_ratio * eig_max)
            d = int(torch.count_nonzero(keep).item())
            if d == 0 and eig.numel() > 0:
                d = 1  # keep at least one

        components = V[:, :d].contiguous()  # [D, d]

        ss = SafeSubspace(
            mean=mean.squeeze(0),
            components=components,
            d=d,
            D=D,
            clamp_min=None if clamp_range is None else clamp_range[0],
            clamp_max=None if clamp_range is None else clamp_range[1],
        )
        return ss


# ---------- Runtime projectors / wrappers ----------

class StateProjector(nn.Module):
    """Torch Module wrapper so you can .to(device), save, load."""

    def __init__(self, safe_subspace: SafeSubspace):
        super().__init__()
        self.register_buffer("mean", safe_subspace.mean.clone())
        self.register_buffer("components", safe_subspace.components.clone())
        self.D = int(safe_subspace.D)
        self.d = int(safe_subspace.d)
        self.clamp_min = (torch.tensor(float("-inf"))
                          if safe_subspace.clamp_min is None else torch.tensor(safe_subspace.clamp_min))
        self.clamp_max = (torch.tensor(float("inf"))
                          if safe_subspace.clamp_max is None else torch.tensor(safe_subspace.clamp_max))

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # same as SafeSubspace.project but uses registered buffers
        is_single = (obs.dim() == 1)
        if is_single:
            obs = obs.unsqueeze(0)
        flat, tail = _flatten_obs(obs)
        flat = flat.to(self.mean.dtype)
        centered = flat - self.mean
        U = self.components
        y = centered @ U
        proj = y @ U.T
        proj = proj + self.mean
        proj = torch.clamp(proj, min=float(self.clamp_min), max=float(self.clamp_max))
        out = _unflatten_obs(proj, tail)
        return out.squeeze(0) if is_single else out


class SanitizedPolicyWrapper(nn.Module):
    """
    Wrap any torch policy with a projector. The base policy must accept preprocessed obs.
    Expect base_policy(obs)-> action logits or distribution (unchanged).
    """

    def __init__(self, base_policy: nn.Module, projector: StateProjector):
        super().__init__()
        self.base_policy = base_policy
        self.projector = projector

    @torch.no_grad()
    def forward(self, obs: torch.Tensor, *args, **kwargs):
        obs_proj = self.projector(obs)
        return self.base_policy(obs_proj, *args, **kwargs)

    # If your policy exposes a `.act(obs)` method:
    @torch.no_grad()
    def act(self, obs: torch.Tensor, *args, **kwargs):
        obs_proj = self.projector(obs)
        if hasattr(self.base_policy, "act"):
            return self.base_policy.act(obs_proj, *args, **kwargs)
        return self.forward(obs_proj, *args, **kwargs)


# ---------- Gym/Gymnasium Observation Wrapper (optional) ----------

try:
    import gymnasium as gym
    # import gym
    from gym import ObservationWrapper
    from PIL import Image


    class ProjectObsWrapper(ObservationWrapper):
        def __init__(self, env: gym.Env, projector: StateProjector):
            super().__init__(env)
            self.projector = projector

        def observation(self, observation):
            # convert to torch, project, back to numpy
            with torch.no_grad():
                x = torch.from_numpy(observation).float()
                if x.dim() >= 2:
                    # ensure channel-first when flattening (Atari is usually CHW already after wrappers)
                    pass
                y = self.projector(x).cpu().numpy()
                return y
except Exception:
    ProjectObsWrapper = None

# ----------------------------
# Env factory (CLEAN ENV)
# ----------------------------
from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


def make_atari_env(env_name: str, render_mode: str = "rgb_array"):
    """Your original make_atari_env; returns an env that outputs 84x84 grayscale frames stacked by 4.
    The final observation is typically LazyFrames with shape (84, 84, 4) (HWC).
    """
    env = gym.make(env_name, render_mode=render_mode)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    # env = EpisodicLifeEnv(env)  # optional
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    # env = ClipRewardEnv(env)    # optional
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)
    return env


# ----------------------------
# Utilities
# ----------------------------

# Adapter: make sanitized policy expose the same API as your Agent
class SanitizedAgentAdapter(torch.nn.Module):
    def __init__(self, base_agent: Agent, projector: StateProjector):
        super().__init__()
        self.base = base_agent
        self.projector = projector

    @torch.no_grad()
    def _proj(self, x: torch.Tensor) -> torch.Tensor:
        # Prefer frame-wise projection if the projector is learned on single-frame (84x84)
        p = self.projector
        mean = getattr(p, "mean")
        U = getattr(p, "components")
        x = x.to(mean.device, dtype=mean.dtype)
        Hx = x.shape[-2] if x.dim() >= 2 else None
        Wx = x.shape[-1] if x.dim() >= 2 else None
        # If projector.D == H*W and input has channels, apply per-frame projection
        if (x.dim() == 4) and (mean.numel() == Hx * Wx):  # [B,C,H,W]
            B, C, H, W = x.shape
            flat = x.reshape(B * C, H * W)
            centered = flat - mean
            y = (centered @ U) @ U.t() + mean
            # clamp like StateProjector
            lo = float(p.clamp_min)
            hi = float(p.clamp_max)
            y = y.clamp(min=lo, max=hi)
            return y.reshape(B, C, H, W)
        elif (x.dim() == 3) and (mean.numel() == Hx * Wx):  # [C,H,W]
            C, H, W = x.shape
            flat = x.reshape(C, H * W)
            centered = flat - mean
            y = (centered @ U) @ U.t() + mean
            lo = float(p.clamp_min)
            hi = float(p.clamp_max)
            y = y.clamp(min=lo, max=hi)
            return y.reshape(C, H, W)
        # Fallback: projector handles other shapes (e.g., when learned on 4*84*84)
        return p(x)

    @torch.no_grad()
    def get_action_and_value(self, x: torch.Tensor, action=None):
        xp = self._proj(x)
        return self.base.get_action_and_value(xp, action)

    @torch.no_grad()
    def get_action_dist(self, x: torch.Tensor):
        xp = self._proj(x)
        return self.base.get_action_dist(xp)

    @torch.no_grad()
    def get_value(self, x: torch.Tensor):
        xp = self._proj(x)
        return self.base.get_value(xp)

    @torch.no_grad()
    def get_action(self, x: torch.Tensor, idx: int = -1):

        # print("before", x.min(), x.max())
        xp = self._proj(x)
        # print("after", xp.min(), xp.max())
        # Save projected image if idx is provided
        if idx != -1:
            arr = xp.detach().clamp(0, 255).to(torch.uint8).cpu()
            # (B,C,H,W) -> (C,H,W)
            if arr.dim() == 4:
                arr = arr[0]
            img = Image.fromarray(arr[-1].numpy(), mode='L')
            # img.save(f"test_results/BreakoutNoFrameskip-v4/b/image_{idx}.png")
            img.save(f"test_results/BreakoutNoFrameskip-v4/b/image_{idx}.png")
        return self.base.get_action(xp)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def as_tensor_chw_uint(obs_np):
    """Ensure CHW (4,84,84) as float32, but keep RAW value scale (0–255)."""
    arr = np.array(obs_np)
    # FrameStack from SB3 returns HWC (84,84,4)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        print("not reasonable")
        exit()
        arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    return torch.as_tensor(arr, dtype=torch.float32)


# ---- Inline trigger utilities (no wrapper needed) ----
class ImagePoison:
    def __init__(self, pattern, min_val: float, max_val: float, numpy: bool = False):
        self.pattern = pattern
        self.min = min_val
        self.max = max_val
        self.numpy = numpy

    def __call__(self, state):
        if self.numpy:
            poisoned = np.float64(state)
            poisoned += self.pattern
            poisoned = np.clip(poisoned, self.min, self.max)
        else:
            poisoned = torch.clone(state)
            poisoned += self.pattern
            poisoned = torch.clamp(poisoned, self.min, self.max)
        return poisoned


def make_checker_pattern(shape=(4, 84, 84), patch_hw=(8, 8), coord=(0, 0), channel="last", device="cpu",
                         value: float = 255.0, checker: bool = True):
    """Build an additive trigger pattern tensor in RAW space.
    - shape: (C,H,W)
    - patch_hw: (ph, pw) size
    - coord: (i,j) top-left coord of the patch (on H,W)
    - channel: "last" | "all" | "0"|"1"|"2"|"3"
    - value: high value used for checkerboard (low=0)
    - checker: if False, fill with constant `value`
    Returns: torch.FloatTensor of `shape` placed on `device`.
    """
    C, H, W = shape
    ph, pw = patch_hw
    i, j = coord
    i = max(0, min(H - ph, i))
    j = max(0, min(W - pw, j))

    pat = torch.zeros(shape, dtype=torch.float32, device=device)

    # choose channels
    if channel == "last":
        ch_idx = [C - 1]
    elif channel == "all":
        ch_idx = list(range(C))
    elif channel in {"0", "1", "2", "3"}:
        ch_idx = [int(channel)]
    else:
        ch_idx = [C - 1]

    # build patch
    if checker:
        yy, xx = torch.meshgrid(torch.arange(ph, device=device), torch.arange(pw, device=device), indexing="ij")
        patch2d = ((yy + xx) % 2).float() * value
    else:
        patch2d = torch.full((ph, pw), fill_value=value, device=device)

    for c in ch_idx:
        pat[c, i:i + ph, j:j + pw] = patch2d
    return pat


def Single_Stacked_Img_Pattern(img_size, trigger_size, loc=(0, 0), min=-255, max=255, checker=True):
    pattern = torch.zeros(size=img_size)
    for i in range(trigger_size[0]):
        for j in range(trigger_size[1]):
            if checker and (i + j) % 2 == 0:
                pattern[:, i + loc[0], j + loc[1]] = min
            else:
                pattern[:, i + loc[0], j + loc[1]] = max
    return pattern.long()


# NOTE: Your Agent expects `envs` to be the actual env object
# (with your own code providing `.single_action_space` / `.single_observation_space` as needed).
# So we directly pass the env into Agent, as requested.

def build_agent(env, device: torch.device):
    agent = Agent(envs=env, image=True).to(device).eval()
    return agent


def maybe_load_agent(agent: Agent, ckpt_path: Optional[str]):
    if ckpt_path and os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location=agent.actor.weight.device)
        # allow both direct state_dict or wrapped dict
        sd = state.get("state_dict", state)
        agent.load_state_dict(sd, strict=False)
        print(f"[INFO] Loaded agent weights from {ckpt_path}")
    return agent


# ---- projector save/load helpers ----

def ensure_dir_for(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def save_projector(path: str, ss: SafeSubspace):
    ensure_dir_for(path)
    payload = {
        "mean": ss.mean.detach().cpu(),
        "components": ss.components.detach().cpu(),
        "d": int(ss.d),
        "D": int(ss.D),
        "clamp_min": ss.clamp_min,
        "clamp_max": ss.clamp_max,
    }
    torch.save(payload, path)


def load_projector(path: str, device: torch.device) -> SafeSubspace:
    blob = torch.load(path, map_location=device)
    mean = blob["mean"].to(device)
    comps = blob["components"].to(device)
    return SafeSubspace(
        mean=mean,
        components=comps,
        d=int(blob.get("d", comps.shape[1])),
        D=int(blob.get("D", mean.numel())),
        clamp_min=blob.get("clamp_min", 0.0),
        clamp_max=blob.get("clamp_max", 255.0),
    )


def append_jsonl(path: str, obj: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


# ----------------------------
# Clean sample iterator (RAW space)
# ----------------------------

def _reset_env(env):
    obs, info = env.reset()
    return obs


def _step_env(env, action):
    # Gymnasium: (obs, reward, terminated, truncated, info)
    obs, r, term, trunc, info = env.step(action)
    done = term or trunc
    return obs, r, done, info


def clean_obs_iter(env_clean, agent: Agent, n_samples: int, device: torch.device, include: bool):
    """Yield RAW obs tensors in CHW for estimator. To learn frame-wise subspace (84x84),
    we intentionally yield 3D tensors [C,H,W] so estimator treats channels as separate samples.
    """
    cnt = 0
    while cnt < n_samples:
        obs = _reset_env(env_clean)
        done = False
        while not done and cnt < n_samples:
            if include:
                yield as_tensor_chw_uint(obs).unsqueeze(0)
            else:
                # NOTE: yield 3D CHW (no batch dim) to get D=84*84
                yield as_tensor_chw_uint(obs)
            with torch.no_grad():
                x = as_tensor_chw_uint(obs).unsqueeze(0).to(device)
                a, _, _, _ = agent.get_action_and_value(x)
            obs, r, done, info = _step_env(env_clean, a.item())
            cnt += 1


# ----------------------------
# Eval loop helper
# ----------------------------

def eval_policy(env, agent_like, episodes: int, device: torch.device, desc: str = ""):
    rets = []
    for ep in range(episodes):
        obs = _reset_env(env)
        done, ep_ret = False, 0.0
        while not done:
            x = as_tensor_chw_uint(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                a, _, _, _ = agent_like.get_action_and_value(x)
            obs, r, done, info = _step_env(env, a.item())
            ep_ret += r
        rets.append(ep_ret)
    print(f"[EVAL]{' ' + desc if desc else ''} episodes={episodes}  mean={np.mean(rets):.2f}  std={np.std(rets):.2f}")
    return rets


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Provable Defense (Projection in RAW 0–255 space)")
    parser.add_argument("--env", type=str, default="BreakoutNoFrameskip-v4", help="Clean env id (default: v4)")
    parser.add_argument("--render-mode", type=str, default="rgb_array", choices=["rgb_array", "human"],
                        help="Render mode")
    parser.add_argument("--samples", type=int, default=8000, help="Number of clean observations to sample for PCA")
    parser.add_argument("--d", type=int, default=1024, help="Kept principal components (ignored if --auto-d)")
    parser.add_argument("--auto-d", action="store_true", help="Enable automatic d selection via dual thresholds")
    parser.add_argument("--eig-drop-ratio", type=float, default=1e-3,
                        help="Auto-d relative eig threshold (keep >= ratio*max)")
    parser.add_argument("--eig-abs-floor", type=float, default=1e-10, help="Auto-d absolute eig floor")
    parser.add_argument("--max-components", type=int, default=None, help="Upper bound on PCA components (for memory)")
    parser.add_argument("--clamp-min", type=float, default=0.0, help="Clamp min in RAW space")
    parser.add_argument("--clamp-max", type=float, default=255.0, help="Clamp max in RAW space")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval-episodes", type=int, default=0, help="Optionally evaluate sanitized agent on CLEAN env")
    parser.add_argument("--agent-ckpt", type=str, default='breakout_models/checker_8_sn.cleanrl_model',
                        help="Optional path to Trojan agent checkpoint (.pt/.pth)")
    # Trigger (inline) settings — no wrapper needed
    parser.add_argument("--do-trigger-eval", action="store_true", help="Evaluate with inline trigger injection")
    parser.add_argument("--poisoning-rate", type=float, default=0.1, help="Per-step probability to apply trigger")
    parser.add_argument("--trigger-size", type=int, default=8)
    parser.add_argument("--checker", action="store_true", help="Use checkerboard (else constant patch)")
    # Save/Load
    parser.add_argument("--save-projector", type=str, default=None, help="Path to save fitted projector .pt")
    parser.add_argument("--load-projector", type=str, default=None, help="Path to load projector .pt and skip fitting")
    parser.add_argument("--save-metrics", type=str, default=None, help="Append eval metrics as JSONL to this path")
    parser.add_argument("--include-stack", action="store_true", help="Include 4, 84, 84 to fit projector")

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    # 1) Build CLEAN env using your factory
    env_clean = make_atari_env(args.env, render_mode=args.render_mode)

    # 2) Build your Agent (Trojan) and optionally load weights
    trojan_agent = build_agent(env_clean, device)
    trojan_agent = maybe_load_agent(trojan_agent, args.agent_ckpt)
    trojan_agent.eval()

    # 3) Fit safe subspace in RAW space
    est = SafeSubspaceEstimator(device=device, dtype=torch.float32)
    fit_kwargs = {
        "d": None if args.auto_d else args.d,
        "clamp_range": (args.clamp_min, args.clamp_max),
    }
    if args.max_components is not None:
        fit_kwargs["max_components"] = args.max_components
    if args.auto_d:
        fit_kwargs["eig_drop_ratio"] = args.eig_drop_ratio
        fit_kwargs["eig_abs_floor"] = args.eig_abs_floor

    if args.load_projector:
        print(f"[INFO] Loading projector from {args.load_projector}")
        ss = load_projector(args.load_projector, device)
    else:
        print("[INFO] Fitting safe subspace…")
        ss = est.fit(
            clean_obs_iter(env_clean, trojan_agent, args.samples, device=device, include=args.include_stack),
            **fit_kwargs,
        )
        if args.save_projector:
            save_projector(args.save_projector, ss)
            print(f"[INFO] Saved projector to {args.save_projector}")
    print(f"[INFO] Subspace fitted: D={ss.D}, d={ss.d}")

    # 4) Wrap the agent with the projector
    projector = StateProjector(ss).to(device).eval()
    sanitized_agent = SanitizedAgentAdapter(trojan_agent, projector).eval()

    # 5) Optional: Evaluate on CLEAN env (sanitized vs original)
    if args.eval_episodes > 0:
        rets_orig = eval_policy(env_clean, trojan_agent, args.eval_episodes, device, desc="clean/original")
        rets_sani = eval_policy(env_clean, sanitized_agent, args.eval_episodes, device, desc="clean/sanitized")
        if args.save_metrics:
            append_jsonl(args.save_metrics, {
                "split": "clean/original",
                "episodes": args.eval_episodes,
                "mean": float(np.mean(rets_orig)),
                "std": float(np.std(rets_orig)),
                "D": int(ss.D),
                "d": int(ss.d),
                "samples": int(args.samples),
                "env": args.env,
                "seed": int(args.seed),
            })
            append_jsonl(args.save_metrics, {
                "split": "clean/sanitized",
                "episodes": args.eval_episodes,
                "mean": float(np.mean(rets_sani)),
                "std": float(np.std(rets_sani)),
                "D": int(ss.D),
                "d": int(ss.d),
                "samples": int(args.samples),
                "env": args.env,
                "seed": int(args.seed),
            })

    # 6) Triggered eval (inline poisoning) — no wrapper needed
    if args.do_trigger_eval:
        # Build trigger function in RAW space
        pattern = Single_Stacked_Img_Pattern((4, 84, 84), (args.trigger_size, args.trigger_size), (0, 0),
                                             checker=args.checker).to(device)
        trigger_fn = lambda x: ImagePoison(pattern, args.clamp_min, args.clamp_max)(x)

        def eval_policy_trigger_inline(env, agent_like, episodes: int, device: torch.device, poisoning_rate: float,
                                       desc: str = ""):
            rets = []
            for ep in range(episodes):
                obs = _reset_env(env)
                done, ep_ret = False, 0.0
                while not done:
                    x = as_tensor_chw_uint(obs).to(device)  # [C,H,W] RAW
                    if np.random.random() < poisoning_rate:
                        x = trigger_fn(x)
                    x = x.unsqueeze(0)
                    with torch.no_grad():
                        a, _, _, _ = agent_like.get_action_and_value(x)
                    obs, r, done, info = _step_env(env, a.item())
                    ep_ret += r
                rets.append(ep_ret)
            print(
                f"[EVAL-TRIGGER]{' ' + desc if desc else ''} episodes={episodes}  mean={np.mean(rets):.2f}  std={np.std(rets):.2f}")
            return rets

        # Evaluate both original and sanitized under trigger
        rets_t_orig = eval_policy_trigger_inline(env_clean, trojan_agent, max(1, args.eval_episodes or 5), device,
                                                 args.poisoning_rate, desc="original")
        rets_t_sani = eval_policy_trigger_inline(env_clean, sanitized_agent, max(1, args.eval_episodes or 5), device,
                                                 args.poisoning_rate, desc="sanitized")
        if args.save_metrics:
            append_jsonl(args.save_metrics, {
                "split": "trigger/original",
                "episodes": int(max(1, args.eval_episodes or 5)),
                "mean": float(np.mean(rets_t_orig)),
                "std": float(np.std(rets_t_orig)),
                "D": int(ss.D),
                "d": int(ss.d),
                "samples": int(args.samples),
                "env": args.env,
                "seed": int(args.seed),
                "poisoning_rate": float(args.poisoning_rate),
            })
            append_jsonl(args.save_metrics, {
                "split": "trigger/sanitized",
                "episodes": int(max(1, args.eval_episodes or 5)),
                "mean": float(np.mean(rets_t_sani)),
                "std": float(np.std(rets_t_sani)),
                "D": int(ss.D),
                "d": int(ss.d),
                "samples": int(args.samples),
                "env": args.env,
                "seed": int(args.seed),
                "poisoning_rate": float(args.poisoning_rate),
            })

    print("[DONE] Projection ready. Use `sanitized_agent` for acting in your triggered env when you add it.")


if __name__ == "__main__":
    main()
