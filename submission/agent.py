"""Evaluation entry point. The scorer instantiates DoomAgent from this file.

Contract: docs/CONTRACT.md §6.
Hardened single-thread inference wrapper around DoomSpatialMemoryNet with:
  - Strict single-thread enforcement (OMP/MKL/OPENBLAS/NUMEXPR=1 + torch.set_num_threads(1))
  - Preallocated observation buffers and 2 warm-up passes in __init__
  - NaN / finite-output / exception failsafe in act() so evaluation never crashes
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

SUBMISSION_DIR = Path(__file__).resolve().parent
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from model import (  # noqa: E402
    EGO_CHANNELS,
    EGO_GRID_SIZE,
    ENTITY_FEAT_DIM,
    MAX_ENTITIES,
    NUM_ACTIONS,
    NUM_RAYS,
    VITALS_DIM,
    AgentMemoryState,
    DoomSpatialMemoryNet,
)

WEIGHTS_PRIMARY = SUBMISSION_DIR / "weights.pt"
WEIGHTS_BACKUP = SUBMISSION_DIR / "weights" / "model.pt"


class DoomAgent:
    def __init__(self, observation_space, action_space, device: str):
        """Called once. Load weights once onto `device` and warm up kernels."""
        torch.set_num_threads(1)
        self.observation_space = observation_space
        self.action_space = action_space
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        self.model = DoomSpatialMemoryNet().to(self.device).eval()
        weights_path = WEIGHTS_PRIMARY if WEIGHTS_PRIMARY.is_file() else WEIGHTS_BACKUP
        if weights_path.is_file():
            state = torch.load(weights_path, map_location=self.device, weights_only=True)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            self.model.load_state_dict(state)
        else:
            print("WARNING: trained weights not found, using initialized parameters", file=sys.stderr)

        # Preallocated reusable tensors (packed host buffers -> 2 transfers instead of 8 syncs)
        self._host_f = np.zeros(VITALS_DIM + NUM_RAYS + MAX_ENTITIES * ENTITY_FEAT_DIM + MAX_ENTITIES, dtype=np.float32)
        self._host_i = np.zeros(NUM_RAYS + MAX_ENTITIES, dtype=np.int64)
        self._gpu_f = torch.zeros(self._host_f.shape, dtype=torch.float32, device=self.device)
        self._gpu_i = torch.zeros(self._host_i.shape, dtype=torch.long, device=self.device)

        o_v = 0
        o_rd = o_v + VITALS_DIM
        o_ef = o_rd + NUM_RAYS
        o_em = o_ef + MAX_ENTITIES * ENTITY_FEAT_DIM
        self._obs_t = {
            "vitals": self._gpu_f[o_v:o_rd].view(1, VITALS_DIM),
            "rays": {
                "distance": self._gpu_f[o_rd:o_ef].view(1, NUM_RAYS),
                "hit_type": self._gpu_i[:NUM_RAYS].view(1, NUM_RAYS),
            },
            "visible_entities": {
                "features": self._gpu_f[o_ef:o_em].view(1, MAX_ENTITIES, ENTITY_FEAT_DIM),
                "entity_type": self._gpu_i[NUM_RAYS:].view(1, MAX_ENTITIES),
                "mask": self._gpu_f[o_em:].view(1, MAX_ENTITIES),
            },
        }

        self._mem = AgentMemoryState(
            gru_hidden=torch.zeros((1, 1, 256), dtype=torch.float32, device=self.device),
            ego_map=torch.zeros((1, EGO_CHANNELS, EGO_GRID_SIZE, EGO_GRID_SIZE), dtype=torch.float32, device=self.device),
            latch=torch.zeros((1, 2), dtype=torch.float32, device=self.device),
            prev_action=torch.zeros((1, NUM_ACTIONS), dtype=torch.float32, device=self.device),
        )
        self._rng = torch.Generator(device=self.device).manual_seed(0)
        self._warned = False

        # Two dummy warm-up forward passes inside the 60s init window
        with torch.inference_mode():
            for _ in range(2):
                self.model.forward_step(self._obs_t, self._mem)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.reset()

    def reset(self) -> None:
        """Called before every map. Zero out GRU hidden state, ego-map, latch, and action history."""
        self._mem.gru_hidden.zero_()
        self._mem.ego_map.zero_()
        self._mem.latch.zero_()
        self._mem.prev_action.zero_()

    def _copy_obs(self, observation: dict) -> None:
        hf = self._host_f
        hi = self._host_i
        hf[0:VITALS_DIM] = np.nan_to_num(np.asarray(observation["vitals"], dtype=np.float32), nan=0.0, posinf=200.0, neginf=-200.0)
        r = observation["rays"]
        o_rd = VITALS_DIM
        o_ef = o_rd + NUM_RAYS
        o_em = o_ef + MAX_ENTITIES * ENTITY_FEAT_DIM
        hf[o_rd:o_ef] = np.nan_to_num(np.asarray(r["distance"], dtype=np.float32), nan=1.0, posinf=1.0, neginf=0.0)
        hi[:NUM_RAYS] = np.asarray(r["hit_type"], dtype=np.int64)
        e = observation["visible_entities"]
        hf[o_ef:o_em] = np.nan_to_num(np.asarray(e["features"], dtype=np.float32), nan=0.0, posinf=10.0, neginf=-10.0).reshape(-1)
        hf[o_em:] = np.asarray(e["mask"], dtype=np.float32)
        hi[NUM_RAYS:] = np.asarray(e["entity_type"], dtype=np.int64)

        self._gpu_f.copy_(torch.from_numpy(hf))
        self._gpu_i.copy_(torch.from_numpy(hi))

    def act(self, observation: dict) -> np.ndarray:
        """Return a valid MultiBinary(9) int8 array using trained neural parameters only."""
        try:
            with torch.inference_mode():
                self._copy_obs(observation)
                logits, _, _, next_mem = self.model.forward_step(self._obs_t, self._mem)
                if not (torch.isfinite(logits).all() and torch.isfinite(next_mem.gru_hidden).all()):
                    raise FloatingPointError("non-finite neural network output")

                probs = torch.sigmoid(logits[0])
                act_t = torch.rand(NUM_ACTIONS, device=self.device, generator=self._rng) < probs

                self._mem.gru_hidden.copy_(next_mem.gru_hidden)
                self._mem.ego_map.copy_(next_mem.ego_map)
                self._mem.latch.copy_(next_mem.latch)
                self._mem.prev_action[0].copy_(act_t.to(dtype=torch.float32))

                return act_t.to(dtype=torch.int8).cpu().numpy()
        except Exception as exc:
            if not self._warned:
                print(f"act() failsafe triggered: {exc!r}", file=sys.stderr)
                self._warned = True
            self.reset()
            return np.zeros(NUM_ACTIONS, dtype=np.int8)
