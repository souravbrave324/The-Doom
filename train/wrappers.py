"""Training-only Gymnasium wrappers for dense reward shaping, aux targets, and curriculum WAD sampling.

NEVER imported by submission/agent.py. Uses privileged_state() strictly for reward shaping
and monitoring during training.
"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import vizdoom as vd
from gymnasium import spaces

from doom_env import make_env
from doom_env import spaces as S
from doom_env.mapinfo import load_mapinfo
from doom_env.sensors import RAY_DOOR, RAY_LOCKED, RAY_ENEMY, KeyTracker, LevelGeometry, Sensors
from doom_env.wrapper import DEFAULT_EPISODE_TICKS, GAME_VARIABLES, IWAD, SKILL, DoomEnv


def spatial_observation_space() -> spaces.Dict:
    """Observation space without pixel buffers (rgb/depth) for fast multi-worker IPC."""
    full = S.observation_space()
    return spaces.Dict(
        {
            "vitals": full["vitals"],
            "rays": full["rays"],
            "visible_entities": full["visible_entities"],
        }
    )


def strip_pixels(obs: dict) -> dict:
    return {
        "vitals": obs["vitals"],
        "rays": obs["rays"],
        "visible_entities": obs["visible_entities"],
    }


class FastTrainingDoomEnv(DoomEnv):
    """Training-only fast DoomEnv variant (same sensors/logic, disables unused depth/sprite rendering).

    Permitted during training by docs/RULES.md §2; evaluation uses unmodified DoomEnv.
    """

    def __init__(
        self,
        wad: str | os.PathLike,
        map_name: str = "MAP01",
        *,
        episode_ticks: int = DEFAULT_EPISODE_TICKS,
    ):
        self.observation_space = S.observation_space()
        self.action_space = S.action_space()
        self.render_mode = None
        self.wad = str(Path(wad).resolve())
        self.map_name = map_name.upper()
        self.episode_ticks = int(episode_ticks)
        self.render_buffers = False
        self.mapinfo = load_mapinfo(self.wad, self.map_name)

        g = vd.DoomGame()
        g.set_doom_game_path(IWAD)
        g.set_doom_scenario_path(self.wad)
        g.set_doom_map(self.map_name)
        g.set_doom_skill(SKILL)
        g.set_mode(vd.Mode.PLAYER)
        g.set_window_visible(False)
        g.set_sound_enabled(False)
        g.set_console_enabled(False)
        g.set_screen_resolution(vd.ScreenResolution.RES_160X120)
        g.set_screen_format(vd.ScreenFormat.GRAY8)
        g.set_depth_buffer_enabled(False)
        g.set_render_hud(False)
        g.set_render_crosshair(False)
        g.set_render_weapon(False)
        g.set_render_decals(False)
        g.set_render_particles(False)
        g.set_render_messages(False)
        g.set_render_corpses(False)
        g.set_objects_info_enabled(True)
        g.set_sectors_info_enabled(True)
        g.set_available_buttons([getattr(vd.Button, b) for b in S.BUTTONS])
        g.set_available_game_variables(GAME_VARIABLES)
        g.set_episode_timeout(self.episode_ticks)
        g.init()
        self._game = g

        self._geometry: LevelGeometry | None = None
        self._sensors: Sensors | None = None
        self._keys = KeyTracker()
        self._last_obs: dict | None = None
        self._last_vars: np.ndarray | None = None
        self._last_sectors = None
        self._exit_reached = False
        self._closed = False


class DoomTrainingWrapper(gym.Wrapper):
    """Wraps DoomEnv with dense spatial/progression/combat shaping and 3 auxiliary supervision targets."""

    def __init__(
        self,
        env: gym.Env,
        *,
        cell_bonus: float = 0.05,
        door_use_bonus: float = 0.03,
        stall_penalty: float = 0.0,
    ):
        super().__init__(env)
        self.observation_space = spatial_observation_space()
        self.cell_bonus = cell_bonus
        self.door_use_bonus = door_use_bonus
        self.stall_penalty = stall_penalty

        self._visited_cells: set[tuple[int, int]] = set()
        self._opened_doors: set[int] = set()
        self._opened_locked: set[int] = set()
        self._prev_keys_sum = 0.0
        self._prev_kills = 0
        self._prev_damage = 0.0
        self._prev_items = 0
        self._prev_ammo = 0.0
        self._prev_ceil: np.ndarray | None = None
        self._prev_action = np.zeros(len(S.BUTTONS), dtype=np.int8)
        self._key_seen = False
        self._locked_seen = False
        self._post_key_locked_bonus = False
        self._pos_history: list[tuple[float, float]] = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        st = self.env.unwrapped.privileged_state()
        self._visited_cells = {(int(st["x"] // 96), int(st["y"] // 96))}
        self._opened_doors = set()
        self._opened_locked = set()
        self._prev_keys_sum = float(np.sum(st["keys"]))
        self._prev_kills = int(st["kills"])
        self._prev_damage = float(st["damage_taken"])
        self._prev_items = int(st["items"])
        self._prev_ammo = float(obs["vitals"][S.VITALS_INDEX["selected_weapon_ammo"]])
        self._prev_ceil = None if st["sector_ceiling"] is None else st["sector_ceiling"].copy()
        self._prev_action = np.zeros(len(S.BUTTONS), dtype=np.int8)
        self._key_seen = False
        self._locked_seen = False
        self._post_key_locked_bonus = False
        self._pos_history = [(st["x"], st["y"])]

        s_obs = strip_pixels(obs)
        aux = self._compute_aux_targets(s_obs)
        info["aux_targets"] = aux
        info["ep_metrics"] = self._metrics(st, info)
        return s_obs, info

    def step(self, action):
        act_arr = np.asarray(action, dtype=np.int8)
        obs_prev = self.env.unwrapped._last_obs
        obs, _, terminated, truncated, info = self.env.step(act_arr)
        st = self.env.unwrapped.privileged_state()
        s_obs = strip_pixels(obs)

        reward = -0.004  # step penalty

        # 1. Exploration: +cell_bonus per new 96x96 grid cell
        cell = (int(st["x"] // 96), int(st["y"] // 96))
        if cell not in self._visited_cells:
            self._visited_cells.add(cell)
            reward += self.cell_bonus

        # Optional anti-stall penalty over 8 decisions
        self._pos_history.append((st["x"], st["y"]))
        if self.stall_penalty > 0.0 and len(self._pos_history) > 8:
            ox, oy = self._pos_history[-9]
            if ((st["x"] - ox) ** 2 + (st["y"] - oy) ** 2) < 16.0**2:
                reward -= self.stall_penalty

        # 2. Keycard pickup (+3.0 per key) & post-keycard locked-door sighting
        keys_sum = float(np.sum(st["keys"]))
        if keys_sum > self._prev_keys_sum:
            reward += 3.0 * (keys_sum - self._prev_keys_sum)
        self._prev_keys_sum = keys_sum
        if keys_sum > 0.5 and not self._post_key_locked_bonus:
            if np.any(s_obs["rays"]["hit_type"] == RAY_LOCKED):
                self._post_key_locked_bonus = True
                reward += 0.6

        # 3. Doors & locked doors opening
        ceil = st["sector_ceiling"]
        dlock = st["door_lock"]
        if ceil is not None and dlock is not None and self._prev_ceil is not None:
            rising = ceil > (self._prev_ceil + 1.0)
            for sid in np.flatnonzero(rising & (dlock >= 0)):
                sid_i = int(sid)
                if dlock[sid_i] > 0 and sid_i not in self._opened_locked:
                    self._opened_locked.add(sid_i)
                    reward += 4.0
                elif dlock[sid_i] == 0 and sid_i not in self._opened_doors:
                    self._opened_doors.add(sid_i)
                    reward += 0.5
        if ceil is not None:
            self._prev_ceil = ceil.copy()

        # Small USE discovery reward when pressing USE on button-down right in front of a closed door
        if self.door_use_bonus > 0.0 and obs_prev is not None and act_arr[7] == 1 and self._prev_action[7] == 0:
            front_idx = (62, 63, 0, 1, 2)
            r_hit = obs_prev["rays"]["hit_type"][list(front_idx)]
            r_dist = obs_prev["rays"]["distance"][list(front_idx)]
            near_door = np.any(((r_hit == RAY_DOOR) | ((r_hit == RAY_LOCKED) & (keys_sum > 0.5))) & (r_dist < (72.0 / S.RAY_RANGE)))
            if near_door:
                reward += self.door_use_bonus

        # 4. Combat: +0.8 per kill + aimed shot engagement bonus + -0.02 per HP lost
        kills = int(st["kills"])
        if kills > self._prev_kills:
            reward += 0.8 * (kills - self._prev_kills)
        self._prev_kills = kills

        ammo_now = float(s_obs["vitals"][S.VITALS_INDEX["selected_weapon_ammo"]])
        if act_arr[6] == 1 and ammo_now < self._prev_ammo and obs_prev is not None:
            # Reward firing while enemy is centered in front cone
            e_mask = obs_prev["visible_entities"]["mask"] > 0
            e_type = obs_prev["visible_entities"]["entity_type"]
            e_feat = obs_prev["visible_entities"]["features"]
            is_enemy = e_mask & ((e_type == 1) | (e_type == 2) | (e_type == 3))
            if np.any(is_enemy & (np.abs(e_feat[:, 1]) < 0.28) & (e_feat[:, 2] > 0.9)):
                reward += 0.05
        self._prev_ammo = ammo_now

        dmg_taken = float(st["damage_taken"])
        if dmg_taken > self._prev_damage:
            reward -= 0.02 * (dmg_taken - self._prev_damage)
        self._prev_damage = dmg_taken

        # Item pickup micro-bonus (health/armor/ammo)
        if int(st["items"]) > 0 and hasattr(self, "_prev_items"):
            if int(st["items"]) > self._prev_items:
                reward += 0.05 * (int(st["items"]) - self._prev_items)
        self._prev_items = int(st["items"])

        # 5. Terminal exit / death
        if info.get("exit_reached", False):
            reward += 15.0
        elif st["dead"]:
            reward -= 1.0

        self._prev_action = act_arr.copy()
        info["aux_targets"] = self._compute_aux_targets(s_obs)
        info["ep_metrics"] = self._metrics(st, info)
        return s_obs, float(reward), terminated, truncated, info

    def _compute_aux_targets(self, obs: dict) -> np.ndarray:
        r_hit = obs["rays"]["hit_type"]
        r_dist = obs["rays"]["distance"]
        e_mask = obs["visible_entities"]["mask"] > 0
        e_type = obs["visible_entities"]["entity_type"]
        e_feat = obs["visible_entities"]["features"]

        ray_enemy_near = bool(np.any((r_hit == RAY_ENEMY) & (r_dist < 0.45)))
        ent_enemy_near = bool(np.any(e_mask & ((e_type == 1) | (e_type == 2) | (e_type == 3)) & (e_feat[:, 0] < 0.45)))
        threat_near = float(ray_enemy_near or ent_enemy_near)

        if np.any(e_mask & (e_type == 9)) or np.sum(obs["vitals"][8:14]) > 0.5:
            self._key_seen = True
        if np.any(r_hit == RAY_LOCKED):
            self._locked_seen = True

        return np.array([threat_near, float(self._key_seen), float(self._locked_seen)], dtype=np.float32)

    def _metrics(self, st: dict, info: dict) -> dict:
        return {
            "exit_reached": float(bool(info.get("exit_reached", False))),
            "dead": float(bool(st["dead"])),
            "keycards": float(np.sum(st["keys"]) > 0.5),
            "doors_opened": float(len(self._opened_doors)),
            "locked_opened": float(len(self._opened_locked)),
            "kills": float(st["kills"]),
            "cells": float(len(self._visited_cells)),
            "tic": float(st["tic"]),
        }


class CurriculumWadEnv(gym.Env):
    """Multi-map environment that samples a new WAD every `wad_episodes` resets for high SPS."""

    def __init__(
        self,
        wad_specs: list[tuple[str, str, int]],  # [(wad_path, map_name, episode_ticks), ...]
        *,
        weights: list[float] | None = None,
        seed: int = 0,
        wad_episodes: int = 16,
        cell_bonus: float = 0.05,
        door_use_bonus: float = 0.03,
        stall_penalty: float = 0.0,
    ):
        super().__init__()
        self.wad_specs = wad_specs
        self.weights = weights
        self.rng = random.Random(seed)
        self.wad_episodes = max(1, wad_episodes)
        self.cell_bonus = cell_bonus
        self.door_use_bonus = door_use_bonus
        self.stall_penalty = stall_penalty
        self.observation_space = spatial_observation_space()
        self.action_space = S.action_space()

        self._cur_spec: tuple[str, str, int] | None = None
        self._env: DoomTrainingWrapper | None = None
        self._ep_on_wad = 0

    def _pick_spec(self) -> tuple[str, str, int]:
        if self.weights is not None:
            return self.rng.choices(self.wad_specs, weights=self.weights, k=1)[0]
        return self.rng.choice(self.wad_specs)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng.seed(seed)
        if self._env is None or self._ep_on_wad >= self.wad_episodes:
            spec = self._pick_spec()
            if self._env is None or spec != self._cur_spec:
                if self._env is not None:
                    self._env.close()
                wad_path, map_name, ep_ticks = spec
                base = FastTrainingDoomEnv(wad_path, map_name, episode_ticks=ep_ticks)
                self._env = DoomTrainingWrapper(
                    base,
                    cell_bonus=self.cell_bonus,
                    door_use_bonus=self.door_use_bonus,
                    stall_penalty=self.stall_penalty,
                )
                self._cur_spec = spec
            self._ep_on_wad = 0
        self._ep_on_wad += 1
        sub_seed = self.rng.randint(0, 2**30)
        return self._env.reset(seed=sub_seed)

    def step(self, action):
        assert self._env is not None
        return self._env.step(action)

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None


def build_stage_wad_specs(wad_root: str | Path, stage: str) -> tuple[list[tuple[str, str, int]], list[float] | None]:
    """Return [(wad_path, map_name, episode_ticks)] and sampling weights for curriculum stage."""
    root = Path(wad_root)
    specs: list[tuple[str, str, int]] = []
    weights: list[float] = []

    if stage == "1":
        for p in sorted((root / "tier1").glob("*.wad")):
            specs.append((str(p), "MAP01", 2100))
            weights.append(1.0)
    elif stage == "2":
        for p in sorted((root / "tier2").glob("*.wad")):
            specs.append((str(p), "MAP01", 2400))
            weights.append(0.80)
        for p in sorted((root / "tier1").glob("*.wad")):
            specs.append((str(p), "MAP01", 2100))
            weights.append(0.20)
    elif stage == "3":
        for p in sorted((root / "tier3").glob("*.wad")):
            specs.append((str(p), "MAP01", 3200))
            weights.append(0.80)
        for p in sorted((root / "tier2").glob("*.wad")):
            specs.append((str(p), "MAP01", 2400))
            weights.append(0.20)
    elif stage == "4":
        t1 = sorted((root / "tier1").glob("*.wad"))
        t2 = sorted((root / "tier2").glob("*.wad"))
        t3 = sorted((root / "tier3").glob("*.wad"))
        t4 = sorted((root / "tier4").glob("*.wad"))
        for p in t1:
            specs.append((str(p), "MAP01", 2100))
            weights.append(0.15 / max(1, len(t1)))
        for p in t2:
            specs.append((str(p), "MAP01", 2400))
            weights.append(0.20 / max(1, len(t2)))
        for p in t3:
            specs.append((str(p), "MAP01", 3200))
            weights.append(0.30 / max(1, len(t3)))
        for p in t4:
            specs.append((str(p), "MAP01", 3600))
            weights.append(0.30 / max(1, len(t4)))
        # Mix 5% standard Freedoom 2 maps (MAP01..MAP06) for domain generalization without diluting exits
        import vizdoom as vd
        iwad = os.path.join(os.path.dirname(vd.__file__), "freedoom2.wad")
        if Path(iwad).is_file():
            fd_maps = ["MAP01", "MAP02", "MAP03", "MAP04", "MAP05", "MAP06"]
            for m in fd_maps:
                specs.append((iwad, m, 3600))
                weights.append(0.05 / len(fd_maps))
    elif stage == "all":
        for idx, ticks in ((1, 2100), (2, 2400), (3, 3200), (4, 3600)):
            for p in sorted((root / f"tier{idx}").glob("*.wad")):
                specs.append((str(p), "MAP01", ticks))
                weights.append(1.0)
    else:
        raise ValueError(f"Invalid stage: {stage}")

    if not specs:
        raise RuntimeError(f"No WADs found in {root} for stage {stage}")
    return specs, weights


def make_curriculum_worker(
    wad_specs: list[tuple[str, str, int]],
    weights: list[float] | None,
    seed: int,
    wad_episodes: int,
    cell_bonus: float,
    door_use_bonus: float,
    stall_penalty: float,
) -> gym.Env:
    """Picklable worker factory kept in train/wrappers.py so workers avoid importing torch."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    return CurriculumWadEnv(
        wad_specs=wad_specs,
        weights=weights,
        seed=seed,
        wad_episodes=wad_episodes,
        cell_bonus=cell_bonus,
        door_use_bonus=door_use_bonus,
        stall_penalty=stall_penalty,
    )
