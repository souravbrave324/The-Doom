"""Vectorized Recurrent PPO + Multi-Task Auxiliary Loss Trainer.

Runs 12 parallel CPU ViZDoom workers via AsyncVectorEnv and trains DoomSpatialMemoryNet
on CUDA with FP16 AMP, truncated BPTT (chunk_len=32), GAE, and entropy annealing.
"""

from __future__ import annotations

import argparse
import os
import time
from collections import deque
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli

from submission.model import (
    EGO_CHANNELS,
    EGO_GRID_SIZE,
    ENTITY_FEAT_DIM,
    MAX_ENTITIES,
    NUM_ACTIONS,
    NUM_RAYS,
    VITALS_DIM,
    AgentMemoryState,
    DoomSpatialMemoryNet,
    count_parameters,
)
from train.wrappers import build_stage_wad_specs, make_curriculum_worker


def _obs_to_torch(obs: dict, device: torch.device) -> dict:
    return {
        "vitals": torch.as_tensor(obs["vitals"], dtype=torch.float32, device=device),
        "rays": {
            "distance": torch.as_tensor(obs["rays"]["distance"], dtype=torch.float32, device=device),
            "hit_type": torch.as_tensor(obs["rays"]["hit_type"], dtype=torch.long, device=device),
        },
        "visible_entities": {
            "features": torch.as_tensor(obs["visible_entities"]["features"], dtype=torch.float32, device=device),
            "entity_type": torch.as_tensor(obs["visible_entities"]["entity_type"], dtype=torch.long, device=device),
            "mask": torch.as_tensor(obs["visible_entities"]["mask"], dtype=torch.float32, device=device),
        },
    }


def _extract_aux_and_metrics(infos: dict, num_envs: int) -> tuple[np.ndarray, list[dict]]:
    aux = np.zeros((num_envs, 3), dtype=np.float32)
    ep_done_metrics: list[dict] = []
    if "aux_targets" in infos:
        aux = np.asarray(infos["aux_targets"], dtype=np.float32).copy()
    if "final_info" in infos:
        has_final = np.asarray(infos.get("_final_info", np.zeros(num_envs, dtype=bool)), dtype=bool)
        fi = infos["final_info"]
        if isinstance(fi, dict):
            if "aux_targets" in fi:
                fi_aux = np.asarray(fi["aux_targets"], dtype=np.float32)
                aux[has_final] = fi_aux[has_final]
            ep_m = fi.get("ep_metrics")
            if isinstance(ep_m, dict):
                keys = ("exit_reached", "keycards", "kills", "tic", "doors_opened", "locked_opened")
                for i in np.flatnonzero(has_final):
                    ep_done_metrics.append({k: float(ep_m[k][i]) for k in keys if k in ep_m})
    return aux, ep_done_metrics


def _stack_obs_list(obs_list: list[dict], device: torch.device) -> dict:
    return {
        "vitals": torch.as_tensor(np.stack([o["vitals"] for o in obs_list]), dtype=torch.float32, device=device),
        "rays": {
            "distance": torch.as_tensor(np.stack([o["rays"]["distance"] for o in obs_list]), dtype=torch.float32, device=device),
            "hit_type": torch.as_tensor(np.stack([o["rays"]["hit_type"] for o in obs_list]), dtype=torch.long, device=device),
        },
        "visible_entities": {
            "features": torch.as_tensor(np.stack([o["visible_entities"]["features"] for o in obs_list]), dtype=torch.float32, device=device),
            "entity_type": torch.as_tensor(np.stack([o["visible_entities"]["entity_type"] for o in obs_list]), dtype=torch.long, device=device),
            "mask": torch.as_tensor(np.stack([o["visible_entities"]["mask"] for o in obs_list]), dtype=torch.float32, device=device),
        },
    }


class RNDModule(nn.Module):
    """Lightweight Random Network Distillation for intrinsic spatial exploration curiosity."""

    def __init__(self, in_dim: int = VITALS_DIM + 2 * NUM_RAYS):
        super().__init__()
        self.target = nn.Sequential(nn.Linear(in_dim, 128), nn.GELU(), nn.Linear(128, 64))
        for p in self.target.parameters():
            p.requires_grad = False
        self.predictor = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 64),
        )

    @staticmethod
    def featurize(obs_t: dict) -> torch.Tensor:
        vit = obs_t["vitals"].clone()
        vit[..., :8] = (vit[..., :8] / 100.0).clamp(0.0, 2.5)
        vit[..., 14:17] = (vit[..., 14:17] / 20.0).clamp(-2.5, 2.5)
        rd = obs_t["rays"]["distance"]
        rh = obs_t["rays"]["hit_type"].float() / 5.0
        return torch.cat([vit, rd, rh], dim=-1)

    def forward(self, obs_t: dict) -> torch.Tensor:
        x = self.featurize(obs_t)
        with torch.no_grad():
            t = self.target(x)
        p = self.predictor(x)
        return ((p - t) ** 2).mean(dim=-1)


def save_checkpoint(
    ckpt_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    score: float,
    recent_paths: deque[Path],
    is_best: bool,
) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
        "score": score,
    }
    step_path = ckpt_dir / f"ckpt_{global_step:09d}.pt"
    torch.save(payload, step_path)
    recent_paths.append(step_path)
    while len(recent_paths) > 3:
        old = recent_paths.popleft()
        if old.exists() and old.name != "best_model.pt":
            old.unlink(missing_ok=True)
    if is_best:
        torch.save(payload, ckpt_dir / "best_model.pt")
        # Weights-only file for fast inference loading
        torch.save(model.state_dict(), ckpt_dir / "best_weights.pt")


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    use_amp = device.type == "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    wad_specs, weights = build_stage_wad_specs(args.wad_dir, args.stage)
    env_fns = [
        partial(
            make_curriculum_worker,
            wad_specs,
            weights,
            args.seed + i * 97,
            args.wad_episodes,
            args.cell_bonus,
            args.door_use_bonus,
            args.stall_penalty,
        )
        for i in range(args.num_envs)
    ]
    envs = gym.vector.AsyncVectorEnv(
        env_fns,
        shared_memory=False,
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )

    model = DoomSpatialMemoryNet().to(device)
    rnd = RNDModule().to(device)
    print(f"[Stage {args.stage}] Device={device} | Params={count_parameters(model):,} | Workers={args.num_envs} | WADs={len(wad_specs)}", flush=True)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(rnd.parameters()),
        lr=args.lr,
        eps=1e-5,
        weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rnd_ema_var = 1.0

    if args.init_from and Path(args.init_from).is_file():
        ckpt = torch.load(args.init_from, map_location=device, weights_only=True)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state_dict)
        print(f"Loaded initial weights from {args.init_from}", flush=True)

    n_envs = args.num_envs
    n_steps = args.n_steps
    seq_len = args.seq_len
    assert n_steps % seq_len == 0, "n_steps must be divisible by seq_len"
    chunks_per_env = n_steps // seq_len
    total_chunks = n_envs * chunks_per_env
    chunks_per_mb = max(1, args.batch_size // seq_len)

    # Preallocated CPU rollout buffers (fast, zero GPU VRAM bloat)
    buf_vitals = torch.zeros((n_steps, n_envs, VITALS_DIM), dtype=torch.float32)
    buf_r_dist = torch.zeros((n_steps, n_envs, NUM_RAYS), dtype=torch.float32)
    buf_r_hit = torch.zeros((n_steps, n_envs, NUM_RAYS), dtype=torch.int64)
    buf_e_feat = torch.zeros((n_steps, n_envs, MAX_ENTITIES, ENTITY_FEAT_DIM), dtype=torch.float32)
    buf_e_type = torch.zeros((n_steps, n_envs, MAX_ENTITIES), dtype=torch.int64)
    buf_e_mask = torch.zeros((n_steps, n_envs, MAX_ENTITIES), dtype=torch.float32)

    buf_actions = torch.zeros((n_steps, n_envs, NUM_ACTIONS), dtype=torch.float32)
    buf_prev_actions = torch.zeros((n_steps, n_envs, NUM_ACTIONS), dtype=torch.float32)
    buf_logprobs = torch.zeros((n_steps, n_envs), dtype=torch.float32)
    buf_rewards = torch.zeros((n_steps, n_envs), dtype=torch.float32)
    buf_dones = torch.zeros((n_steps, n_envs), dtype=torch.float32)
    buf_values = torch.zeros((n_steps, n_envs), dtype=torch.float32)
    buf_aux = torch.zeros((n_steps, n_envs, 3), dtype=torch.float32)

    # Chunk-start recurrent states (chunks_per_env, n_envs, ...)
    buf_init_h = torch.zeros((chunks_per_env, n_envs, 256), dtype=torch.float32)
    buf_init_ego = torch.zeros((chunks_per_env, n_envs, EGO_CHANNELS, EGO_GRID_SIZE, EGO_GRID_SIZE), dtype=torch.float32)
    buf_init_latch = torch.zeros((chunks_per_env, n_envs, 2), dtype=torch.float32)

    obs_np, _ = envs.reset(seed=args.seed)
    mem = model.init_memory(n_envs, device)

    ep_rewards = np.zeros(n_envs, dtype=np.float64)
    recent_ep_rewards: deque[float] = deque(maxlen=100)
    recent_exits: deque[float] = deque(maxlen=100)
    recent_keys: deque[float] = deque(maxlen=100)
    recent_kills: deque[float] = deque(maxlen=100)
    recent_tics: deque[float] = deque(maxlen=100)
    recent_doors: deque[float] = deque(maxlen=100)
    recent_locked: deque[float] = deque(maxlen=100)

    ckpt_dir = Path(args.ckpt_dir) / f"stage{args.stage}"
    recent_ckpts: deque[Path] = deque()
    best_score = -1e9
    global_step = 0
    start_time = time.perf_counter()
    max_seconds = args.max_minutes * 60.0 if args.max_minutes > 0 else float("inf")
    last_log_step = 0
    update_idx = 0

    try:
        while global_step < args.total_steps and (time.perf_counter() - start_time) < max_seconds:
            update_idx += 1
            model.eval()
            for step in range(n_steps):
                global_step += n_envs
                if step % seq_len == 0:
                    c_idx = step // seq_len
                    buf_init_h[c_idx] = mem.gru_hidden.squeeze(0).cpu()
                    buf_init_ego[c_idx] = mem.ego_map.cpu()
                    buf_init_latch[c_idx] = mem.latch.cpu()

                obs_t = _obs_to_torch(obs_np, device)
                buf_vitals[step] = obs_t["vitals"].cpu()
                buf_r_dist[step] = obs_t["rays"]["distance"].cpu()
                buf_r_hit[step] = obs_t["rays"]["hit_type"].cpu()
                buf_e_feat[step] = obs_t["visible_entities"]["features"].cpu()
                buf_e_type[step] = obs_t["visible_entities"]["entity_type"].cpu()
                buf_e_mask[step] = obs_t["visible_entities"]["mask"].cpu()
                buf_prev_actions[step] = mem.prev_action.cpu()

                with torch.inference_mode():
                    logits, value, _, next_mem = model.forward_step(obs_t, mem)
                    dist = Bernoulli(logits=logits)
                    action = dist.sample()
                    logprob = dist.log_prob(action).sum(dim=-1)

                act_np = action.cpu().numpy().astype(np.int8)
                buf_actions[step] = action.cpu()
                buf_logprobs[step] = logprob.cpu()
                buf_values[step] = value.float().cpu()

                next_obs_np, rewards_np, terms_np, truncs_np, infos = envs.step(act_np)
                dones_np = np.logical_or(terms_np, truncs_np)
                aux_np, ep_metrics_list = _extract_aux_and_metrics(infos, n_envs)

                # RND intrinsic novelty reward (training-only exploration signal)
                if args.rnd_coef > 0.0:
                    with torch.inference_mode():
                        rnd_err = rnd(obs_t).cpu().numpy()
                    rnd_ema_var = 0.99 * rnd_ema_var + 0.01 * max(1e-6, float(np.var(rnd_err) + np.mean(rnd_err) ** 2))
                    rnd_bonus = np.clip(args.rnd_coef * rnd_err / np.sqrt(rnd_ema_var), 0.0, 0.15).astype(np.float32)
                    rewards_np = rewards_np + rnd_bonus

                # Proper SAME_STEP truncation bootstrapping: add gamma * V(final_obs) when truncated & not terminated
                trunc_only = np.logical_and(truncs_np, np.logical_not(terms_np))
                if np.any(trunc_only) and "final_obs" in infos:
                    has_fo = np.asarray(infos.get("_final_obs", np.zeros(n_envs, dtype=bool)), dtype=bool)
                    t_idx = np.flatnonzero(np.logical_and(trunc_only, has_fo))
                    if len(t_idx) > 0:
                        fo_list = [infos["final_obs"][i] for i in t_idx]
                        fo_t = _stack_obs_list(fo_list, device)
                        t_idx_t = torch.as_tensor(t_idx, dtype=torch.long, device=device)
                        trunc_mem = AgentMemoryState(
                            gru_hidden=next_mem.gru_hidden[:, t_idx_t],
                            ego_map=next_mem.ego_map[t_idx_t],
                            latch=next_mem.latch[t_idx_t],
                            prev_action=action[t_idx_t].float(),
                        )
                        with torch.inference_mode():
                            _, t_vals, _, _ = model.forward_step(fo_t, trunc_mem)
                        rewards_np[t_idx] = rewards_np[t_idx] + args.gamma * t_vals.cpu().numpy()

                buf_rewards[step] = torch.from_numpy(rewards_np.astype(np.float32))
                buf_dones[step] = torch.from_numpy(dones_np.astype(np.float32))
                buf_aux[step] = torch.from_numpy(aux_np)

                ep_rewards += rewards_np
                for i in range(n_envs):
                    if dones_np[i]:
                        recent_ep_rewards.append(float(ep_rewards[i]))
                        ep_rewards[i] = 0.0
                for em in ep_metrics_list:
                    recent_exits.append(em["exit_reached"])
                    recent_keys.append(em["keycards"])
                    recent_kills.append(em["kills"])
                    recent_tics.append(em["tic"])
                    recent_doors.append(em["doors_opened"])
                    recent_locked.append(em["locked_opened"])

                # Update memory & zero out finished envs cleanly
                if np.any(dones_np):
                    keep = torch.from_numpy((~dones_np).astype(np.float32)).to(device)
                    mem = AgentMemoryState(
                        gru_hidden=next_mem.gru_hidden * keep.view(1, n_envs, 1),
                        ego_map=next_mem.ego_map * keep.view(n_envs, 1, 1, 1),
                        latch=next_mem.latch * keep.view(n_envs, 1),
                        prev_action=action.float() * keep.view(n_envs, 1),
                    )
                else:
                    mem = AgentMemoryState(
                        gru_hidden=next_mem.gru_hidden,
                        ego_map=next_mem.ego_map,
                        latch=next_mem.latch,
                        prev_action=action.float(),
                    )
                obs_np = next_obs_np

            # Bootstrap value for GAE
            with torch.inference_mode():
                obs_t = _obs_to_torch(obs_np, device)
                _, next_value, _, _ = model.forward_step(obs_t, mem)
                next_value = next_value.cpu()

            advantages = torch.zeros_like(buf_rewards)
            lastgaelam = torch.zeros(n_envs, dtype=torch.float32)
            for t in reversed(range(n_steps)):
                if t == n_steps - 1:
                    nextnonterminal = 1.0 - torch.zeros(n_envs, dtype=torch.float32)
                    # Note: buf_dones[t] already reflects whether step t ended an episode!
                    nextnonterminal = 1.0 - buf_dones[t]
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - buf_dones[t]
                    nextvalues = buf_values[t + 1]
                delta = buf_rewards[t] + args.gamma * nextvalues * nextnonterminal - buf_values[t]
                lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + buf_values

            # Reshape (n_steps, n_envs, ...) -> (total_chunks, seq_len, ...)
            def to_chunks(x: torch.Tensor) -> torch.Tensor:
                # x: (chunks_per_env * seq_len, n_envs, ...) -> (n_envs, chunks_per_env, seq_len, ...)
                x = x.reshape(chunks_per_env, seq_len, n_envs, *x.shape[2:])
                x = x.permute(2, 0, 1, *range(3, x.ndim))
                return x.reshape(total_chunks, seq_len, *x.shape[3:])

            c_vitals = to_chunks(buf_vitals)
            c_r_dist = to_chunks(buf_r_dist)
            c_r_hit = to_chunks(buf_r_hit)
            c_e_feat = to_chunks(buf_e_feat)
            c_e_type = to_chunks(buf_e_type)
            c_e_mask = to_chunks(buf_e_mask)
            c_actions = to_chunks(buf_actions)
            c_prev_actions = to_chunks(buf_prev_actions)
            c_logprobs = to_chunks(buf_logprobs)
            c_returns = to_chunks(returns)
            c_advs = to_chunks(advantages)
            c_values = to_chunks(buf_values)
            c_dones = to_chunks(buf_dones)
            c_aux = to_chunks(buf_aux)

            c_init_h = buf_init_h.permute(1, 0, 2).reshape(total_chunks, 256)
            c_init_ego = buf_init_ego.permute(1, 0, 2, 3, 4).reshape(total_chunks, EGO_CHANNELS, EGO_GRID_SIZE, EGO_GRID_SIZE)
            c_init_latch = buf_init_latch.permute(1, 0, 2).reshape(total_chunks, 2)

            # Entropy annealing
            progress = min(1.0, max(global_step / max(1, args.total_steps), (time.perf_counter() - start_time) / max(1.0, max_seconds) if max_seconds < float("inf") else 0.0))
            ent_coef = args.ent_start + (args.ent_end - args.ent_start) * progress

            model.train()
            pg_losses, v_losses, aux_losses, ents = [], [], [], []
            perm = np.arange(total_chunks)
            for _ in range(args.n_epochs):
                np.random.shuffle(perm)
                for start in range(0, total_chunks, chunks_per_mb):
                    mb = perm[start : start + chunks_per_mb]
                    obs_mb = {
                        "vitals": c_vitals[mb].to(device),
                        "rays": {"distance": c_r_dist[mb].to(device), "hit_type": c_r_hit[mb].to(device)},
                        "visible_entities": {
                            "features": c_e_feat[mb].to(device),
                            "entity_type": c_e_type[mb].to(device),
                            "mask": c_e_mask[mb].to(device),
                        },
                    }
                    init_mem_mb = AgentMemoryState(
                        gru_hidden=c_init_h[mb].unsqueeze(0).to(device),
                        ego_map=c_init_ego[mb].to(device),
                        latch=c_init_latch[mb].to(device),
                        prev_action=torch.zeros((len(mb), NUM_ACTIONS), dtype=torch.float32, device=device),
                    )
                    dones_mb = c_dones[mb].to(device)
                    pa_mb = c_prev_actions[mb].to(device)
                    act_mb = c_actions[mb].to(device)
                    old_lp_mb = c_logprobs[mb].to(device)
                    ret_mb = c_returns[mb].to(device)
                    adv_mb = c_advs[mb].to(device)
                    old_v_mb = c_values[mb].to(device)
                    aux_mb = c_aux[mb].to(device)

                    adv_mb = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)

                    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                        logits, new_values, aux_logits = model.forward_sequence(obs_mb, dones_mb, pa_mb, init_mem_mb)

                    # Compute losses in FP32 for numerical stability
                    dist = Bernoulli(logits=logits.float())
                    new_lp = dist.log_prob(act_mb).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1).mean()
                    ratio = torch.exp(new_lp - old_lp_mb)

                    pg1 = -adv_mb * ratio
                    pg2 = -adv_mb * torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
                    pg_loss = torch.max(pg1, pg2).mean()

                    v_clipped = old_v_mb + torch.clamp(new_values.float() - old_v_mb, -args.clip_coef, args.clip_coef)
                    v_loss = 0.5 * torch.max((new_values.float() - ret_mb) ** 2, (v_clipped - ret_mb) ** 2).mean()

                    aux_loss = F.binary_cross_entropy_with_logits(aux_logits.float(), aux_mb)
                    obs_rnd = {
                        "vitals": obs_mb["vitals"].reshape(-1, VITALS_DIM),
                        "rays": {
                            "distance": obs_mb["rays"]["distance"].reshape(-1, NUM_RAYS),
                            "hit_type": obs_mb["rays"]["hit_type"].reshape(-1, NUM_RAYS),
                        },
                    }
                    rnd_loss = rnd(obs_rnd).mean() if args.rnd_coef > 0.0 else torch.zeros((), device=device)
                    loss = pg_loss + args.vf_coef * v_loss - ent_coef * entropy + args.aux_coef * aux_loss + 0.5 * rnd_loss

                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    scaler.step(optimizer)
                    scaler.update()

                    pg_losses.append(float(pg_loss.item()))
                    v_losses.append(float(v_loss.item()))
                    aux_losses.append(float(aux_loss.item()))
                    ents.append(float(entropy.item()))

            elapsed = time.perf_counter() - start_time
            sps = global_step / max(1e-6, elapsed)
            if update_idx <= 3 or (global_step - last_log_step) >= args.log_interval or (time.perf_counter() - start_time) >= max_seconds:
                last_log_step = global_step
                mean_r = float(np.mean(recent_ep_rewards)) if recent_ep_rewards else 0.0
                exit_rate = float(np.mean(recent_exits)) * 100.0 if recent_exits else 0.0
                key_rate = float(np.mean(recent_keys)) * 100.0 if recent_keys else 0.0
                mean_kills = float(np.mean(recent_kills)) if recent_kills else 0.0
                mean_doors = float(np.mean(recent_doors)) if recent_doors else 0.0
                mean_locked = float(np.mean(recent_locked)) * 100.0 if recent_locked else 0.0
                mean_tic = float(np.mean(recent_tics)) if recent_tics else 0.0
                vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
                print(
                    f"step={global_step:,} | sps={sps:,.0f} | r={mean_r:.2f} | exit={exit_rate:.1f}% | "
                    f"key={key_rate:.1f}% | lock={mean_locked:.1f}% | doors={mean_doors:.2f} | kills={mean_kills:.2f} | "
                    f"tic={mean_tic:.0f} | pg={np.mean(pg_losses):.3f} | vf={np.mean(v_losses):.3f} | "
                    f"aux={np.mean(aux_losses):.3f} | ent={np.mean(ents):.3f} (c={ent_coef:.4f}) | vram={vram_mb:.0f}MB",
                    flush=True,
                )

                # Composite checkpoint selection score matching evaluation priorities
                score = exit_rate * 2.0 + key_rate * 0.6 + mean_locked * 0.8 + mean_r
                is_best = (len(recent_ep_rewards) >= 10 and score > best_score) or (best_score < -1e8 and update_idx >= 1)
                if is_best:
                    best_score = max(best_score, score)
                save_checkpoint(ckpt_dir, model, optimizer, global_step, score, recent_ckpts, is_best=is_best)
    finally:
        envs.close()

    # Final checkpoint save
    mean_r = float(np.mean(recent_ep_rewards)) if recent_ep_rewards else 0.0
    exit_rate = float(np.mean(recent_exits)) * 100.0 if recent_exits else 0.0
    key_rate = float(np.mean(recent_keys)) * 100.0 if recent_keys else 0.0
    mean_locked = float(np.mean(recent_locked)) * 100.0 if recent_locked else 0.0
    final_score = exit_rate * 2.0 + key_rate * 0.6 + mean_locked * 0.8 + mean_r
    save_checkpoint(ckpt_dir, model, optimizer, global_step, final_score, recent_ckpts, is_best=(final_score >= best_score))
    print(f"Training complete. Checkpoints saved under {ckpt_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train DoomSpatialMemoryNet with Vectorized Recurrent PPO")
    ap.add_argument("--stage", default="1", choices=["1", "2", "3", "4", "all"], help="Curriculum stage")
    ap.add_argument("--wad-dir", default="train/generated_wads", help="Generated WADs directory")
    ap.add_argument("--ckpt-dir", default="train/checkpoints", help="Checkpoint directory")
    ap.add_argument("--init-from", default="", help="Path to checkpoint/weights to initialize from")
    ap.add_argument("--num-envs", type=int, default=12, help="Parallel AsyncVectorEnv workers")
    ap.add_argument("--n-steps", type=int, default=256, help="Rollout steps per worker")
    ap.add_argument("--seq-len", type=int, default=32, help="Truncated BPTT sequence chunk length")
    ap.add_argument("--batch-size", type=int, default=512, help="PPO minibatch transitions")
    ap.add_argument("--n-epochs", type=int, default=4, help="PPO epochs per rollout")
    ap.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    ap.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    ap.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    ap.add_argument("--clip-coef", type=float, default=0.2, help="PPO clip coefficient")
    ap.add_argument("--vf-coef", type=float, default=0.5, help="Value loss weight")
    ap.add_argument("--aux-coef", type=float, default=0.2, help="Auxiliary BCE loss weight")
    ap.add_argument("--rnd-coef", type=float, default=0.03, help="RND intrinsic curiosity reward scale")
    ap.add_argument("--ent-start", type=float, default=0.03, help="Initial entropy coefficient")
    ap.add_argument("--ent-end", type=float, default=0.005, help="Final entropy coefficient")
    ap.add_argument("--wad-episodes", type=int, default=32, help="Episodes per WAD before rotating")
    ap.add_argument("--cell-bonus", type=float, default=0.05, help="New 128x128 cell visitation reward")
    ap.add_argument("--door-use-bonus", type=float, default=0.03, help="Door USE discovery shaping reward")
    ap.add_argument("--stall-penalty", type=float, default=0.0, help="Penalty if stuck <16 units over 8 steps")
    ap.add_argument("--total-steps", type=int, default=5_000_000, help="Max total environment steps")
    ap.add_argument("--max-minutes", type=float, default=0.0, help="Wall-clock cap in minutes (0 = no cap)")
    ap.add_argument("--log-interval", type=int, default=50_000, help="Steps between log & checkpoint prints")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--cpu", action="store_true", help="Force CPU training")
    return ap.parse_args()


if __name__ == "__main__":
    train(parse_args())
