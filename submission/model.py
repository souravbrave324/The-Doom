"""Bespoke Spatial-Memory Actor-Critic with Auxiliary Supervision (DoomSpatialMemoryNet).

Pure PyTorch architecture (~920K parameters) designed specifically for partial observability
and long-horizon progression in ViZDoom without global coordinates:
  1. CircularRayEncoder: 1D circular convolutions over 64 360-degree ego-centric raycasts.
  2. EntityCrossAttention: Multi-head cross-attention with 2 learned task queries
     (Combat Threat Query & Progression/Loot Query) over variable-length visible entities.
  3. EgocentricMapBuilder: Differentiable projection of rays onto a 24x24 local occupancy map,
     motion-warped by ego velocity/yaw and encoded via 2-layer Conv2d + persistent landmark latch.
  4. TemporalCore: Fusion + GRU(256) recurrent memory.
  5. Heads: MultiBinary(9) Actor logits, Critic value, and 3-logit Auxiliary Supervision head.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_RAYS = 64
NUM_RAY_TYPES = 6
MAX_ENTITIES = 32
NUM_ENTITY_TYPES = 12
ENTITY_FEAT_DIM = 5
VITALS_DIM = 17
NUM_ACTIONS = 9
EGO_GRID_SIZE = 24
EGO_CHANNELS = 6  # [free_space, wall, door, locked_door, enemy, item]
EGO_CELL_UNITS = 48.0  # 24 * 48 = 1152 map units (~1024 ray range)
RAY_RANGE = 1024.0
TURN_RAD_PER_STEP = 0.08  # Approx ViZDoom yaw delta per 4-tick turn step


@dataclass
class AgentMemoryState:
    """Recurrent and spatial memory state carried across steps in an episode."""

    gru_hidden: torch.Tensor  # (1, B, 256)
    ego_map: torch.Tensor     # (B, 6, 24, 24)
    latch: torch.Tensor       # (B, 2) -> [key_seen_trace, locked_door_seen_trace]
    prev_action: torch.Tensor # (B, 9)


class CircularRayEncoder(nn.Module):
    """Encodes 64 360-degree ego-centric raycasts with zero boundary seam."""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv1d(1 + NUM_RAY_TYPES, 48, kernel_size=5, padding=2, padding_mode="circular")
        self.conv2 = nn.Conv1d(48, 64, kernel_size=5, padding=2, padding_mode="circular")
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.proj = nn.Sequential(
            nn.Linear(64 * 8, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, distance: torch.Tensor, hit_type: torch.Tensor) -> torch.Tensor:
        # distance: (N, 64), hit_type: (N, 64) long
        onehot = F.one_hot(hit_type.clamp(0, NUM_RAY_TYPES - 1), num_classes=NUM_RAY_TYPES).to(distance.dtype)
        x = torch.cat([distance.unsqueeze(-1), onehot], dim=-1).transpose(1, 2)  # (N, 7, 64)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        # Roll by 4 rays so sector 0 of the 8-bin pool centers directly ahead (rays [-4..+3])
        x = torch.roll(x, shifts=4, dims=-1)
        x = self.pool(x).flatten(1)  # (N, 512)
        return self.proj(x)


class EntityCrossAttention(nn.Module):
    """Cross-attends 2 learned task queries (Combat Threat vs Progression/Loot) over visible entities."""

    def __init__(self, d_model: int = 128, num_heads: int = 4):
        super().__init__()
        self.type_emb = nn.Embedding(NUM_ENTITY_TYPES, 32)
        self.entity_mlp = nn.Sequential(
            nn.Linear(ENTITY_FEAT_DIM + 32, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.queries = nn.Parameter(torch.randn(1, 2, d_model) * 0.02)
        # Always-unmasked null token prevents NaN softmax when zero entities are visible
        self.null_kv = nn.Parameter(torch.zeros(1, 1, d_model))
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.out_proj = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor, entity_type: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # features: (N, 32, 5), entity_type: (N, 32), mask: (N, 32)
        n = features.shape[0]
        scaled_feats = torch.stack(
            [
                features[..., 0].clamp(0.0, 4.0),
                features[..., 1],
                features[..., 2],
                (features[..., 3] / 20.0).clamp(-2.0, 2.0),
                (features[..., 4] / 20.0).clamp(-2.0, 2.0),
            ],
            dim=-1,
        )
        te = self.type_emb(entity_type.clamp(0, NUM_ENTITY_TYPES - 1))
        kv = self.entity_mlp(torch.cat([scaled_feats, te], dim=-1))  # (N, 32, d)
        null_tok = self.null_kv.expand(n, -1, -1).to(kv.dtype)
        kv_full = torch.cat([null_tok, kv], dim=1)  # (N, 33, d)

        # key_padding_mask: True means ignored
        ent_pad = mask <= 0.5
        null_pad = torch.zeros((n, 1), dtype=torch.bool, device=mask.device)
        key_padding_mask = torch.cat([null_pad, ent_pad], dim=1)

        q = self.queries.expand(n, -1, -1).to(kv.dtype)
        attended, _ = self.attn(q, kv_full, kv_full, key_padding_mask=key_padding_mask, need_weights=False)
        return self.out_proj(attended.flatten(1))


class EgocentricMapBuilder(nn.Module):
    """Differentiable 24x24 egocentric occupancy tensor with motion warping and 2-layer Conv2d."""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        angles = torch.arange(NUM_RAYS, dtype=torch.float32) * (2.0 * math.pi / NUM_RAYS)
        self.register_buffer("cos_a", torch.cos(angles), persistent=False)
        self.register_buffer("sin_a", torch.sin(angles), persistent=False)
        # Base normalized grid coordinates in [-1, 1] for 24x24 map
        lin = torch.linspace(-1.0, 1.0, EGO_GRID_SIZE, dtype=torch.float32)
        gy, gx = torch.meshgrid(lin, lin, indexing="ij")
        self.register_buffer("base_gx", gx, persistent=False)
        self.register_buffer("base_gy", gy, persistent=False)

        self.conv = nn.Sequential(
            nn.Conv2d(EGO_CHANNELS, 24, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((3, 3)),
        )
        self.proj = nn.Sequential(
            nn.Linear(48 * 9, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def splat_rays(self, distance: torch.Tensor, hit_type: torch.Tensor) -> torch.Tensor:
        """Project 64 ego rays onto a fresh (N, 6, 24, 24) occupancy grid."""
        n = distance.shape[0]
        device = distance.device
        # Use float32 for coordinate math cleanly
        dist_units = distance.float().clamp(0.0, 1.0) * RAY_RANGE  # (N, 64)
        # Ego coordinates: +x = forward (col), +y = left (row)
        hx = dist_units * self.cos_a.unsqueeze(0)
        hy = dist_units * self.sin_a.unsqueeze(0)

        half_span = (EGO_GRID_SIZE * EGO_CELL_UNITS) * 0.5  # 576 units
        center = (EGO_GRID_SIZE - 1) * 0.5  # 11.5
        col = ((hx / half_span) * center + center).round().long().clamp(0, EGO_GRID_SIZE - 1)
        row = ((hy / half_span) * center + center).round().long().clamp(0, EGO_GRID_SIZE - 1)

        splat = torch.zeros((n, EGO_CHANNELS, EGO_GRID_SIZE, EGO_GRID_SIZE), dtype=torch.float32, device=device)

        # Free space along 3 intermediate fractions (0.25, 0.5, 0.75) of each ray
        batch_idx = torch.arange(n, device=device).unsqueeze(1).expand(n, NUM_RAYS)
        for frac in (0.25, 0.5, 0.75):
            fc = ((hx * frac / half_span) * center + center).round().long().clamp(0, EGO_GRID_SIZE - 1)
            fr = ((hy * frac / half_span) * center + center).round().long().clamp(0, EGO_GRID_SIZE - 1)
            splat[batch_idx, 0, fr, fc] = 1.0

        # Hit endpoints for hit_type in 1..5 -> channels 1..5 (no CPU-GPU .any() sync)
        ht = hit_type.long().clamp(0, NUM_RAY_TYPES - 1)
        hit_val = (ht > 0).to(torch.float32)
        splat[batch_idx, ht, row, col] = torch.maximum(splat[batch_idx, ht, row, col], hit_val)
        return splat

    def warp_map(self, prev_map: torch.Tensor, vel_fwd: torch.Tensor, vel_right: torch.Tensor, turn_dir: torch.Tensor) -> torch.Tensor:
        """Warp (B, 6, 24, 24) previous ego-map by ego displacement and yaw delta."""
        b = prev_map.shape[0]
        half_span = (EGO_GRID_SIZE * EGO_CELL_UNITS) * 0.5
        # 4 tics per step -> displacement in normalized [-1, 1] grid coordinates
        dx_norm = (vel_fwd.float() * 4.0 / half_span).clamp(-0.5, 0.5)
        dy_norm = (-vel_right.float() * 4.0 / half_span).clamp(-0.5, 0.5)
        d_theta = (turn_dir.float() * TURN_RAD_PER_STEP)

        cos_t = torch.cos(d_theta).view(b, 1, 1)
        sin_t = torch.sin(d_theta).view(b, 1, 1)
        gx = self.base_gx.unsqueeze(0)
        gy = self.base_gy.unsqueeze(0)

        # Inverse transform from new frame to previous frame
        tx = gx + dx_norm.view(b, 1, 1)
        ty = gy + dy_norm.view(b, 1, 1)
        src_x = cos_t * tx - sin_t * ty
        src_y = sin_t * tx + cos_t * ty
        grid = torch.stack([src_x, src_y], dim=-1)  # (B, 24, 24, 2)

        return F.grid_sample(prev_map.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    def update_and_encode_step(
        self,
        prev_map: torch.Tensor,
        distance: torch.Tensor,
        hit_type: torch.Tensor,
        vitals: torch.Tensor,
        turn_dir: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.amp.autocast(device_type=prev_map.device.type, enabled=False):
            warped = self.warp_map(prev_map, vitals[:, 14], vitals[:, 15], turn_dir)
            splat = self.splat_rays(distance, hit_type)
            new_map = torch.clamp(torch.maximum(0.88 * warped, splat), 0.0, 1.0)
        enc = self.proj(self.conv(new_map.to(self.conv[0].weight.dtype)).flatten(1))
        return enc, new_map

    def encode_only(self, ego_maps: torch.Tensor) -> torch.Tensor:
        return self.proj(self.conv(ego_maps.to(self.conv[0].weight.dtype)).flatten(1))


class DoomSpatialMemoryNet(nn.Module):
    """Spatial-Memory Actor-Critic with Auxiliary Landmark & Threat Supervision."""

    def __init__(self):
        super().__init__()
        self.ray_encoder = CircularRayEncoder(out_dim=128)
        self.entity_attn = EntityCrossAttention(d_model=128, num_heads=4)
        self.ego_map = EgocentricMapBuilder(out_dim=128)

        # Vitals (17) + persistent latch (2) + prev_action (9) = 28
        self.vitals_encoder = nn.Sequential(
            nn.Linear(VITALS_DIM + 2 + NUM_ACTIONS, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )

        # Fusion: 128 + 128 + 128 + 64 = 448 -> 256
        self.fusion = nn.Sequential(
            nn.Linear(128 + 128 + 128 + 64, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )
        self.gru = nn.GRU(256, 256, batch_first=True)

        self.actor_head = nn.Sequential(
            nn.Linear(256, 192),
            nn.GELU(),
            nn.Linear(192, NUM_ACTIONS),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(256, 192),
            nn.GELU(),
            nn.Linear(192, 1),
        )
        # [enemy_threat_near, keycard_seen_in_episode, locked_door_seen_in_episode]
        self.aux_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 3),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head[-1].weight, gain=1.0)
        nn.init.orthogonal_(self.aux_head[-1].weight, gain=0.1)

    def init_memory(self, batch_size: int, device: torch.device) -> AgentMemoryState:
        return AgentMemoryState(
            gru_hidden=torch.zeros((1, batch_size, 256), dtype=torch.float32, device=device),
            ego_map=torch.zeros((batch_size, EGO_CHANNELS, EGO_GRID_SIZE, EGO_GRID_SIZE), dtype=torch.float32, device=device),
            latch=torch.zeros((batch_size, 2), dtype=torch.float32, device=device),
            prev_action=torch.zeros((batch_size, NUM_ACTIONS), dtype=torch.float32, device=device),
        )

    @staticmethod
    def _normalize_vitals(vitals: torch.Tensor) -> torch.Tensor:
        out = vitals.clone()
        out[..., 0] = (vitals[..., 0] / 100.0).clamp(0.0, 2.5)
        out[..., 1] = (vitals[..., 1] / 100.0).clamp(0.0, 2.5)
        out[..., 2] = (vitals[..., 2] / 8.0).clamp(0.0, 1.5)
        out[..., 3:8] = (vitals[..., 3:8] / 100.0).clamp(0.0, 4.0)
        out[..., 14:17] = (vitals[..., 14:17] / 20.0).clamp(-2.5, 2.5)
        return out

    @staticmethod
    def update_latch(prev_latch: torch.Tensor, obs_rays_hit: torch.Tensor, obs_ent_type: torch.Tensor, obs_ent_mask: torch.Tensor, vitals: torch.Tensor) -> torch.Tensor:
        """Update persistent [keycard_seen, locked_door_seen] episodic memory latch."""
        key_vis = ((obs_ent_type == 9) & (obs_ent_mask > 0.5)).any(dim=-1).float()
        key_held = (vitals[..., 8:14].sum(dim=-1) > 0.5).float()
        locked_vis = (obs_rays_hit == 3).any(dim=-1).float()
        k_flag = torch.maximum(prev_latch[..., 0], torch.maximum(key_vis, key_held))
        l_flag = torch.maximum(prev_latch[..., 1], locked_vis)
        return torch.stack([k_flag, l_flag], dim=-1)

    def forward_step(
        self,
        obs: dict,
        mem: AgentMemoryState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, AgentMemoryState]:
        """Single-step forward for rollout collection and inference.

        obs tensors have shape (B, ...).
        """
        r_dist = obs["rays"]["distance"]
        r_hit = obs["rays"]["hit_type"]
        e_feat = obs["visible_entities"]["features"]
        e_type = obs["visible_entities"]["entity_type"]
        e_mask = obs["visible_entities"]["mask"]
        vitals = obs["vitals"]

        turn_dir = mem.prev_action[:, 4] - mem.prev_action[:, 5]
        new_latch = self.update_latch(mem.latch, r_hit, e_type, e_mask, vitals)

        ray_emb = self.ray_encoder(r_dist, r_hit)
        ent_emb = self.entity_attn(e_feat, e_type, e_mask)
        ego_emb, new_ego_map = self.ego_map.update_and_encode_step(mem.ego_map, r_dist, r_hit, vitals, turn_dir)

        v_norm = self._normalize_vitals(vitals)
        v_in = torch.cat([v_norm, new_latch.to(v_norm.dtype), mem.prev_action.to(v_norm.dtype)], dim=-1)
        vit_emb = self.vitals_encoder(v_in)

        fused = self.fusion(torch.cat([ray_emb, ent_emb, ego_emb, vit_emb], dim=-1))
        gru_out, new_h = self.gru(fused.unsqueeze(1), mem.gru_hidden.float() if fused.dtype == torch.float32 else mem.gru_hidden.to(fused.dtype))
        feat = gru_out.squeeze(1)

        logits = self.actor_head(feat)
        value = self.critic_head(feat).squeeze(-1)
        aux_logits = self.aux_head(feat)

        next_mem = AgentMemoryState(
            gru_hidden=new_h.detach(),
            ego_map=new_ego_map.detach(),
            latch=new_latch.detach(),
            prev_action=mem.prev_action,
        )
        return logits, value, aux_logits, next_mem

    def forward_sequence(
        self,
        obs_seq: dict,
        dones_seq: torch.Tensor,
        prev_actions_seq: torch.Tensor,
        init_mem: AgentMemoryState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unroll sequence (B, T, ...) with mid-chunk episode resets for recurrent PPO training."""
        b, t = obs_seq["vitals"].shape[:2]

        # Vectorize ray encoder and entity cross-attention over B*T
        r_dist = obs_seq["rays"]["distance"].reshape(b * t, NUM_RAYS)
        r_hit = obs_seq["rays"]["hit_type"].reshape(b * t, NUM_RAYS)
        e_feat = obs_seq["visible_entities"]["features"].reshape(b * t, MAX_ENTITIES, ENTITY_FEAT_DIM)
        e_type = obs_seq["visible_entities"]["entity_type"].reshape(b * t, MAX_ENTITIES)
        e_mask = obs_seq["visible_entities"]["mask"].reshape(b * t, MAX_ENTITIES)
        vitals_bt = obs_seq["vitals"].reshape(b * t, VITALS_DIM)

        ray_emb = self.ray_encoder(r_dist, r_hit).reshape(b, t, -1)
        ent_emb = self.entity_attn(e_feat, e_type, e_mask).reshape(b, t, -1)

        # Precompute splats for all B*T in parallel, then warp/latch/GRU sequentially across T
        with torch.amp.autocast(device_type=vitals_bt.device.type, enabled=False):
            all_splats = self.ego_map.splat_rays(r_dist, r_hit).reshape(b, t, EGO_CHANNELS, EGO_GRID_SIZE, EGO_GRID_SIZE)

        ego_maps_list = []
        latches_list = []
        cur_map = init_mem.ego_map.float()
        cur_latch = init_mem.latch.float()

        for step in range(t):
            if step > 0:
                not_done = (1.0 - dones_seq[:, step - 1].float())
                cur_map = cur_map * not_done.view(b, 1, 1, 1)
                cur_latch = cur_latch * not_done.view(b, 1)

            pa = prev_actions_seq[:, step]
            turn_dir = pa[:, 4] - pa[:, 5]
            v_step = obs_seq["vitals"][:, step]
            with torch.amp.autocast(device_type=cur_map.device.type, enabled=False):
                warped = self.ego_map.warp_map(cur_map, v_step[:, 14], v_step[:, 15], turn_dir)
                cur_map = torch.clamp(torch.maximum(0.88 * warped, all_splats[:, step]), 0.0, 1.0)
            cur_latch = self.update_latch(
                cur_latch,
                obs_seq["rays"]["hit_type"][:, step],
                obs_seq["visible_entities"]["entity_type"][:, step],
                obs_seq["visible_entities"]["mask"][:, step],
                v_step,
            )
            ego_maps_list.append(cur_map)
            latches_list.append(cur_latch)

        ego_maps_bt = torch.stack(ego_maps_list, dim=1).reshape(b * t, EGO_CHANNELS, EGO_GRID_SIZE, EGO_GRID_SIZE)
        ego_emb = self.ego_map.encode_only(ego_maps_bt).reshape(b, t, -1)
        latches_seq = torch.stack(latches_list, dim=1)

        v_norm = self._normalize_vitals(obs_seq["vitals"])
        v_in = torch.cat([v_norm, latches_seq.to(v_norm.dtype), prev_actions_seq.to(v_norm.dtype)], dim=-1)
        vit_emb = self.vitals_encoder(v_in.reshape(b * t, -1)).reshape(b, t, -1)

        fused = self.fusion(torch.cat([ray_emb, ent_emb, ego_emb, vit_emb], dim=-1).reshape(b * t, -1)).reshape(b, t, -1)

        # GRU step-by-step only where dones occur, or fast whole-chunk GRU if no dones inside chunk
        if not dones_seq[:, :-1].any():
            gru_out, _ = self.gru(fused, init_mem.gru_hidden.to(fused.dtype))
        else:
            h = init_mem.gru_hidden.to(fused.dtype)
            outs = []
            for step in range(t):
                if step > 0:
                    nd = (1.0 - dones_seq[:, step - 1].to(fused.dtype)).view(1, b, 1)
                    h = h * nd
                o, h = self.gru(fused[:, step : step + 1], h)
                outs.append(o)
            gru_out = torch.cat(outs, dim=1)

        feat_bt = gru_out.reshape(b * t, 256)
        logits = self.actor_head(feat_bt).reshape(b, t, NUM_ACTIONS)
        values = self.critic_head(feat_bt).reshape(b, t)
        aux_logits = self.aux_head(feat_bt).reshape(b, t, 3)
        return logits, values, aux_logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
