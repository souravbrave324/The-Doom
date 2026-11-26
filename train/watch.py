"""Live DOOM gameplay viewer + DoomSpatialMemoryNet HUD (Ego-Map, Rays, Aux Predictions).

Run in CMD:
  .venv\\Scripts\\python.exe -m train.watch
  .venv\\Scripts\\python.exe -m train.watch --map train\\generated_wads\\tier1\\tier1_combat_000.wad:MAP01
  .venv\\Scripts\\python.exe -m train.watch --map train\\generated_wads\\tier3\\tier3_keycard_000.wad:MAP01
  .venv\\Scripts\\python.exe -m train.watch --freedoom MAP01
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pygame
import torch
import vizdoom as vd

from doom_env import make_env
from doom_env import spaces as S
from submission.agent import DoomAgent

HIT_COLORS = {
    0: (40, 40, 40),     # none
    1: (180, 180, 190),  # wall
    2: (0, 220, 220),    # door
    3: (60, 130, 255),   # locked door
    4: (255, 60, 60),    # enemy
    5: (255, 220, 50),   # item
}


def _ego_map_to_surface(ego_map: np.ndarray, size: int = 240) -> pygame.Surface:
    """Convert (6, 24, 24) ego_map tensor into a top-down Forward=UP RGB surface."""
    # ego_map coords: row=y (+left), col=x (+forward) -> rotate so +forward is UP, +left is LEFT
    grid = np.zeros((24, 24, 3), dtype=np.uint8)
    grid[:, :] = (18, 22, 28)
    # Channel 0: free space
    free = np.clip(ego_map[0], 0.0, 1.0)
    grid[..., 1] = np.maximum(grid[..., 1], (free * 55).astype(np.uint8))
    # Channels 1..5: wall, door, locked_door, enemy, item
    cmap = {
        1: (170, 175, 185),
        2: (0, 210, 220),
        3: (65, 140, 255),
        4: (255, 65, 65),
        5: (255, 220, 55),
    }
    for ch, rgb in cmap.items():
        mask = ego_map[ch] > 0.25
        for c in range(3):
            grid[..., c] = np.where(mask, np.clip(ego_map[ch] * rgb[c], 0, 255).astype(np.uint8), grid[..., c])

    # Rotate 90 deg CCW so col(+forward) points UP and row(+left) points LEFT
    disp = np.rot90(grid, k=1)  # shape (24, 24, 3) -> make (W, H, 3) for surfarray
    surf = pygame.surfarray.make_surface(np.transpose(disp, (1, 0, 2)))
    return pygame.transform.scale(surf, (size, size))


def _build_playlist(args: argparse.Namespace) -> list[tuple[str, str, int]]:
    if args.tour:
        iwad = os.path.join(os.path.dirname(vd.__file__), "freedoom2.wad")
        playlist = [
            ("maps/two_rooms.wad", "MAP01", 2100),
            ("train/generated_wads/tier1/tier1_combat_000.wad", "MAP01", 2100),
            ("train/generated_wads/tier2/tier2_doors_000.wad", "MAP01", 2400),
            ("train/generated_wads/tier3/tier3_keycard_000.wad", "MAP01", 3200),
            ("train/generated_wads/tier4/tier4_gauntlet_000.wad", "MAP01", 3600),
        ]
        if Path(iwad).is_file():
            playlist.append((iwad, "MAP01", 4200))
        return [(w, m, t) for w, m, t in playlist if Path(w).is_file()]
    if args.freedoom:
        iwad = os.path.join(os.path.dirname(vd.__file__), "freedoom2.wad")
        return [(iwad, args.freedoom.upper(), args.ticks)] * args.episodes
    wad_str, _, mname = args.map.partition(":")
    return [(wad_str, (mname or "MAP01").upper(), args.ticks)] * args.episodes


def watch(args: argparse.Namespace) -> None:
    playlist = _build_playlist(args)
    if not playlist:
        raise FileNotFoundError("No valid WADs found for playlist")

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device)
    probe_env = make_env(playlist[0][0], playlist[0][1], episode_ticks=playlist[0][2], render_buffers=False)
    agent = DoomAgent(probe_env.observation_space, probe_env.action_space, device)
    probe_env.close()

    if not args.no_window:
        pygame.init()
        screen = pygame.display.set_mode((980, 520))
        pygame.display.set_caption("DoomSpatialMemoryNet — Live Demo")
        font = pygame.font.SysFont("consolas", 15)
        font_b = pygame.font.SysFont("consolas", 17, bold=True)
        clock = pygame.time.Clock()
    else:
        screen = font = font_b = clock = None

    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    cur_env = None
    cur_key = None
    try:
        for ep, (wad_path, map_name, ep_ticks) in enumerate(playlist):
            if cur_env is None or cur_key != (wad_path, map_name, ep_ticks):
                if cur_env is not None:
                    cur_env.close()
                cur_env = make_env(wad_path, map_name, episode_ticks=ep_ticks, render_buffers=True)
                cur_key = (wad_path, map_name, ep_ticks)
            env = cur_env
            obs, info = env.reset(seed=args.seed + ep)
            agent.reset()
            step = 0
            running = True
            while running and step < args.max_steps:
                if not args.no_window:
                    for ev in pygame.event.get():
                        if ev.type == pygame.QUIT:
                            return
                        if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                            return

                t0 = time.perf_counter()
                action = agent.act(obs)
                lat_ms = (time.perf_counter() - t0) * 1000.0

                with torch.inference_mode():
                    aux_probs = torch.sigmoid(agent.model.aux_head(agent._mem.gru_hidden.squeeze(0)))[0].detach().cpu().numpy()

                obs, _, term, trunc, info = env.step(action)
                st = env.privileged_state()
                step += 1

                if not args.no_window:
                    screen.fill((14, 16, 22))
                    rgb = obs["rgb"]
                    game_surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
                    game_surf = pygame.transform.scale(game_surf, (640, 480))
                    screen.blit(game_surf, (16, 20))

                    ego_np = agent._mem.ego_map[0].detach().cpu().numpy()
                    ego_surf = _ego_map_to_surface(ego_np, size=240)
                    screen.blit(ego_surf, (690, 44))
                    pygame.draw.rect(screen, (90, 95, 110), (690, 44, 240, 240), 1)
                    pygame.draw.polygon(screen, (255, 255, 255), [(810, 156), (805, 170), (815, 170)])
                    screen.blit(font_b.render("24x24 Neural Ego-Map", True, (220, 230, 245)), (690, 20))

                    y = 296
                    keys_held = int(np.sum(obs["vitals"][8:14]) > 0.5)
                    lines = [
                        f"Map: {Path(wad_path).name}:{map_name} ({ep+1}/{len(playlist)})",
                        f"Step: {step} | Tic: {st['tic']}/{ep_ticks} | {lat_ms:.1f}ms",
                        f"HP: {st['health']:.0f} | Kills: {st['kills']} | Key: {'YES' if keys_held else 'NO'}",
                        f"Aux Threat: {aux_probs[0]:.2f} | Key: {aux_probs[1]:.2f} | Lock: {aux_probs[2]:.2f}",
                    ]
                    for ln in lines:
                        screen.blit(font.render(ln, True, (210, 215, 225)), (674, y))
                        y += 22

                    btn_names = ["FWD", "BCK", "L", "R", "TL", "TR", "ATK", "USE", "WPN"]
                    bx = 674
                    for i, bname in enumerate(btn_names):
                        active = bool(action[i])
                        col = (210, 50, 50) if active else (45, 50, 62)
                        pygame.draw.rect(screen, col, (bx, y + 6, 30, 22), border_radius=3)
                        screen.blit(font.render(bname[:3], True, (240, 240, 240)), (bx + 2, y + 9))
                        bx += 33

                    pygame.display.flip()
                    if args.save_dir:
                        pygame.image.save(screen, str(Path(args.save_dir) / f"ep{ep:02d}_step{step:04d}.png"))
                    clock.tick(args.fps)

                if term or trunc:
                    outcome = "EXIT REACHED!" if info.get("exit_reached") else ("DIED" if st["dead"] else "TIME UP")
                    print(f"[{Path(wad_path).name}:{map_name}] {outcome} | steps={step} | kills={st['kills']}", flush=True)
                    if not args.no_window:
                        time.sleep(1.2)
                    break
    finally:
        if cur_env is not None:
            cur_env.close()
        if not args.no_window:
            pygame.quit()


def main() -> None:
    ap = argparse.ArgumentParser(description="Watch DoomAgent play DOOM live with neural Ego-Map HUD")
    ap.add_argument("--map", default="maps/two_rooms.wad:MAP01", help="WAD[:MAP] to play")
    ap.add_argument("--freedoom", default="", help="Optional Freedoom2 map name (e.g. MAP01)")
    ap.add_argument("--tour", action="store_true", help="Play multi-tier + Freedoom2 showcase tour")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--ticks", type=int, default=2100)
    ap.add_argument("--fps", type=int, default=12, help="Playback decisions/sec (8.75 is real-time)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu | cuda | auto")
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--save-dir", default="", help="Optional folder to save PNG frames")
    ap.add_argument("--no-window", action="store_true", help="Run without opening pygame window")
    args = ap.parse_args()
    watch(args)


if __name__ == "__main__":
    main()
