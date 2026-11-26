"""Local benchmark evaluator across 20 freshly generated unseen test WADs (5 per tier).

Instantiates DoomAgent from submission/agent.py and evaluates under vanilla doom_env
rules, reporting Exit Rate %, Keycards Found %, Kills, and Avg Decision Latency (ms).
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import vizdoom as vd

from doom_env import make_env
from submission.agent import DoomAgent
from train.wad_gen import ProceduralWadGenerator


def evaluate(per_tier: int = 5, seed_base: int = 990000, device: str = "auto", out_json: str = "") -> dict:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    gen = ProceduralWadGenerator()
    tier_ticks = {
        "tier1_combat": 2100,
        "tier2_doors": 2400,
        "tier3_keycard": 3200,
        "tier4_gauntlet": 3600,
    }

    with tempfile.TemporaryDirectory(prefix="doom-eval-") as tmpdir:
        tmp = Path(tmpdir)
        # Initialize DoomAgent once, matching evaluation contract
        probe_wad = gen.generate_map("tier1_combat", seed_base, tmp / "probe.wad")
        probe_env = make_env(str(probe_wad), "MAP01", episode_ticks=2100)
        agent = DoomAgent(probe_env.observation_space, probe_env.action_space, device)
        probe_env.close()

        summary: dict[str, dict] = {}
        all_lat_ms: list[float] = []

        for t_idx, tier in enumerate(ProceduralWadGenerator.TIERS, start=1):
            exits, keys_found, kills_list, lats_ms, cells_list = [], [], [], [], []
            ticks = tier_ticks[tier]
            for i in range(per_tier):
                wad_path = gen.generate_map(tier, seed_base + t_idx * 1000 + i, tmp / f"{tier}_{i}.wad")
                env = make_env(str(wad_path), "MAP01", episode_ticks=ticks, render_buffers=False)
                try:
                    obs, _ = env.reset(seed=seed_base + i)
                    agent.reset()
                    while True:
                        t0 = time.perf_counter()
                        action = agent.act(obs)
                        dt_ms = (time.perf_counter() - t0) * 1000.0
                        lats_ms.append(dt_ms)
                        all_lat_ms.append(dt_ms)
                        obs, _, term, trunc, info = env.step(action)
                        if term or trunc:
                            st = env.privileged_state()
                            exits.append(1.0 if info.get("exit_reached", False) else 0.0)
                            keys_found.append(1.0 if np.sum(st["keys"]) > 0.5 else 0.0)
                            kills_list.append(float(st["kills"]))
                            break
                finally:
                    env.close()

            summary[tier] = {
                "maps": per_tier,
                "exit_rate_pct": float(np.mean(exits) * 100.0),
                "keycards_pct": float(np.mean(keys_found) * 100.0) if t_idx >= 3 else None,
                "avg_kills": float(np.mean(kills_list)),
                "avg_latency_ms": float(np.mean(lats_ms)),
                "p99_latency_ms": float(np.percentile(lats_ms, 99)),
            }

        # Held-out real Freedoom 2 MAP01 check
        iwad = os.path.join(os.path.dirname(vd.__file__), "freedoom2.wad")
        if Path(iwad).is_file():
            env = make_env(iwad, "MAP01", episode_ticks=3600, render_buffers=False)
            fd_lats = []
            try:
                obs, _ = env.reset(seed=seed_base)
                agent.reset()
                while True:
                    t0 = time.perf_counter()
                    action = agent.act(obs)
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    fd_lats.append(dt_ms)
                    all_lat_ms.append(dt_ms)
                    obs, _, term, trunc, info = env.step(action)
                    if term or trunc:
                        st = env.privileged_state()
                        summary["freedoom2_MAP01"] = {
                            "maps": 1,
                            "exit_rate_pct": 100.0 if info.get("exit_reached", False) else 0.0,
                            "keycards_pct": None,
                            "avg_kills": float(st["kills"]),
                            "avg_latency_ms": float(np.mean(fd_lats)),
                            "p99_latency_ms": float(np.percentile(fd_lats, 99)),
                        }
                        break
            finally:
                env.close()

    print("\n### Unseen Procedural WAD Benchmark (20 Maps — 5 per Tier)\n")
    print("| Tier | Maps | Exit Rate (%) | Keycards Found (%) | Avg Kills | Avg Decision Latency (ms) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for tier, m in summary.items():
        key_str = f"{m['keycards_pct']:.1f}%" if m["keycards_pct"] is not None else "N/A"
        print(f"| `{tier}` | {m['maps']} | {m['exit_rate_pct']:.1f}% | {key_str} | {m['avg_kills']:.2f} | {m['avg_latency_ms']:.2f} ms |")

    overall = {
        "overall_exit_rate_pct": float(np.mean([v["exit_rate_pct"] for v in summary.values()])),
        "overall_avg_kills": float(np.mean([v["avg_kills"] for v in summary.values()])),
        "overall_avg_latency_ms": float(np.mean(all_lat_ms)),
        "overall_p99_latency_ms": float(np.percentile(all_lat_ms, 99)),
        "tiers": summary,
    }
    print(f"\nOverall Avg Exit Rate: {overall['overall_exit_rate_pct']:.1f}% | Avg Kills: {overall['overall_avg_kills']:.2f} | Avg Latency: {overall['overall_avg_latency_ms']:.2f} ms (p99 {overall['overall_p99_latency_ms']:.2f} ms)")

    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(overall, indent=2))
    return overall


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate DoomAgent on unseen procedural WADs")
    ap.add_argument("--per-tier", type=int, default=5)
    ap.add_argument("--seed-base", type=int, default=990000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()
    evaluate(per_tier=args.per_tier, seed_base=args.seed_base, device=args.device, out_json=args.out_json)


if __name__ == "__main__":
    main()
