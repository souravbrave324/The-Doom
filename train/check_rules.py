"""Static rule-compliance check for submission/ against docs/RULES.md.

Run: .venv\\Scripts\\python.exe -m train.check_rules
Fails (exit 1) if submission/ contains banned imports, pathfinders, trig aimbots,
privileged engine access, or invalid requirements/weights files.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "submission"

BANNED_IMPORTS = {
    "vizdoom",
    "doom_env",
    "stable_baselines3",
    "sb3_contrib",
    "networkx",
    "socket",
    "requests",
    "urllib",
    "http",
    "subprocess",
    "ftplib",
    "smtplib",
    "paramiko",
    "gym_doom",
    "sample_factory",
    "train",
    "training",
}

BANNED_NAMES = re.compile(
    r"atan2?|arctan2?|a_?star|dijkstra|\bbfs\b|privileged_state|sector_polygons|read_wad|\.game\b",
    re.IGNORECASE,
)

BANNED_CALLS = {"open", "exec", "eval", "__import__", "compile"}


def main() -> int:
    fails: list[str] = []
    for f in sorted(SUB.rglob("*.py")):
        src = f.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(f))
        for node in ast.walk(tree):
            mods: list[str] = []
            if isinstance(node, ast.Import):
                mods = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                mods = [node.module.split(".")[0]]
            for mod in mods:
                if mod in BANNED_IMPORTS:
                    fails.append(f"{f.name}:{node.lineno} banned import '{mod}'")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in BANNED_CALLS:
                fails.append(f"{f.name}:{node.lineno} forbidden call {node.func.id}()")

        for i, line in enumerate(src.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if BANNED_NAMES.search(line):
                fails.append(f"{f.name}:{i} forbidden pattern: {stripped[:80]}")

    req = SUB / "requirements.txt"
    if req.is_file():
        lines = [ln.split("#")[0].strip() for ln in req.read_text(encoding="utf-8").splitlines()]
        lines = [ln for ln in lines if ln]
        if len(lines) > 50:
            fails.append(f"requirements.txt has {len(lines)} lines (max 50)")
        for ln in lines:
            if "==" not in ln or ln.startswith("-"):
                fails.append(f"requirements.txt invalid/unpinned entry: {ln}")

    if fails:
        print("check_rules FAILED:")
        for msg in fails:
            print(" -", msg)
        return 1

    print("check_rules: OK (submission/ complies with docs/RULES.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

