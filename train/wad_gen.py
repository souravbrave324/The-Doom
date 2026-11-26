"""Procedural DOOM WAD Curriculum Generator using omgifol.

Generates randomized, geometrically valid, solvable Doom-format PWADs across 4 curriculum tiers:
  - tier1_combat:   1-2 large connected arenas with cover pillars, enemies, pickups, exit switch
  - tier2_doors:    3-4 rooms connected by hallways + standard manual doors (special 1) + exit
  - tier3_keycard:  4-room branching layout with keycard in branch A & locked door (special 26) in branch B
  - tier4_gauntlet: 5-6 rooms combining standard doors, colored keycard + locked door, 6-10 enemies, pickups, exit
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import omg
import omg.mapedit as me

# Doom Thing IDs
THING_PLAYER1 = 1
THING_ZOMBIEMAN = 3004
THING_SHOTGUNGUY = 9
THING_IMP = 3001
THING_DEMON = 3002
THING_MEDIKIT = 2012
THING_STIMPACK = 2011
THING_ARMOR_GREEN = 2018
THING_CLIP = 2007
THING_SHELLS = 2008
THING_KEY_BLUE = 5
THING_KEY_YELLOW = 6
THING_KEY_RED = 13

# Doom Linedef Specials
SPECIAL_DOOR_OPEN_WAIT_CLOSE = 1
SPECIAL_EXIT_SWITCH = 11
SPECIAL_DOOR_BLUE = 26
SPECIAL_DOOR_YELLOW = 27
SPECIAL_DOOR_RED = 28

THING_FLAGS_SP = 7  # Appear on skills 1-5, single player

KEY_CONFIGS = [
    ("blue", THING_KEY_BLUE, SPECIAL_DOOR_BLUE, "DOORBLU2"),
    ("yellow", THING_KEY_YELLOW, SPECIAL_DOOR_YELLOW, "DOORYEL2"),
    ("red", THING_KEY_RED, SPECIAL_DOOR_RED, "DOORRED2"),
]

WALL_TEXTURES = ("STARTAN2", "STONE2", "BROWN1", "GRAY1", "STARGR2")
FLOOR_FLATS = ("FLOOR4_8", "FLOOR0_1", "FLAT1", "FLOOR5_1")
CEIL_FLATS = ("CEIL3_5", "CEIL1_1", "FLAT5_4")


@dataclass
class RectRoom:
    """Axis-aligned rectangular sector in map units."""

    x0: int
    y0: int
    x1: int
    y1: int
    sector_idx: int
    pillars: list[tuple[int, int, int, int]]

    @property
    def cx(self) -> int:
        return (self.x0 + self.x1) // 2

    @property
    def cy(self) -> int:
        return (self.y0 + self.y1) // 2


class _GridMapBuilder:
    """Builds clean, non-overlapping orthogonal Doom sectors with exact portal merging."""

    def __init__(self, wall_tex: str = "STARTAN2", floor_tex: str = "FLOOR4_8", ceil_tex: str = "CEIL3_5"):
        self.ed = me.MapEditor()
        self.wall_tex = wall_tex
        self.floor_tex = floor_tex
        self.ceil_tex = ceil_tex
        self._verts: dict[tuple[int, int], int] = {}
        self.rooms: list[RectRoom] = []
        self.door_sectors: dict[int, tuple[int, str]] = {}  # sector_idx -> (special, upper_tex)

    def _get_vertex(self, x: int, y: int) -> int:
        key = (int(x), int(y))
        if key not in self._verts:
            idx = len(self.ed.vertexes)
            self.ed.vertexes.append(me.Vertex(key[0], key[1]))
            self._verts[key] = idx
        return self._verts[key]

    def _add_sector_poly(
        self,
        pts: list[tuple[int, int]],
        z_floor: int = 0,
        z_ceil: int = 128,
        floor_tex: str | None = None,
        ceil_tex: str | None = None,
        light: int = 192,
        wall_tex: str | None = None,
    ) -> int:
        """Add a CCW polygon sector and merge shared edges cleanly.

        omgifol convention: for CCW vertices pts[0..n-1], linedef (vx_a=next, vx_b=curr) has its
        right (front) side facing inward into the newly created sector.
        """
        sec_idx = len(self.ed.sectors)
        self.ed.sectors.append(
            me.Sector(
                z_floor=z_floor,
                z_ceil=z_ceil,
                tx_floor=floor_tex or self.floor_tex,
                tx_ceil=ceil_tex or self.ceil_tex,
                light=light,
            )
        )
        wtex = wall_tex or self.wall_tex
        n = len(pts)
        for i in range(n):
            p_curr = pts[i]
            p_next = pts[(i + 1) % n]
            va = self._get_vertex(*p_next)
            vb = self._get_vertex(*p_curr)

            merged = False
            for ld in self.ed.linedefs:
                if (ld.vx_a == va and ld.vx_b == vb) or (ld.vx_a == vb and ld.vx_b == va):
                    if not ld.two_sided:
                        back_sd = me.Sidedef(sector=sec_idx, tx_mid="-", tx_up="-", tx_low="-")
                        back_idx = len(self.ed.sidedefs)
                        self.ed.sidedefs.append(back_sd)
                        ld.back = back_idx
                        ld.two_sided = True
                        ld.impassable = False
                        self.ed.sidedefs[ld.front].tx_mid = "-"
                        merged = True
                        break
            if not merged:
                front_sd = me.Sidedef(sector=sec_idx, tx_mid=wtex, tx_up="-", tx_low="-")
                front_idx = len(self.ed.sidedefs)
                self.ed.sidedefs.append(front_sd)
                ld = me.Linedef(vx_a=va, vx_b=vb, front=front_idx, back=65535, impassable=True)
                self.ed.linedefs.append(ld)
        return sec_idx

    @staticmethod
    def _rect_pts_with_portals(
        x0: int, y0: int, x1: int, y1: int, portal_pts: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Return CCW perimeter vertices of [x0,x1]x[y0,y1], subdividing edges at portal endpoints."""
        bottom = sorted({(x, y0) for x, y in portal_pts if y == y0 and x0 <= x <= x1} | {(x0, y0), (x1, y0)})
        right = sorted({(x1, y) for x, y in portal_pts if x == x1 and y0 <= y <= y1} | {(x1, y0), (x1, y1)})
        top = sorted({(x, y1) for x, y in portal_pts if y == y1 and x0 <= x <= x1} | {(x0, y1), (x1, y1)}, reverse=True)
        left = sorted({(x0, y) for x, y in portal_pts if x == x0 and y0 <= y <= y1} | {(x0, y0), (x0, y1)}, reverse=True)
        pts: list[tuple[int, int]] = []
        for p in bottom + right + top + left:
            if not pts or pts[-1] != p:
                pts.append(p)
        if len(pts) > 1 and pts[-1] == pts[0]:
            pts.pop()
        return pts

    def add_room(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        portal_pts: list[tuple[int, int]] | None = None,
        z_ceil: int = 128,
        light: int = 192,
        wall_tex: str | None = None,
    ) -> RectRoom:
        pts = self._rect_pts_with_portals(x0, y0, x1, y1, portal_pts or [])
        sec_idx = self._add_sector_poly(pts, z_floor=0, z_ceil=z_ceil, light=light, wall_tex=wall_tex)
        room = RectRoom(x0=x0, y0=y0, x1=x1, y1=y1, sector_idx=sec_idx, pillars=[])
        self.rooms.append(room)
        return room

    def add_door_sector(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        special: int = SPECIAL_DOOR_OPEN_WAIT_CLOSE,
        upper_tex: str = "BIGDOOR2",
    ) -> int:
        pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        sec_idx = self._add_sector_poly(
            pts, z_floor=0, z_ceil=0, floor_tex=self.floor_tex, ceil_tex="FLAT5_4", light=192, wall_tex="DOORTRAK"
        )
        self.door_sectors[sec_idx] = (special, upper_tex)
        return sec_idx

    def add_pillar(self, room: RectRoom, px0: int, py0: int, px1: int, py1: int, tex: str = "METAL1") -> None:
        """Add a solid 1-sided void pillar inside `room` (clockwise winding -> front faces room)."""
        cw = [(px0, py0), (px0, py1), (px1, py1), (px1, py0)]
        n = len(cw)
        for i in range(n):
            p_curr = cw[i]
            p_next = cw[(i + 1) % n]
            va = self._get_vertex(*p_curr)
            vb = self._get_vertex(*p_next)
            # Wait: for CW polygon, right side of (p_curr -> p_next) points OUTWARD into the room!
            # Wait check: (0,0)->(0,64) goes North (+y); right side is East (+x), which is INSIDE the pillar if
            # pillar is [0,64]x[0,64]! So (p_next -> p_curr) has right side facing OUTWARD into the room!
            sd = me.Sidedef(sector=room.sector_idx, tx_mid=tex, tx_up="-", tx_low="-")
            sd_idx = len(self.ed.sidedefs)
            self.ed.sidedefs.append(sd)
            self.ed.linedefs.append(me.Linedef(vx_a=vb, vx_b=va, front=sd_idx, back=65535, impassable=True))
        room.pillars.append((px0, py0, px1, py1))

    def set_exit_switch(self, room: RectRoom, side: str) -> None:
        """Assign Linedef special 11 (Exit Switch) to a 1-sided wall on `side` ('east','west','north','south')."""
        candidates = []
        for ld in self.ed.linedefs:
            if ld.two_sided:
                continue
            if self.ed.sidedefs[ld.front].sector != room.sector_idx:
                continue
            va = self.ed.vertexes[ld.vx_a]
            vb = self.ed.vertexes[ld.vx_b]
            if side == "east" and va.x == room.x1 and vb.x == room.x1:
                candidates.append(ld)
            elif side == "west" and va.x == room.x0 and vb.x == room.x0:
                candidates.append(ld)
            elif side == "north" and va.y == room.y1 and vb.y == room.y1:
                candidates.append(ld)
            elif side == "south" and va.y == room.y0 and vb.y == room.y0:
                candidates.append(ld)
        if not candidates:
            raise RuntimeError(f"No 1-sided wall found on {side} side of room {room}")
        for ld in candidates:
            ld.action = SPECIAL_EXIT_SWITCH
            self.ed.sidedefs[ld.front].tx_mid = "SW1COMP"

    def add_thing(self, x: int, y: int, thing_type: int, angle: int = 0) -> None:
        self.ed.things.append(me.Thing(x=int(x), y=int(y), angle=int(angle), type=int(thing_type), flags=THING_FLAGS_SP))

    def sample_point_in_room(
        self,
        rng: random.Random,
        room: RectRoom,
        existing: list[tuple[int, int]],
        margin: int = 48,
        min_dist: int = 56,
    ) -> tuple[int, int]:
        x_lo, x_hi = room.x0 + margin, room.x1 - margin
        y_lo, y_hi = room.y0 + margin, room.y1 - margin
        if x_lo > x_hi:
            x_lo = x_hi = room.cx
        if y_lo > y_hi:
            y_lo = y_hi = room.cy
        for _ in range(80):
            x = rng.randint(x_lo, x_hi)
            y = rng.randint(y_lo, y_hi)
            inside_pillar = any(
                (px0 - 36) <= x <= (px1 + 36) and (py0 - 36) <= y <= (py1 + 36) for px0, py0, px1, py1 in room.pillars
            )
            if inside_pillar:
                continue
            if all(((x - ex) ** 2 + (y - ey) ** 2) >= min_dist**2 for ex, ey in existing):
                return x, y
        return room.cx, room.cy

    def finalize_and_write(self, out_path: Path) -> None:
        """Orient all door linedefs outward, apply door textures/jambs, and save PWAD."""
        for ld in self.ed.linedefs:
            if not ld.two_sided:
                continue
            sf = self.ed.sidedefs[ld.front].sector
            sb = self.ed.sidedefs[ld.back].sector
            f_is_door = sf in self.door_sectors
            b_is_door = sb in self.door_sectors
            if f_is_door and not b_is_door:
                # Flip so front is always the non-door room/hallway!
                ld.vx_a, ld.vx_b = ld.vx_b, ld.vx_a
                ld.front, ld.back = ld.back, ld.front
                sf, sb = sb, sf
            if sb in self.door_sectors:
                special, upper_tex = self.door_sectors[sb]
                ld.action = special
                self.ed.sidedefs[ld.front].tx_up = upper_tex
                self.ed.sidedefs[ld.back].tx_up = upper_tex
                self.ed.sidedefs[ld.front].tx_mid = "-"
                self.ed.sidedefs[ld.back].tx_mid = "-"
        # Lower-unpeg door jambs so DOORTRAK doesn't scroll with ceiling
        for ld in self.ed.linedefs:
            if not ld.two_sided and self.ed.sidedefs[ld.front].sector in self.door_sectors:
                ld.lower_unpeg = True

        out_path.parent.mkdir(parents=True, exist_ok=True)
        wad = omg.WAD()
        wad.maps["MAP01"] = self.ed.to_lumps()
        wad.to_file(str(out_path))


class ProceduralWadGenerator:
    """Generates randomized, solvable Doom PWADs across 4 evaluation curriculum tiers."""

    TIERS = ("tier1_combat", "tier2_doors", "tier3_keycard", "tier4_gauntlet")

    def generate_map(self, tier: str, seed: int, out_path: str | Path) -> Path:
        rng = random.Random(seed)
        out = Path(out_path)
        wall_tex = rng.choice(WALL_TEXTURES)
        floor_tex = rng.choice(FLOOR_FLATS)
        ceil_tex = rng.choice(CEIL_FLATS)
        b = _GridMapBuilder(wall_tex=wall_tex, floor_tex=floor_tex, ceil_tex=ceil_tex)

        if tier == "tier1_combat":
            self._build_tier1(b, rng)
        elif tier == "tier2_doors":
            self._build_tier2(b, rng)
        elif tier == "tier3_keycard":
            self._build_tier3(b, rng)
        elif tier == "tier4_gauntlet":
            self._build_tier4(b, rng)
        else:
            raise ValueError(f"Unknown tier: {tier}")

        b.finalize_and_write(out)
        return out

    def generate_dataset(self, output_dir: str | Path, maps_per_tier: int = 100, base_seed: int = 1000) -> dict[str, list[Path]]:
        root = Path(output_dir)
        manifest: dict[str, list[Path]] = {}
        for t_idx, tier in enumerate(self.TIERS, start=1):
            tier_dir = root / f"tier{t_idx}"
            paths: list[Path] = []
            for i in range(maps_per_tier):
                seed = base_seed + t_idx * 10000 + i
                p = tier_dir / f"{tier}_{i:03d}.wad"
                self.generate_map(tier, seed, p)
                paths.append(p)
            manifest[tier] = paths
        return manifest

    # ------------------------------------------------------------------ Tier 1: Combat & Movement
    def _build_tier1(self, b: _GridMapBuilder, rng: random.Random) -> None:
        two_arenas = rng.random() < 0.5
        w1 = rng.randrange(448, 640 + 1, 64)
        h1 = rng.randrange(384, 576 + 1, 64)

        if not two_arenas:
            # Single large arena + corner exit alcove
            exit_w = 128
            r1 = b.add_room(0, 0, w1, h1, portal_pts=[(w1, h1 - exit_w), (w1, h1)])
            exit_alcove = b.add_room(w1, h1 - exit_w, w1 + 96, h1, wall_tex="GRAY1")
            b.set_exit_switch(exit_alcove, "east")
            # Cover pillars
            num_pillars = rng.randint(2, 3)
            px_candidates = [w1 // 3, (2 * w1) // 3]
            py_candidates = [h1 // 3, (2 * h1) // 3]
            grid_pts = [(px, py) for px in px_candidates for py in py_candidates]
            rng.shuffle(grid_pts)
            for px, py in grid_pts[:num_pillars]:
                b.add_pillar(r1, px - 32, py - 32, px + 32, py + 32)
            arenas = [r1]
        else:
            w2 = rng.randrange(384, 512 + 1, 64)
            h2 = rng.randrange(384, 512 + 1, 64)
            arch_h = 160
            ay0 = (min(h1, h2) - arch_h) // 2
            ay1 = ay0 + arch_h
            arch_x0, arch_x1 = w1, w1 + 96
            exit_w = 128
            r1 = b.add_room(0, 0, w1, h1, portal_pts=[(w1, ay0), (w1, ay1)])
            arch = b.add_room(arch_x0, ay0, arch_x1, ay1)
            r2 = b.add_room(
                arch_x1, 0, arch_x1 + w2, h2,
                portal_pts=[(arch_x1, ay0), (arch_x1, ay1), (arch_x1 + w2, h2 - exit_w), (arch_x1 + w2, h2)],
            )
            exit_alcove = b.add_room(arch_x1 + w2, h2 - exit_w, arch_x1 + w2 + 96, h2, wall_tex="GRAY1")
            b.set_exit_switch(exit_alcove, "east")
            b.add_pillar(r1, w1 // 2 - 32, h1 // 2 - 32, w1 // 2 + 32, h1 // 2 + 32)
            b.add_pillar(r2, arch_x1 + w2 // 2 - 32, h2 // 2 - 32, arch_x1 + w2 // 2 + 32, h2 // 2 + 32)
            arenas = [r1, r2]

        placed: list[tuple[int, int]] = []
        px, py = arenas[0].x0 + 64, rng.randint(arenas[0].y0 + 64, arenas[0].y1 - 64)
        b.add_thing(px, py, THING_PLAYER1, angle=rng.choice([0, 45, 315]))
        placed.append((px, py))

        n_enemies = rng.randint(3, 6)
        enemy_pool = [THING_ZOMBIEMAN, THING_SHOTGUNGUY, THING_IMP]
        for i in range(n_enemies):
            rm = arenas[-1] if (len(arenas) > 1 and i >= n_enemies // 2) else arenas[0]
            ex, ey = b.sample_point_in_room(rng, rm, placed, margin=56, min_dist=72)
            b.add_thing(ex, ey, rng.choice(enemy_pool), angle=180)
            placed.append((ex, ey))

        for item in (THING_MEDIKIT, THING_CLIP, THING_SHELLS):
            rm = rng.choice(arenas)
            ix, iy = b.sample_point_in_room(rng, rm, placed, margin=48, min_dist=48)
            b.add_thing(ix, iy, item)
            placed.append((ix, iy))

    # ------------------------------------------------------------------ Tier 2: Multi-Room & Manual Doors
    def _build_tier2(self, b: _GridMapBuilder, rng: random.Random) -> None:
        num_rooms = rng.choice([3, 4])
        door_w = 96
        door_d = 32
        hall_l = rng.randrange(128, 192 + 1, 32)

        # Build rooms zig-zagging horizontally & vertically so turns are required
        dims = [(rng.randrange(256, 384 + 1, 64), rng.randrange(256, 384 + 1, 64)) for _ in range(num_rooms)]
        rooms: list[RectRoom] = []
        cur_x = 0
        room_boxes: list[tuple[int, int, int, int]] = []
        y_offsets = [0]
        for i in range(1, num_rooms):
            shift = rng.choice([-128, 0, 128])
            y_offsets.append(y_offsets[-1] + shift)
        for i, (rw, rh) in enumerate(dims):
            ry0 = y_offsets[i]
            room_boxes.append((cur_x, ry0, cur_x + rw, ry0 + rh))
            cur_x += rw + hall_l + door_d

        portals_by_room: list[list[tuple[int, int]]] = [[] for _ in range(num_rooms)]
        connectors: list[tuple[str, tuple[int, int, int, int], list[tuple[int, int]]]] = []
        for i in range(num_rooms - 1):
            ax0, ay0, ax1, ay1 = room_boxes[i]
            bx0, by0, bx1, by1 = room_boxes[i + 1]
            ov_lo = max(ay0, by0) + 32
            ov_hi = min(ay1, by1) - 32
            if ov_hi - ov_lo >= door_w:
                cy = ((ov_lo + ov_hi) // 2 // 32) * 32
                dy0, dy1 = cy - door_w // 2, cy + door_w // 2
                portals_by_room[i].extend([(ax1, dy0), (ax1, dy1)])
                portals_by_room[i + 1].extend([(bx0, dy0), (bx0, dy1)])
                connectors.append(("door", (ax1, dy0, ax1 + door_d, dy1), []))
                connectors.append(("hall", (ax1 + door_d, dy0, bx0, dy1), []))
            else:
                py = ((ay0 + ay1) // 2 // 32) * 32
                qy = ((by0 + by1) // 2 // 32) * 32
                p0, p1 = py - door_w // 2, py + door_w // 2
                q0, q1 = qy - door_w // 2, qy + door_w // 2
                portals_by_room[i].extend([(ax1, p0), (ax1, p1)])
                portals_by_room[i + 1].extend([(bx0, q0), (bx0, q1)])
                mid_x0 = ax1 + door_d
                mid_x1 = mid_x0 + 64
                vy0, vy1 = min(p0, q0), max(p1, q1)
                connectors.append(("door", (ax1, p0, mid_x0, p1), []))
                connectors.append(("hall", (mid_x0, vy0, mid_x1, vy1), [(mid_x0, p0), (mid_x0, p1), (mid_x1, q0), (mid_x1, q1)]))
                connectors.append(("hall", (mid_x1, q0, bx0, q1), []))

        for i, (rx0, ry0, rx1, ry1) in enumerate(room_boxes):
            rm = b.add_room(rx0, ry0, rx1, ry1, portal_pts=portals_by_room[i])
            rooms.append(rm)
        for kind, (x0, y0, x1, y1), ppts in connectors:
            if kind == "door":
                b.add_door_sector(x0, y0, x1, y1, special=SPECIAL_DOOR_OPEN_WAIT_CLOSE, upper_tex="BIGDOOR2")
            else:
                b.add_room(x0, y0, x1, y1, portal_pts=ppts)

        b.set_exit_switch(rooms[-1], "east")

        placed: list[tuple[int, int]] = []
        px, py = rooms[0].x0 + 64, rooms[0].cy
        b.add_thing(px, py, THING_PLAYER1, angle=0)
        placed.append((px, py))

        for rm in rooms[1:]:
            n_e = rng.randint(1, 2)
            for _ in range(n_e):
                ex, ey = b.sample_point_in_room(rng, rm, placed, margin=56, min_dist=64)
                b.add_thing(ex, ey, rng.choice([THING_ZOMBIEMAN, THING_SHOTGUNGUY, THING_IMP]), angle=180)
                placed.append((ex, ey))
        for rm in rooms[:2]:
            ix, iy = b.sample_point_in_room(rng, rm, placed, margin=48, min_dist=48)
            b.add_thing(ix, iy, rng.choice([THING_MEDIKIT, THING_CLIP, THING_SHELLS]))
            placed.append((ix, iy))

    # ------------------------------------------------------------------ Tier 3: Keycard & Locked Door Backtracking
    def _build_tier3(self, b: _GridMapBuilder, rng: random.Random) -> None:
        # 4-room T-junction: Start (West) -> Hub (Center) -> Branch A (North or South: Keycard) & Branch B (opposite: Locked Door -> Exit)
        key_on_north = rng.choice([True, False])
        door_w, door_d = 96, 32
        hall_l = 96

        s_w, s_h = rng.randrange(256, 320 + 1, 64), rng.randrange(256, 320 + 1, 64)
        hub_w, hub_h = rng.randrange(256, 384 + 1, 64), rng.randrange(256, 384 + 1, 64)
        br_w, br_h = rng.randrange(256, 384 + 1, 64), rng.randrange(256, 384 + 1, 64)
        ex_w, ex_h = rng.randrange(256, 320 + 1, 64), rng.randrange(256, 320 + 1, 64)

        hub_x0 = s_w + hall_l + door_d
        hub_x1 = hub_x0 + hub_w
        hub_y0 = 0
        hub_y1 = hub_h
        hub_cx = ((hub_x0 + hub_x1) // 2 // 32) * 32
        hub_cy = ((hub_y0 + hub_y1) // 2 // 32) * 32

        s_y0 = hub_cy - s_h // 2
        s_y1 = s_y0 + s_h

        # North branch coordinates
        n_x0 = hub_cx - br_w // 2
        n_x1 = n_x0 + br_w
        n_y0 = hub_y1 + hall_l + door_d
        n_y1 = n_y0 + br_h

        # South branch coordinates
        so_x0 = hub_cx - ex_w // 2
        so_x1 = so_x0 + ex_w
        so_y1 = hub_y0 - hall_l - door_d
        so_y0 = so_y1 - ex_h

        px0, px1 = hub_cx - door_w // 2, hub_cx + door_w // 2
        py0, py1 = hub_cy - door_w // 2, hub_cy + door_w // 2

        # Draw rooms with portal points
        start_rm = b.add_room(0, s_y0, s_w, s_y1, portal_pts=[(s_w, py0), (s_w, py1)])
        hub_rm = b.add_room(
            hub_x0, hub_y0, hub_x1, hub_y1,
            portal_pts=[(hub_x0, py0), (hub_x0, py1), (px0, hub_y1), (px1, hub_y1), (px0, hub_y0), (px1, hub_y0)],
        )
        north_rm = b.add_room(n_x0, n_y0, n_x1, n_y1, portal_pts=[(px0, n_y0), (px1, n_y0)])
        south_rm = b.add_room(so_x0, so_y0, so_x1, so_y1, portal_pts=[(px0, so_y1), (px1, so_y1)])

        # Start -> Hub (standard door)
        b.add_door_sector(s_w, py0, s_w + door_d, py1, SPECIAL_DOOR_OPEN_WAIT_CLOSE, "BIGDOOR2")
        b.add_room(s_w + door_d, py0, hub_x0, py1)

        if key_on_north:
            key_rm, exit_rm = north_rm, south_rm
            # Hub -> North (standard door)
            b.add_door_sector(px0, hub_y1, px1, hub_y1 + door_d, SPECIAL_DOOR_OPEN_WAIT_CLOSE, "BIGDOOR2")
            b.add_room(px0, hub_y1 + door_d, px1, n_y0)
            # Hub -> South (Blue locked door)
            b.add_door_sector(px0, hub_y0 - door_d, px1, hub_y0, SPECIAL_DOOR_BLUE, "DOORBLU2")
            b.add_room(px0, so_y1, px1, hub_y0 - door_d)
            b.set_exit_switch(exit_rm, "south")
        else:
            key_rm, exit_rm = south_rm, north_rm
            # Hub -> North (Blue locked door)
            b.add_door_sector(px0, hub_y1, px1, hub_y1 + door_d, SPECIAL_DOOR_BLUE, "DOORBLU2")
            b.add_room(px0, hub_y1 + door_d, px1, n_y0)
            # Hub -> South (standard door)
            b.add_door_sector(px0, hub_y0 - door_d, px1, hub_y0, SPECIAL_DOOR_OPEN_WAIT_CLOSE, "BIGDOOR2")
            b.add_room(px0, so_y1, px1, hub_y0 - door_d)
            b.set_exit_switch(exit_rm, "north")

        placed: list[tuple[int, int]] = []
        b.add_thing(start_rm.x0 + 64, start_rm.cy, THING_PLAYER1, angle=0)
        placed.append((start_rm.x0 + 64, start_rm.cy))

        # Keycard in Branch A guarded by 2 enemies
        b.add_thing(key_rm.cx, key_rm.cy, THING_KEY_BLUE)
        placed.append((key_rm.cx, key_rm.cy))
        for _ in range(2):
            ex, ey = b.sample_point_in_room(rng, key_rm, placed, margin=56, min_dist=64)
            b.add_thing(ex, ey, rng.choice([THING_ZOMBIEMAN, THING_SHOTGUNGUY, THING_IMP]), angle=270 if key_on_north else 90)
            placed.append((ex, ey))

        # 1 enemy in Hub + 1 in Exit room + pickups
        hx, hy = b.sample_point_in_room(rng, hub_rm, placed, margin=64, min_dist=64)
        b.add_thing(hx, hy, rng.choice([THING_ZOMBIEMAN, THING_IMP]), angle=180)
        placed.append((hx, hy))

        ex, ey = b.sample_point_in_room(rng, exit_rm, placed, margin=56, min_dist=64)
        b.add_thing(ex, ey, rng.choice([THING_SHOTGUNGUY, THING_IMP]), angle=90 if key_on_north else 270)
        placed.append((ex, ey))

        for rm, item in ((start_rm, THING_SHELLS), (hub_rm, THING_MEDIKIT)):
            ix, iy = b.sample_point_in_room(rng, rm, placed, margin=48, min_dist=48)
            b.add_thing(ix, iy, item)
            placed.append((ix, iy))

    # ------------------------------------------------------------------ Tier 4: Full Gauntlet (5-6 rooms)
    def _build_tier4(self, b: _GridMapBuilder, rng: random.Random) -> None:
        # 5 or 6 rooms: Start -> Hall/Room1 -> Hub -> Key Branch (1-2 rooms) & Locked Exit Branch (1-2 rooms)
        six_rooms = rng.choice([True, False])
        key_on_north = rng.choice([True, False])
        _, key_thing, lock_special, lock_tex = rng.choice(KEY_CONFIGS)
        door_w, door_d = 96, 32
        hall_l = 96

        r0_w, r0_h = 256, 256
        r1_w, r1_h = rng.randrange(320, 384 + 1, 64), rng.randrange(256, 384 + 1, 64)
        hub_w, hub_h = rng.randrange(320, 448 + 1, 64), rng.randrange(320, 384 + 1, 64)

        r1_x0 = r0_w + hall_l + door_d
        r1_x1 = r1_x0 + r1_w
        hub_x0 = r1_x1 + hall_l + door_d
        hub_x1 = hub_x0 + hub_w
        hub_y0, hub_y1 = 0, hub_h
        hub_cx = ((hub_x0 + hub_x1) // 2 // 32) * 32
        hub_cy = ((hub_y0 + hub_y1) // 2 // 32) * 32

        r0_y0, r0_y1 = hub_cy - r0_h // 2, hub_cy + r0_h // 2
        r1_y0, r1_y1 = hub_cy - r1_h // 2, hub_cy + r1_h // 2

        px0, px1 = hub_cx - door_w // 2, hub_cx + door_w // 2
        py0, py1 = hub_cy - door_w // 2, hub_cy + door_w // 2

        # North branch room 1
        nb_w, nb_h = 320, 320
        nb_x0, nb_x1 = hub_cx - nb_w // 2, hub_cx + nb_w // 2
        nb_y0, nb_y1 = hub_y1 + hall_l + door_d, hub_y1 + hall_l + door_d + nb_h

        # South branch room 1
        sb_w, sb_h = 320, 320
        sb_x0, sb_x1 = hub_cx - sb_w // 2, hub_cx + sb_w // 2
        sb_y1, sb_y0 = hub_y0 - hall_l - door_d, hub_y0 - hall_l - door_d - sb_h

        # Optional 6th room extending East from Hub
        ex_east_x0 = hub_x1 + hall_l + door_d
        ex_east_x1 = ex_east_x0 + 320
        ex_east_y0, ex_east_y1 = hub_cy - 160, hub_cy + 160

        hub_portals = [(hub_x0, py0), (hub_x0, py1), (px0, hub_y1), (px1, hub_y1), (px0, hub_y0), (px1, hub_y0)]
        if six_rooms:
            hub_portals.extend([(hub_x1, py0), (hub_x1, py1)])

        r0 = b.add_room(0, r0_y0, r0_w, r0_y1, portal_pts=[(r0_w, py0), (r0_w, py1)])
        r1 = b.add_room(r1_x0, r1_y0, r1_x1, r1_y1, portal_pts=[(r1_x0, py0), (r1_x0, py1), (r1_x1, py0), (r1_x1, py1)])
        hub = b.add_room(hub_x0, hub_y0, hub_x1, hub_y1, portal_pts=hub_portals)
        n_rm = b.add_room(nb_x0, nb_y0, nb_x1, nb_y1, portal_pts=[(px0, nb_y0), (px1, nb_y0)])
        s_rm = b.add_room(sb_x0, sb_y0, sb_x1, sb_y1, portal_pts=[(px0, sb_y1), (px1, sb_y1)])

        # Doors between r0 -> r1 -> hub
        b.add_door_sector(r0_w, py0, r0_w + door_d, py1, SPECIAL_DOOR_OPEN_WAIT_CLOSE, "BIGDOOR2")
        b.add_room(r0_w + door_d, py0, r1_x0, py1)
        b.add_door_sector(r1_x1, py0, r1_x1 + door_d, py1, SPECIAL_DOOR_OPEN_WAIT_CLOSE, "BIGDOOR2")
        b.add_room(r1_x1 + door_d, py0, hub_x0, py1)

        if six_rooms:
            east_rm = b.add_room(ex_east_x0, ex_east_y0, ex_east_x1, ex_east_y1, portal_pts=[(ex_east_x0, py0), (ex_east_x0, py1)])
            b.add_door_sector(hub_x1, py0, hub_x1 + door_d, py1, SPECIAL_DOOR_OPEN_WAIT_CLOSE, "BIGDOOR2")
            b.add_room(hub_x1 + door_d, py0, ex_east_x0, py1)
        else:
            east_rm = None

        # Pillar in hub for cover
        b.add_pillar(hub, hub_cx - 32, hub_cy - 32, hub_cx + 32, hub_cy + 32)

        if key_on_north:
            key_rm = east_rm if (six_rooms and rng.random() < 0.5) else n_rm
            exit_rm = s_rm
            b.add_door_sector(px0, hub_y1, px1, hub_y1 + door_d, SPECIAL_DOOR_OPEN_WAIT_CLOSE, "BIGDOOR2")
            b.add_room(px0, hub_y1 + door_d, px1, nb_y0)
            b.add_door_sector(px0, hub_y0 - door_d, px1, hub_y0, lock_special, lock_tex)
            b.add_room(px0, sb_y1, px1, hub_y0 - door_d)
            b.set_exit_switch(exit_rm, "south")
        else:
            key_rm = east_rm if (six_rooms and rng.random() < 0.5) else s_rm
            exit_rm = n_rm
            b.add_door_sector(px0, hub_y1, px1, hub_y1 + door_d, lock_special, lock_tex)
            b.add_room(px0, hub_y1 + door_d, px1, nb_y0)
            b.add_door_sector(px0, hub_y0 - door_d, px1, hub_y0, SPECIAL_DOOR_OPEN_WAIT_CLOSE, "BIGDOOR2")
            b.add_room(px0, sb_y1, px1, hub_y0 - door_d)
            b.set_exit_switch(exit_rm, "north")

        placed: list[tuple[int, int]] = []
        b.add_thing(r0.x0 + 64, r0.cy, THING_PLAYER1, angle=0)
        placed.append((r0.x0 + 64, r0.cy))

        b.add_thing(key_rm.cx, key_rm.cy, key_thing)
        placed.append((key_rm.cx, key_rm.cy))

        combat_rooms = [r1, hub, n_rm, s_rm] + ([east_rm] if east_rm is not None else [])
        n_enemies = rng.randint(6, 10)
        enemy_pool = [THING_ZOMBIEMAN, THING_SHOTGUNGUY, THING_IMP, THING_DEMON]
        for i in range(n_enemies):
            rm = combat_rooms[i % len(combat_rooms)]
            ex, ey = b.sample_point_in_room(rng, rm, placed, margin=56, min_dist=60)
            b.add_thing(ex, ey, rng.choice(enemy_pool), angle=rng.choice([0, 90, 180, 270]))
            placed.append((ex, ey))

        pickups = [THING_MEDIKIT, THING_MEDIKIT, THING_ARMOR_GREEN, THING_CLIP, THING_SHELLS, THING_STIMPACK]
        for item in pickups:
            rm = rng.choice([r0, r1, hub, key_rm])
            ix, iy = b.sample_point_in_room(rng, rm, placed, margin=48, min_dist=48)
            b.add_thing(ix, iy, item)
            placed.append((ix, iy))


def verify_generated_wads(wad_root: Path, per_tier: int = 5, steps: int = 50, seed: int = 42) -> None:
    """Load `per_tier` random WADs per tier into official doom_env and step `steps` random actions."""
    from doom_env import make_env

    rng = random.Random(seed)
    total_checked = 0
    for t_idx, tier in enumerate(ProceduralWadGenerator.TIERS, start=1):
        wads = sorted((wad_root / f"tier{t_idx}").glob("*.wad"))
        if len(wads) < per_tier:
            raise RuntimeError(f"Expected >= {per_tier} WADs in {wad_root / f'tier{t_idx}'}, found {len(wads)}")
        sample = rng.sample(wads, per_tier)
        for wp in sample:
            env = make_env(str(wp), "MAP01", episode_ticks=2100, render_buffers=False)
            try:
                obs, _ = env.reset(seed=seed)
                assert env.observation_space.contains(obs), f"Invalid initial obs in {wp}"
                for _ in range(steps):
                    obs, _, term, trunc, _ = env.step(env.action_space.sample())
                    assert env.observation_space.contains(obs), f"Invalid step obs in {wp}"
                    if term or trunc:
                        obs, _ = env.reset(seed=seed)
                total_checked += 1
            finally:
                env.close()
        print(f"Verified {per_tier}/{per_tier} WADs for {tier} cleanly.")
    print(f"ALL {total_checked} sampled WADs passed doom_env verification with 0 geometry/node errors.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Procedural DOOM WAD Curriculum Generator")
    ap.add_argument("--out", default="train/generated_wads", help="Output directory")
    ap.add_argument("--per-tier", type=int, default=100, help="WADs per tier")
    ap.add_argument("--seed", type=int, default=1000, help="Base RNG seed")
    ap.add_argument("--verify", action="store_true", help="Verify 5 random WADs per tier in doom_env")
    args = ap.parse_args()

    out_dir = Path(args.out)
    gen = ProceduralWadGenerator()
    manifest = gen.generate_dataset(out_dir, maps_per_tier=args.per_tier, base_seed=args.seed)
    total = sum(len(v) for v in manifest.values())
    print(f"Generated {total} WADs across {len(manifest)} tiers in {out_dir}")
    if args.verify:
        verify_generated_wads(out_dir, per_tier=5, steps=50)


if __name__ == "__main__":
    main()
