#!/usr/bin/env python3
"""
real_boids_swarm_square.py
===========================
Real-hardware Boids swarm for TWO Crazyflie drones via LPS positioning,
with integrated post-flight analysis plots.

Mission profile
---------------
  1. Both drones take off to their assigned altitude planes (0.3 m separation).
  2. Fly the four corners of a square path inside the safe zone:
       C0 (0.15, 1.15)  →  C1 (2.85, 1.15)
       C1 (2.85, 1.15)  →  C2 (2.85, 3.85)
       C2 (2.85, 3.85)  →  C3 (0.15, 3.85)
       C3 (0.15, 3.85)  →  C0 (0.15, 1.15)   ← closes the loop
  3. Run Boids swarm loop for 60 s (starting from C0 after the square).
  4. Navigate to dedicated landing pads and land:
       Drone 7  →  (2.2, 2.0)
       Drone 2  →  (2.0, 3.0)

Post-flight plots (auto-generated after landing)
-------------------------------------------------
  Figure 1 – XY plane       : actual vs expected trajectory (top-down)
  Figure 2 – X / Y / Z axes : position vs time, actual vs expected
  Figure 3 – RMS error      : per-axis bar charts + rolling 3-D RMS curve
  Figure 4 – Separation     : XY and 3-D inter-drone distance vs time
  Figure 5 – Boids forces   : sep / cohesion / wall / net magnitudes per tick

Hardware config
---------------
  Drone 7 : radio://0/80/2M/E7E7E7E7E7
  Drone 2 : radio://0/80/2M/E7E7E7E7E2
  Channel : 80 / 2M  (shared)

Safe zone (XY)
--------------
  X range : [0.0, 3.0]   Y range : [1.0, 4.0]   Z max : 1.2

Usage
-----
  # Simulate (no Crazyflie hardware needed):
  python real_boids_swarm_square.py --sim

  # Real hardware flight:
  python real_boids_swarm_square.py

  # Save plots to a custom folder:
  python real_boids_swarm_square.py --sim --out ./my_results

Dependencies
------------
  pip install cflib matplotlib numpy
"""

import time
import threading
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')          # headless-safe; switch to 'TkAgg' for live windows
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ── Crazyflie imports (guarded so the script runs in plot-only / sim mode) ──
try:
    import cflib.crtp
    from cflib.crazyflie import Crazyflie
    from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
    from cflib.crazyflie.log import LogConfig
    CF_AVAILABLE = True
except ImportError:
    CF_AVAILABLE = False

# =============================================================================
#  SECTION 1 — CONSTANTS  (shared by flight code AND plot code)
# =============================================================================

# Radio URIs
URI_7 = 'radio://0/80/2M/E7E7E7E7E7'
URI_2 = 'radio://0/80/2M/E7E7E7E7E2'

# Safe zone (absolute lab coordinates, metres)
SAFE_X_MIN, SAFE_X_MAX = 0.0, 3.0
SAFE_Y_MIN, SAFE_Y_MAX = 1.0, 4.0
SAFE_Z_MIN, SAFE_Z_MAX = 0.0, 1.2
CLAMP_MARGIN = 0.15
WALL_MARGIN  = 0.35

# Flight parameters
CRUISE_Z_7  = 0.6
CRUISE_Z_2  = 0.9
TAKEOFF_DUR = 2.5
LPS_SETTLE  = 4.0

# ── Square path corners (counter-clockwise, inset by CLAMP_MARGIN) ──────────
#   C0 = bottom-left   C1 = bottom-right
#   C3 = top-left      C2 = top-right
SQ_C0 = (SAFE_X_MIN + CLAMP_MARGIN, SAFE_Y_MIN + CLAMP_MARGIN)  # (0.15, 1.15)
SQ_C1 = (SAFE_X_MAX - CLAMP_MARGIN, SAFE_Y_MIN + CLAMP_MARGIN)  # (2.85, 1.15)
SQ_C2 = (SAFE_X_MAX - CLAMP_MARGIN, SAFE_Y_MAX - CLAMP_MARGIN)  # (2.85, 3.85)
SQ_C3 = (SAFE_X_MIN + CLAMP_MARGIN, SAFE_Y_MAX - CLAMP_MARGIN)  # (0.15, 3.85)
SQUARE_CORNERS = (SQ_C0, SQ_C1, SQ_C2, SQ_C3)

# Time (seconds) to fly each side of the square; sides are ~2.7 m long
SIDE_DUR = 5.0   # s per side  (≈ 0.54 m/s average ground speed)

# Landing pads
LAND_7 = (2.2, 2.0)
LAND_2 = (2.0, 3.0)

# Boids parameters
SEP_DIST     = 0.8
SEP_WEIGHT   = 1.5
COH_WEIGHT   = 0.4
WALL_WEIGHT  = 2.0
MAX_STEP     = 0.15
BOIDS_DT     = 0.20
MISSION_TIME = 60.0
RADIO_STAGGER = 0.10

# ── Timeline derived from the square (used by synthetic generator & plots) ──
# takeoff  : 0 → T_TAKEOFF_END
# entry    : T_TAKEOFF_END → T_ENTRY_END   (fly to C0 from spawn)
# square   : T_ENTRY_END  → T_SQ_END       (4 sides × SIDE_DUR)
# boids    : T_SQ_END     → T_BOIDS_END
# landnav  : T_BOIDS_END  → T_LANDNAV_END
# land     : T_LANDNAV_END → T_TOTAL
T_TAKEOFF_END  = 3.5
T_ENTRY_END    = 7.5
T_SQ_END       = T_ENTRY_END + 4 * SIDE_DUR          # 27.5 s
T_BOIDS_END    = T_SQ_END + MISSION_TIME              # 87.5 s
T_LANDNAV_END  = T_BOIDS_END + 4.5                   # 92.0 s
T_TOTAL        = T_LANDNAV_END + 4.5                  # 96.5 s

# Plot colour palette
C7  = '#3266ad'   # Drone 7 — blue
C2  = '#c94a2a'   # Drone 2 — coral
CG  = '#888780'   # neutral / reference
CA  = '#3a8a5a'   # green  — safe / corners
CB  = '#c4870a'   # amber  — cohesion
CW  = '#7f2b8a'   # purple — wall force

# =============================================================================
#  SECTION 2 — SHARED FLIGHT STATE
# =============================================================================

_state_lock = threading.Lock()

_drone_pos = {
    URI_7: np.array([1.0, 2.0, 0.0]),
    URI_2: np.array([2.0, 3.0, 0.0]),
}

_telem_lock = threading.Lock()
_telem = {
    URI_7: {'t': [], 'ax': [], 'ay': [], 'az': [],
            'ex': [], 'ey': [], 'ez': []},
    URI_2: {'t': [], 'ax': [], 'ay': [], 'az': [],
            'ex': [], 'ey': [], 'ez': []},
}
_t0_flight = None


def _log_telem(uri, t, actual_xyz, expected_xyz):
    with _telem_lock:
        d = _telem[uri]
        d['t'].append(float(t))
        d['ax'].append(float(actual_xyz[0])); d['ay'].append(float(actual_xyz[1])); d['az'].append(float(actual_xyz[2]))
        d['ex'].append(float(expected_xyz[0])); d['ey'].append(float(expected_xyz[1])); d['ez'].append(float(expected_xyz[2]))


def _telem_arrays(uri):
    with _telem_lock:
        d = _telem[uri]
        return {k: np.array(d[k]) for k in d}

# =============================================================================
#  SECTION 3 — FLIGHT UTILITIES
# =============================================================================

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _clip_to_safe(pos: np.ndarray, target_z: float) -> np.ndarray:
    cm = CLAMP_MARGIN
    return np.array([
        _clamp(pos[0], SAFE_X_MIN + cm, SAFE_X_MAX - cm),
        _clamp(pos[1], SAFE_Y_MIN + cm, SAFE_Y_MAX - cm),
        _clamp(pos[2], target_z - 0.05, target_z + 0.05),
    ])


def _wall_repulsion(pos: np.ndarray) -> np.ndarray:
    force = np.zeros(3)
    wm = WALL_MARGIN
    excess = (SAFE_X_MIN + wm) - pos[0];  force[0] += excess if excess > 0 else 0
    excess = pos[0] - (SAFE_X_MAX - wm);  force[0] -= excess if excess > 0 else 0
    excess = (SAFE_Y_MIN + wm) - pos[1];  force[1] += excess if excess > 0 else 0
    excess = pos[1] - (SAFE_Y_MAX - wm);  force[1] -= excess if excess > 0 else 0
    return force

# =============================================================================
#  SECTION 4 — BOIDS CORE
# =============================================================================

def _boids_next_target(my_uri: str, other_uri: str, my_target_z: float) -> np.ndarray:
    """Compute next waypoint using Boids rules, locked to the drone's altitude plane."""
    with _state_lock:
        me    = _drone_pos[my_uri].copy()
        other = _drone_pos[other_uri].copy()

    diff = me - other
    diff[2] = 0.0
    dist = np.linalg.norm(diff)

    # 1. Separation
    if dist < SEP_DIST and dist > 1e-4:
        sep_force = (diff / dist) * (SEP_DIST - dist)
    else:
        sep_force = np.zeros(3)

    # 2. Cohesion
    midpoint  = (me + other) / 2.0
    coh_force = midpoint - me
    coh_force[2] = 0.0

    # 3. Wall repulsion
    wall_force = _wall_repulsion(me)

    net = (SEP_WEIGHT  * sep_force +
           COH_WEIGHT  * coh_force +
           WALL_WEIGHT * wall_force)
    net[2] = 0.0

    norm = np.linalg.norm(net)
    if norm > MAX_STEP:
        net = net / norm * MAX_STEP

    return _clip_to_safe(me + net, my_target_z)

# =============================================================================
#  SECTION 5 — LPS POSITION READ
# =============================================================================

def _read_lps_position(cf) -> np.ndarray:
    print(f"  [LPS] Settling for {LPS_SETTLE} s ...")
    time.sleep(LPS_SETTLE)
    pos  = {}
    done = [False]

    def _cb(ts, data, lc):
        if not done[0]:
            pos['x'] = data.get('kalman.stateX', 0.0)
            pos['y'] = data.get('kalman.stateY', 0.0)
            pos['z'] = data.get('kalman.stateZ', 0.0)
            done[0]  = True

    lc = LogConfig(name='SpawnPos', period_in_ms=100)
    lc.add_variable('kalman.stateX', 'float')
    lc.add_variable('kalman.stateY', 'float')
    lc.add_variable('kalman.stateZ', 'float')
    cf.log.add_config(lc)
    lc.data_received_cb.add_callback(_cb)
    lc.start()

    t0 = time.time()
    while not done[0] and (time.time() - t0) < 6.0:
        time.sleep(0.05)
    lc.stop()

    if not done[0]:
        raise RuntimeError("LPS position read timed out.")
    return np.array([pos['x'], pos['y'], pos['z']])


def _start_position_tracking(cf, uri: str) -> 'LogConfig':
    def _cb(ts, data, lc):
        with _state_lock:
            _drone_pos[uri] = np.array([
                data.get('kalman.stateX', _drone_pos[uri][0]),
                data.get('kalman.stateY', _drone_pos[uri][1]),
                data.get('kalman.stateZ', _drone_pos[uri][2]),
            ])

    lc = LogConfig(name='PosTrack', period_in_ms=100)
    lc.add_variable('kalman.stateX', 'float')
    lc.add_variable('kalman.stateY', 'float')
    lc.add_variable('kalman.stateZ', 'float')
    cf.log.add_config(lc)
    lc.data_received_cb.add_callback(_cb)
    lc.start()
    return lc

# =============================================================================
#  SECTION 6 — PER-DRONE FLIGHT THREAD
# =============================================================================

def _drone_thread(uri: str,
                  other_uri: str,
                  ready_evt: threading.Event,
                  go_evt:    threading.Event,
                  stop_evt:  threading.Event):

    global _t0_flight
    label = uri[-1]
    my_cruise_z = CRUISE_Z_7 if uri == URI_7 else CRUISE_Z_2
    land_x, land_y = LAND_7 if uri == URI_7 else LAND_2

    print(f"\n[Drone {label}] Connecting to {uri} ...")
    try:
        with SyncCrazyflie(uri, cf=Crazyflie(rw_cache=f'./cache_{label}')) as scf:
            cf  = scf.cf
            hlc = cf.high_level_commander

            cf.param.set_value('commander.enHighLevel', '1')
            time.sleep(0.3)
            try:
                cf.supervisor.send_arming_request(True)
            except AttributeError:
                pass
            time.sleep(0.5)

            spawn = _read_lps_position(cf)
            with _state_lock:
                _drone_pos[uri] = spawn
            print(f"[Drone {label}] Spawned at: x={spawn[0]:.2f}  y={spawn[1]:.2f}  z={spawn[2]:.2f}")

            cm = CLAMP_MARGIN
            spawn_outside = not (
                SAFE_X_MIN + cm <= spawn[0] <= SAFE_X_MAX - cm and
                SAFE_Y_MIN + cm <= spawn[1] <= SAFE_Y_MAX - cm
            )
            if spawn_outside:
                entry_x = _clamp(spawn[0], SAFE_X_MIN + cm, SAFE_X_MAX - cm)
                entry_y = _clamp(spawn[1], SAFE_Y_MIN + cm, SAFE_Y_MAX - cm)
                print(f"[Drone {label}] ⚠ Spawn outside safe zone – entering at ({entry_x:.2f},{entry_y:.2f})")
            else:
                entry_x, entry_y = spawn[0], spawn[1]
                print(f"[Drone {label}] ✓ Spawn inside safe zone.")

            pos_log = _start_position_tracking(cf, uri)

            print(f"[Drone {label}] Ready – waiting for partner ...")
            ready_evt.set()
            go_evt.wait()

            t_flight_start = time.time()

            # ── Step 1: Takeoff ──────────────────────────────────────────────
            print(f"[Drone {label}] TAKEOFF → z={my_cruise_z:.2f} m")
            hlc.takeoff(my_cruise_z, TAKEOFF_DUR)
            time.sleep(TAKEOFF_DUR + 1.0)
            _log_telem(uri, time.time() - t_flight_start,
                       _drone_pos[uri], [spawn[0], spawn[1], my_cruise_z])

            # ── Step 2: Safe-zone entry (if spawned outside) ────────────────
            if spawn_outside:
                dur = max(3.0, (abs(entry_x - spawn[0]) + abs(entry_y - spawn[1])) / 0.4)
                hlc.go_to(entry_x, entry_y, my_cruise_z, yaw=0.0,
                          duration_s=dur, relative=False)
                time.sleep(dur + 0.8)
                _log_telem(uri, time.time() - t_flight_start,
                           _drone_pos[uri], [entry_x, entry_y, my_cruise_z])

            # ── Step 3: Fly to square start corner C0 ───────────────────────
            print(f"[Drone {label}] SQUARE START → C0 {SQ_C0}")
            hlc.go_to(SQ_C0[0], SQ_C0[1], my_cruise_z, yaw=0.0,
                      duration_s=4.0, relative=False)
            time.sleep(4.5)
            _log_telem(uri, time.time() - t_flight_start,
                       _drone_pos[uri], [SQ_C0[0], SQ_C0[1], my_cruise_z])

            # ── Step 4: Traverse all four sides of the square ───────────────
            #   C0 → C1 → C2 → C3 → C0
            corner_names = ['C1 (bottom-right)', 'C2 (top-right)',
                            'C3 (top-left)',     'C0 (bottom-left, close)']
            for corner, name in zip((SQ_C1, SQ_C2, SQ_C3, SQ_C0), corner_names):
                print(f"[Drone {label}] SQUARE → {name}  ({corner[0]:.2f},{corner[1]:.2f})  "
                      f"dur={SIDE_DUR:.0f}s")
                hlc.go_to(corner[0], corner[1], my_cruise_z, yaw=0.0,
                          duration_s=SIDE_DUR, relative=False)
                time.sleep(SIDE_DUR + 0.3)
                _log_telem(uri, time.time() - t_flight_start,
                           _drone_pos[uri], [corner[0], corner[1], my_cruise_z])

            # ── Step 5: Boids swarm loop (starting from C0) ─────────────────
            if uri == URI_2:
                time.sleep(RADIO_STAGGER)

            print(f"[Drone {label}] BOIDS START  mission={MISSION_TIME:.0f}s")
            t_start   = time.time()
            tick      = 0
            prev_tx, prev_ty = SQ_C0[0], SQ_C0[1]

            while not stop_evt.is_set():
                elapsed = time.time() - t_start
                if elapsed > MISSION_TIME:
                    break

                target = _boids_next_target(uri, other_uri, my_cruise_z)
                tx, ty, tz = float(target[0]), float(target[1]), float(target[2])

                hlc.go_to(tx, ty, tz, yaw=0.0,
                          duration_s=BOIDS_DT * 1.6, relative=False)

                _log_telem(uri,
                           time.time() - t_flight_start,
                           _drone_pos[uri].copy(),
                           [tx, ty, tz])

                tick += 1
                if tick % 10 == 0:
                    print(f"[Drone {label}] BOIDS t={elapsed:5.1f}s  "
                          f"({prev_tx:.2f},{prev_ty:.2f}) → ({tx:.2f},{ty:.2f})  z={tz:.2f}")
                prev_tx, prev_ty = tx, ty
                time.sleep(BOIDS_DT)

            # ── Step 6: Fly to assigned landing pad ─────────────────────────
            print(f"[Drone {label}] LAND-NAV → ({land_x:.2f},{land_y:.2f})")
            hlc.go_to(land_x, land_y, my_cruise_z, yaw=0.0,
                      duration_s=4.0, relative=False)
            time.sleep(4.5)
            _log_telem(uri, time.time() - t_flight_start,
                       _drone_pos[uri], [land_x, land_y, my_cruise_z])

            # ── Step 7: Controlled vertical landing ────────────────────────
            print(f"[Drone {label}] LANDING ↓")
            hlc.land(0.0, duration_s=4.0)
            time.sleep(4.5)
            _log_telem(uri, time.time() - t_flight_start,
                       _drone_pos[uri], [land_x, land_y, 0.0])
            hlc.stop()

            pos_log.stop()
            print(f"[Drone {label}] Landed safely.")

    except Exception as exc:
        print(f"[Drone {label}] ERROR: {exc}")
        stop_evt.set()
        ready_evt.set()
        raise

# =============================================================================
#  SECTION 7 — SYNTHETIC DATA GENERATOR  (sim mode / testing without hardware)
# =============================================================================

def _lcg(seed):
    """Deterministic LCG pseudo-RNG yielding values in [-1, 1]."""
    s = int(seed) & 0xFFFFFFFF
    while True:
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        yield (s >> 1) / 0x7FFFFFFF - 1.0


def generate_synthetic_mission(seed7=42, seed2=77):
    """
    Simulate the complete square-path mission for both drones.

    Phase timeline
    --------------
      takeoff  0         → T_TAKEOFF_END   (3.5 s)
      entry    3.5       → T_ENTRY_END     (7.5 s)
      square   7.5       → T_SQ_END        (27.5 s, 4 × 5 s sides)
      boids    T_SQ_END  → T_BOIDS_END     (+60 s)
      landnav  ...       → T_LANDNAV_END   (+4.5 s)
      land     ...       → T_TOTAL         (+4.5 s)

    Returns
    -------
    d7, d2 : dict of numpy arrays
        Keys: t, ax, ay, az (actual), ex, ey, ez (expected/planned)
    """
    rng7 = _lcg(seed7)
    rng2 = _lcg(seed2)
    dt   = BOIDS_DT
    N    = int(T_TOTAL / dt) + 2
    T    = np.arange(N) * dt

    def clamp(v, lo, hi): return float(np.clip(v, lo, hi))

    def ease(f):
        f = clamp(f, 0, 1)
        return 2f**2 if f < 0.5 else 1 - (-2f + 2)**2 / 2

    def lerp(t, t0, t1, v0, v1):
        return v0 + (v1 - v0) * ease((t - t0) / max(t1 - t0, 1e-6))

    # Square corner sequence with time windows
    # Each side: (start_corner, end_corner, t_start, t_end)
    sq_sides = []
    corners_seq = [SQ_C0, SQ_C1, SQ_C2, SQ_C3, SQ_C0]  # close the loop
    for i in range(4):
        t0_side = T_ENTRY_END + i * SIDE_DUR
        t1_side = t0_side + SIDE_DUR
        sq_sides.append((corners_seq[i], corners_seq[i+1], t0_side, t1_side))

    def square_expected(t):
        """Return (ex, ey) along the square path for time t."""
        for (c0, c1, ts, te) in sq_sides:
            if t <= te:
                ex = lerp(t, ts, te, c0[0], c1[0])
                ey = lerp(t, ts, te, c0[1], c1[1])
                return ex, ey
        return SQ_C0  # after loop closes

    def phase(t):
        if t <= T_TAKEOFF_END:  return 'takeoff'
        if t <= T_ENTRY_END:    return 'entry'
        if t <= T_SQ_END:       return 'square'
        if t <= T_BOIDS_END:    return 'boids'
        if t <= T_LANDNAV_END:  return 'landnav'
        return 'land'

    out7 = {k: [] for k in ('t','ax','ay','az','ex','ey','ez')}
    out2 = {k: [] for k in ('t','ax','ay','az','ex','ey','ez')}

    last_bx7, last_by7 = SQ_C0
    last_bx2, last_by2 = SQ_C0

    for i, t in enumerate(T):
        if t > T_TOTAL:
            break
        ph = phase(t)
        n7 = next(rng7)
        n2 = next(rng2)

        # ── expected (planned) positions ───────────────────────────────────
        if ph == 'takeoff':
            ex7, ey7 = 1.0, 2.0;  ez7 = lerp(t, 0, T_TAKEOFF_END, 0, CRUISE_Z_7)
            ex2, ey2 = 2.0, 3.0;  ez2 = lerp(t, 0, T_TAKEOFF_END, 0, CRUISE_Z_2)

        elif ph == 'entry':
            ez7, ez2 = CRUISE_Z_7, CRUISE_Z_2
            # Both drones converge on C0 from their spawn positions
            ex7 = lerp(t, T_TAKEOFF_END, T_ENTRY_END, 1.0, SQ_C0[0])
            ey7 = lerp(t, T_TAKEOFF_END, T_ENTRY_END, 2.0, SQ_C0[1])
            ex2 = lerp(t, T_TAKEOFF_END, T_ENTRY_END, 2.0, SQ_C0[0])
            ey2 = lerp(t, T_TAKEOFF_END, T_ENTRY_END, 3.0, SQ_C0[1])

        elif ph == 'square':
            ez7, ez2 = CRUISE_Z_7, CRUISE_Z_2
            # Both drones follow the same square but are offset in time
            # Drone 7 leads; Drone 2 lags by half a side (2.5 s)
            ex7, ey7 = square_expected(t)
            ex2, ey2 = square_expected(max(T_ENTRY_END, t - SIDE_DUR / 2))

        elif ph == 'boids':
            ez7, ez2 = CRUISE_Z_7, CRUISE_Z_2
            frac  = (t - T_SQ_END) / MISSION_TIME
            angle = frac * np.pi * 4
            r7 = 0.6 + 0.3 * np.sin(frac * np.pi * 2)
            r2 = 0.5 + 0.3 * np.cos(frac * np.pi * 2 + 1.2)
            cx, cy = 1.5, 2.5
            ex7 = clamp(cx + r7*np.cos(angle),        CLAMP_MARGIN, SAFE_X_MAX-CLAMP_MARGIN)
            ey7 = clamp(cy + r7*np.sin(angle),        SAFE_Y_MIN+CLAMP_MARGIN, SAFE_Y_MAX-CLAMP_MARGIN)
            ex2 = clamp(cx + r2*np.cos(angle+np.pi),  CLAMP_MARGIN, SAFE_X_MAX-CLAMP_MARGIN)
            ey2 = clamp(cy + r2*np.sin(angle+np.pi),  SAFE_Y_MIN+CLAMP_MARGIN, SAFE_Y_MAX-CLAMP_MARGIN)
            last_bx7, last_by7 = ex7, ey7
            last_bx2, last_by2 = ex2, ey2

        elif ph == 'landnav':
            ez7, ez2 = CRUISE_Z_7, CRUISE_Z_2
            ex7 = lerp(t, T_BOIDS_END, T_LANDNAV_END, last_bx7, LAND_7[0])
            ey7 = lerp(t, T_BOIDS_END, T_LANDNAV_END, last_by7, LAND_7[1])
            ex2 = lerp(t, T_BOIDS_END, T_LANDNAV_END, last_bx2, LAND_2[0])
            ey2 = lerp(t, T_BOIDS_END, T_LANDNAV_END, last_by2, LAND_2[1])

        else:  # land
            ex7, ey7 = LAND_7;  ez7 = lerp(t, T_LANDNAV_END, T_TOTAL, CRUISE_Z_7, 0.0)
            ex2, ey2 = LAND_2;  ez2 = lerp(t, T_LANDNAV_END, T_TOTAL, CRUISE_Z_2, 0.0)

        # ── actual = expected + realistic sensor noise ─────────────────────
        scale = 1.8 if ph == 'boids' else 0.8

        for d, ex, ey, ez, n in [(out7, ex7, ey7, ez7, n7), (out2, ex2, ey2, ez2, n2)]:
            d['t'].append(float(t))
            d['ex'].append(ex);  d['ey'].append(ey);  d['ez'].append(ez)
            d['ax'].append(ex + n0.04scale)
            d['ay'].append(ey + n0.03scale)
            d['az'].append(ez + n*0.025)

    def to_np(d): return {k: np.array(v) for k, v in d.items()}
    return to_np(out7), to_np(out2)

# =============================================================================
#  SECTION 8 — PLOT HELPERS
# =============================================================================

matplotlib.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'axes.grid': True,
    'grid.alpha': 0.22,
    'grid.linestyle': '--',
    'figure.dpi': 120,
    'lines.linewidth': 1.5,
    'savefig.bbox': 'tight',
    'savefig.dpi': 140,
})


def _rms(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - expected) ** 2)))


def _rolling_rms_3d(d, e, win):
    err2 = (d['ax']-e['ax'])**2 + (d['ay']-e['ay'])**2 + (d['az']-e['az'])**2
    kernel = np.ones(win) / win
    return np.sqrt(np.convolve(err2, kernel, mode='same'))


def _add_safezone(ax):
    w = SAFE_X_MAX - SAFE_X_MIN
    h = SAFE_Y_MAX - SAFE_Y_MIN
    rect = mpatches.FancyBboxPatch(
        (SAFE_X_MIN, SAFE_Y_MIN), w, h,
        boxstyle='square,pad=0', linewidth=1.0,
        edgecolor='#444', facecolor='#eef3f8', zorder=0, alpha=0.45)
    ax.add_patch(rect)


def _draw_square_path(ax):
    """Draw the planned square as a dashed grey box on an XY axes."""
    xs = [c[0] for c in SQUARE_CORNERS] + [SQ_C0[0]]
    ys = [c[1] for c in SQUARE_CORNERS] + [SQ_C0[1]]
    ax.plot(xs, ys, ':', color=CG, lw=1.0, alpha=0.6, zorder=1, label='Square path (planned)')


def _phase_vlines(ax, ymax):
    """Annotate mission phase boundaries as vertical dotted lines."""
    phases = [
        (T_TAKEOFF_END, 'Takeoff'),
        (T_ENTRY_END,   'Entry'),
        (T_ENTRY_END + SIDE_DUR,     'Sq C1'),
        (T_ENTRY_END + 2*SIDE_DUR,   'Sq C2'),
        (T_ENTRY_END + 3*SIDE_DUR,   'Sq C3'),
        (T_SQ_END,      'Boids'),
        (T_BOIDS_END,   'Land nav'),
    ]
    for tv, name in phases:
        ax.axvline(tv, color=CG, ls=':', lw=0.6, alpha=0.45)
        ax.text(tv + 0.3, ymax * 0.93, name, fontsize=6.5, color=CG, rotation=0)

# =============================================================================
#  SECTION 9 — INDIVIDUAL PLOT FUNCTIONS
# =============================================================================

def plot_xy(d7, d2, out_dir):
    """Figure 1: XY plane — actual vs expected (top-down view) with square overlay."""
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.suptitle('Figure 1 — XY Plane: Actual vs Expected Trajectory (Square Path)', fontweight='bold')
    _add_safezone(ax)
    _draw_square_path(ax)

    every = 5
    for d, color, label in [(d7, C7, 'Drone 7'), (d2, C2, 'Drone 2')]:
        ax.plot(d['ex'][::every], d['ey'][::every], '--', color=color, lw=0.9, alpha=0.5)
        ax.plot(d['ax'][::every], d['ay'][::every], '-',  color=color, lw=1.4,
                label=f'{label} actual')

    # Square corners
    for i, (cx, cy) in enumerate(SQUARE_CORNERS):
        ax.scatter(cx, cy, marker='s', s=80, color=CA, zorder=5)
        ax.annotate(f'C{i}', (cx, cy), textcoords='offset points', xytext=(5, 5), fontsize=8)

    # Landing pads
    ax.scatter(LAND_7, marker='', s=130, color=C7, zorder=5, label='Land pad 7')
    ax.scatter(LAND_2, marker='', s=130, color=C2, zorder=5, label='Land pad 2')
    ax.annotate('L7', LAND_7, textcoords='offset points', xytext=(5, 5), fontsize=8)
    ax.annotate('L2', LAND_2, textcoords='offset points', xytext=(5, 5), fontsize=8)

    legend_handles = [
        Line2D([0],[0], color=C7, lw=1.4,  label='Drone 7 actual'),
        Line2D([0],[0], color=C7, lw=0.9, ls='--', alpha=0.6, label='Drone 7 expected'),
        Line2D([0],[0], color=C2, lw=1.4,  label='Drone 2 actual'),
        Line2D([0],[0], color=C2, lw=0.9, ls='--', alpha=0.6, label='Drone 2 expected'),
        Line2D([0],[0], color=CG, lw=1.0, ls=':', alpha=0.7, label='Square path (planned)'),
        Line2D([0],[0], marker='s', color=CA, lw=0, markersize=7, label='Square corners C0–C3'),
        Line2D([0],[0], marker='*', color=C7, lw=0, markersize=9, label='Land pad 7'),
        Line2D([0],[0], marker='*', color=C2, lw=0, markersize=9, label='Land pad 2'),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc='upper left', framealpha=0.85)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_xlim(-0.15, 3.15); ax.set_ylim(0.75, 4.25)
    ax.set_aspect('equal')
    ax.set_title('Solid = LPS actual  |  Dashed = planned  |  □ = square corners', fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, '01_xy_trajectory.png')
    fig.savefig(path); plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_axes(d7, d2, out_dir):
    """Figure 2: X, Y, Z position vs time with square-phase markers."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle('Figure 2 — Per-Axis Position over Time: Actual vs Expected', fontweight='bold')
    every = 5
    T = d7['t'][::every]

    configs = [
        ('ax','ex', 'X position (m)', (SAFE_X_MIN-0.1, SAFE_X_MAX+0.1)),
        ('ay','ey', 'Y position (m)', (SAFE_Y_MIN-0.1, SAFE_Y_MAX+0.1)),
        ('az','ez', 'Altitude Z (m)', (-0.05, 1.05)),
    ]
    for ax, (ak, ek, ylabel, ylim) in zip(axes, configs):
        for d, color, label in [(d7,C7,'Drone 7'), (d2,C2,'Drone 2')]:
            ax.plot(T, d[ek][::every], '--', color=color, lw=0.9, alpha=0.5)
            ax.plot(T, d[ak][::every], '-',  color=color, lw=1.4, label=f'{label} actual')

        if ak == 'az':
            ax.axhline(CRUISE_Z_7, color=C7, ls=':', lw=0.8, alpha=0.6,
                       label=f'Cruise Z7={CRUISE_Z_7}m')
            ax.axhline(CRUISE_Z_2, color=C2, ls=':', lw=0.8, alpha=0.6,
                       label=f'Cruise Z2={CRUISE_Z_2}m')
            ax.fill_betweenx([CRUISE_Z_7, CRUISE_Z_2], T[0], T[-1],
                             alpha=0.05, color=CG, label='0.3 m Z separation')

        _phase_vlines(ax, ylim[1])
        ax.set_ylabel(ylabel); ax.set_ylim(*ylim)
        ax.legend(fontsize=7, ncol=2, loc='upper right', framealpha=0.8)

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    path = os.path.join(out_dir, '02_xyz_axes.png')
    fig.savefig(path); plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_rms(d7, d2, out_dir):
    """Figure 3: RMS error — per-axis bars + rolling 3D curve."""
    rms7 = {a: _rms(d7[f'a{a}'], d7[f'e{a}']) for a in ('x','y','z')}
    rms2 = {a: _rms(d2[f'a{a}'], d2[f'e{a}']) for a in ('x','y','z')}
    rms7['3D'] = float(np.sqrt(sum(rms7[a]**2 for a in ('x','y','z'))))
    rms2['3D'] = float(np.sqrt(sum(rms2[a]**2 for a in ('x','y','z'))))

    win   = 25   # ~5 s at 0.2 s/tick
    roll7 = _rolling_rms_3d(d7, d7, win)
    roll2 = _rolling_rms_3d(d2, d2, win)

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle('Figure 3 — RMS Tracking Error Analysis', fontweight='bold')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.3)

    bar_labels = ['X', 'Y', 'Z', '3D']
    key_labels = ['x', 'y', 'z', '3D']
    for idx, (drone_lbl, rms_d, color) in enumerate([('7', rms7, C7), ('2', rms2, C2)]):
        ax = fig.add_subplot(gs[0, idx])
        vals = [rms_d[k] for k in key_labels]
        bars = ax.bar(bar_labels, vals,
                      color=[color+'88']*3 + [color],
                      edgecolor=color, linewidth=0.8, width=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0003,
                    f'{val*100:.2f} cm', ha='center', va='bottom', fontsize=8)
        ax.set_title(f'Per-Axis RMS — Drone {drone_lbl}')
        ax.set_ylabel('RMS error (m)')
        ax.set_ylim(0, max(vals) * 1.35)

    ax_roll = fig.add_subplot(gs[1, :])
    every = 5
    T7 = d7['t'][::every]; T2 = d2['t'][::every]
    ax_roll.plot(T7, roll7[::every], color=C7, lw=1.4, label='Drone 7 rolling 3D RMS')
    ax_roll.plot(T2, roll2[::every], color=C2, lw=1.4, label='Drone 2 rolling 3D RMS')
    ax_roll.fill_between(T7, roll7[::every], alpha=0.12, color=C7)
    ax_roll.fill_between(T2, roll2[::every], alpha=0.12, color=C2)
    _phase_vlines(ax_roll, max(roll7.max(), roll2.max()) * 1.1 or 0.15)
    ax_roll.set_xlabel('Time (s)'); ax_roll.set_ylabel('3D RMS error (m)')
    ax_roll.set_title(
        f'Rolling 3D RMS (win={win*BOIDS_DT:.0f}s) | '
        f'Drone7={rms7["3D"]*100:.2f}cm  Drone2={rms2["3D"]*100:.2f}cm')
    ax_roll.legend(fontsize=9)

    path = os.path.join(out_dir, '03_rms_error.png')
    fig.savefig(path); plt.close(fig)
    print(f"  Saved: {path}")
    return path, rms7, rms2


def plot_separation(d7, d2, out_dir):
    """Figure 4: Inter-drone XY and 3D separation vs time."""
    # Align arrays to the shorter one in case they differ by 1 sample
    n = min(len(d7['t']), len(d2['t']))
    sep_xy = np.sqrt((d7['ax'][:n]-d2['ax'][:n])**2 + (d7['ay'][:n]-d2['ay'][:n])**2)
    sep_3d = np.sqrt((d7['ax'][:n]-d2['ax'][:n])**2 +
                     (d7['ay'][:n]-d2['ay'][:n])**2 +
                     (d7['az'][:n]-d2['az'][:n])**2)
    T = d7['t'][:n]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle('Figure 4 — Inter-Drone Separation Distance', fontweight='bold')
    every = 5

    ax = axes[0]
    ax.plot(T[::every], sep_xy[::every], color=C7, lw=1.4, label='XY horizontal distance')
    ax.axhline(SEP_DIST, color=CA, ls='--', lw=1.0,
               label=f'Boids sep. threshold ({SEP_DIST}m)')
    ax.axhline(0.4, color=C2, ls='--', lw=1.0, label='Risk zone (< 0.4m)')
    ax.fill_between(T[::every], 0, np.minimum(sep_xy[::every], 0.4),
                    alpha=0.18, color=C2)
    ax.fill_between(T[::every], sep_xy[::every], SEP_DIST,
                    where=sep_xy[::every] < SEP_DIST, alpha=0.07, color=CA)
    _phase_vlines(ax, sep_xy.max() * 1.15)
    ax.set_ylabel('XY distance (m)')
    ax.set_ylim(0, sep_xy.max() * 1.20)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_title(
        f'Horizontal XY  |  min={sep_xy.min():.3f}m  avg={sep_xy.mean():.3f}m', fontsize=9)

    ax2 = axes[1]
    ax2.plot(T[::every], sep_3d[::every], color=CB, lw=1.4, label='3D Euclidean distance')
    ax2.axhline(0.3, color=CG, ls=':', lw=0.8, label='Z separation (0.3m)')
    _phase_vlines(ax2, sep_3d.max() * 1.15)
    ax2.set_ylabel('3D distance (m)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylim(0, sep_3d.max() * 1.20)
    ax2.legend(fontsize=8, loc='upper right')
    ax2.set_title(
        f'3D Euclidean  |  min={sep_3d.min():.3f}m  Z-plane separation maintained at 0.3m',
        fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, '04_separation.png')
    fig.savefig(path); plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_boids_forces(d7, out_dir):
    """Figure 5: Boids rule force magnitudes during the swarm phase."""
    T    = d7['t']
    mask = (T >= T_SQ_END) & (T <= T_BOIDS_END)
    px   = d7['ax'][mask]; py = d7['ay'][mask]
    t_b  = T[mask]

    # Approximate 'other' drone position (mirror around centroid)
    rng  = np.random.default_rng(42)
    ox   = np.clip(3.0 - px + rng.normal(0, 0.05, px.shape),
                   CLAMP_MARGIN, SAFE_X_MAX-CLAMP_MARGIN)
    oy   = np.clip(5.0 - py + rng.normal(0, 0.05, py.shape),
                   SAFE_Y_MIN+CLAMP_MARGIN, SAFE_Y_MAX-CLAMP_MARGIN)

    dist    = np.maximum(np.sqrt((px-ox)**2 + (py-oy)**2), 1e-4)
    sep_mag = np.where(dist < SEP_DIST, SEP_WEIGHT * (SEP_DIST - dist), 0.0)
    coh_mag = COH_WEIGHT * np.sqrt(((px+ox)/2 - px)**2 + ((py+oy)/2 - py)**2)

    wf  = np.zeros(len(px))
    wf += np.where(px < SAFE_X_MIN+WALL_MARGIN, WALL_WEIGHT*(SAFE_X_MIN+WALL_MARGIN-px), 0)
    wf += np.where(px > SAFE_X_MAX-WALL_MARGIN, WALL_WEIGHT*(px-(SAFE_X_MAX-WALL_MARGIN)), 0)
    wf += np.where(py < SAFE_Y_MIN+WALL_MARGIN, WALL_WEIGHT*(SAFE_Y_MIN+WALL_MARGIN-py), 0)
    wf += np.where(py > SAFE_Y_MAX-WALL_MARGIN, WALL_WEIGHT*(py-(SAFE_Y_MAX-WALL_MARGIN)), 0)
    net = np.minimum(np.sqrt((sep_mag+coh_mag)**2 + wf**2), MAX_STEP)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(
        f'Figure 5 — Boids Force Magnitudes (Drone 7, Swarm Phase '
        f'{T_SQ_END:.0f}–{T_BOIDS_END:.0f} s)', fontweight='bold')

    every = 3
    ax = axes[0]
    ax.plot(t_b[::every], sep_mag[::every], color=C2, lw=1.2,
            label=f'Separation  (w={SEP_WEIGHT})')
    ax.plot(t_b[::every], coh_mag[::every], color=CA, lw=1.2,
            label=f'Cohesion    (w={COH_WEIGHT})')
    ax.plot(t_b[::every], wf[::every],      color=CW, lw=1.2,
            label=f'Wall repulsion (w={WALL_WEIGHT})')
    ax.set_ylabel('Force magnitude (m)')
    ax.legend(fontsize=9)
    ax.set_title('Individual Boids rule forces')

    ax2 = axes[1]
    ax2.plot(t_b[::every], net[::every], color=C7, lw=1.6, label='Net force (capped)')
    ax2.fill_between(t_b[::every], 0, net[::every], alpha=0.15, color=C7)
    ax2.axhline(MAX_STEP, color=CG, ls='--', lw=0.9,
                label=f'Max step cap ({MAX_STEP}m/tick)')
    ax2.set_ylabel('Net force (m/tick)')
    ax2.set_xlabel('Time (s)')
    ax2.legend(fontsize=9)
    ax2.set_title('Net combined force — applied displacement per Boids tick')

    plt.tight_layout()
    path = os.path.join(out_dir, '05_boids_forces.png')
    fig.savefig(path); plt.close(fig)
    print(f"  Saved: {path}")
    return path


def generate_all_plots(d7, d2, out_dir='.'):
    """Run all five plot functions and print the RMS summary table."""
    os.makedirs(out_dir, exist_ok=True)
    print('\n' + '='*60)
    print('  Post-Flight Analysis — Generating Plots')
    print('='*60)

    plot_xy(d7, d2, out_dir)
    plot_axes(d7, d2, out_dir)
    _, rms7, rms2 = plot_rms(d7, d2, out_dir)
    plot_separation(d7, d2, out_dir)
    plot_boids_forces(d7, out_dir)

    print()
    print('  ─── RMS Summary ────────────────────────────────────────')
    for lbl, rd in [('Drone 7', rms7), ('Drone 2', rms2)]:
        print(f'  {lbl}  X:{rd["x"]*100:.2f}cm  Y:{rd["y"]*100:.2f}cm  '
              f'Z:{rd["z"]*100:.2f}cm  →  3D total: {rd["3D"]*100:.2f}cm')
    print('  ────────────────────────────────────────────────────────')
    print(f'\n  ✓ All plots saved to "{os.path.abspath(out_dir)}"')
    print()

# =============================================================================
#  SECTION 10 — REAL FLIGHT MAIN
# =============================================================================

def run_real_flight(out_dir):
    if not CF_AVAILABLE:
        raise RuntimeError(
            "cflib not found. Install it with:  pip install cflib\n"
            "Or run in simulation mode:          python real_boids_swarm_square.py --sim")

    print('='*60)
    print('  Real-Hardware Swarm: Square Path + Boids + Landing Pads')
    print('='*60)
    print(f'  Drone 7  z={CRUISE_Z_7}m  land={LAND_7}')
    print(f'  Drone 2  z={CRUISE_Z_2}m  land={LAND_2}')
    print(f'  Square corners:  C0={SQ_C0}  C1={SQ_C1}  C2={SQ_C2}  C3={SQ_C3}')
    print(f'  Safe zone  X[{SAFE_X_MIN}, {SAFE_X_MAX}]  Y[{SAFE_Y_MIN}, {SAFE_Y_MAX}]')
    print('='*60)
    input('\nPress ENTER to verify anchors, spawn positions, and start: ')

    cflib.crtp.init_drivers()

    stop_evt = threading.Event()
    go_evt   = threading.Event()
    ready_7  = threading.Event()
    ready_2  = threading.Event()

    t7 = threading.Thread(
        target=_drone_thread,
        args=(URI_7, URI_2, ready_7, go_evt, stop_evt), daemon=True)
    t2 = threading.Thread(
        target=_drone_thread,
        args=(URI_2, URI_7, ready_2, go_evt, stop_evt), daemon=True)
    t7.start(); t2.start()

    ready_7.wait(timeout=90)
    ready_2.wait(timeout=90)

    if not ready_7.is_set() or not ready_2.is_set() or stop_evt.is_set():
        print('[Main] Pre-flight timeout or error. Terminating.')
        stop_evt.set()
        return

    print('[Main] Both drones ready. Synchronized departure in 3 s ...')
    time.sleep(3.0)
    go_evt.set()

    try:
        t7.join(); t2.join()
    except KeyboardInterrupt:
        print('\n[Main] Abort — forcing stop ...')
        stop_evt.set()
        t7.join(timeout=5); t2.join(timeout=5)

    print('\n[Main] Mission complete.')

    d7 = _telem_arrays(URI_7)
    d2 = _telem_arrays(URI_2)
    generate_all_plots(d7, d2, out_dir)

# =============================================================================
#  SECTION 11 — ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            'Crazyflie Boids swarm — square path pre-flight + Boids loop + landing.\n'
            'Use --sim for a full simulation run without any hardware.'),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sim',   action='store_true',
                        help='Simulate mission (no Crazyflie hardware required)')
    parser.add_argument('--out',   metavar='DIR', default='./plots',
                        help='Output directory for PNG plots (default: ./plots)')
    parser.add_argument('--show',  action='store_true',
                        help='Open plot windows interactively (needs a display)')
    parser.add_argument('--seed7', type=int, default=42,
                        help='RNG seed for Drone 7 position noise (default: 42)')
    parser.add_argument('--seed2', type=int, default=77,
                        help='RNG seed for Drone 2 position noise (default: 77)')
    args = parser.parse_args()

    if args.show:
        matplotlib.use('TkAgg')

    if args.sim:
        print('='*60)
        print('  Simulation mode — no hardware required')
        print(f'  Square: C0{SQ_C0} → C1{SQ_C1} → C2{SQ_C2} → C3{SQ_C3} → C0')
        print(f'  Each side: {SIDE_DUR}s  |  Total mission: ~{T_TOTAL:.0f}s')
        print('='*60)
        d7, d2 = generate_synthetic_mission(seed7=args.seed7, seed2=args.seed2)
        t_end = d7['t'][-1]
        print(f'  Generated {len(d7["t"])} samples  ({t_end:.1f}s)')
        generate_all_plots(d7, d2, args.out)
        if args.show:
            plt.show()
    else:
        run_real_flight(args.out)


if __name__ == '__main__':
    main()
