#!/usr/bin/env python3
"""
Multi-drone circle formation flight — generalized to N drones.
Default configured for N_DRONES = 3, phase offset = 2*pi / N_DRONES (120 deg).

To scale further (e.g. 4 or 5 drones), just extend the URIS / LAND_PADS /
COLORS lists below — everything else (threads, plots, metrics) already
loops generically over N_DRONES.
"""

import csv
import math
import time
import threading
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless-safe; pass --show for interactive windows
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from itertools import combinations

try:
    import cflib.crtp
    from cflib.crazyflie import Crazyflie
    from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
    from cflib.crazyflie.log import LogConfig
    CF_AVAILABLE = True
except ImportError:
    CF_AVAILABLE = False

# =============================================================================
# SECTION 1 — CONSTANTS
# =============================================================================

# Radio URIs — add/remove entries here to change the number of drones
URIS = [
    'radio://0/80/2M/E7E7E7E702',
    'radio://0/80/2M/E7E7E7E703',
    'radio://0/80/2M/E7E7E7E704',
]
N_DRONES = len(URIS)
LABELS = [u[-1] for u in URIS]  # '1','2','3' — used in prints/filenames

# Safe zone
SAFE_X_MIN, SAFE_X_MAX = 0.0, 3.0
SAFE_Y_MIN, SAFE_Y_MAX = 1.0, 4.0

# Flight — all drones fly at the SAME altitude
FLY_Z = 1.2       # m cruise altitude
TAKEOFF_DUR = 2.5  # s

# LPS
LPS_SETTLE = 10.0  # s – wait for Kalman filter after connect

# Circle
CIRCLE_CX = (SAFE_X_MIN + SAFE_X_MAX) / 2.0  # 1.5 m
CIRCLE_CY = (SAFE_Y_MIN + SAFE_Y_MAX) / 2.0  # 2.5 m
CIRCLE_R = 1.2      # m (fits inside safe zone with >=0.15 m wall clearance)
OMEGA = 0.7         # rad/s -> one full lap ~= 8.98 s
CIRCLE_DT = 0.1      # s per waypoint (10 Hz)
CIRCLE_LAPS = 3
CIRCLE_DURATION = CIRCLE_LAPS * (2 * math.pi / OMEGA)

# Phase offsets — evenly spaced around the circle: 0, 2pi/N, 4pi/N, ...
PHASE_STEP = 2 * math.pi / N_DRONES
START_ANGLES = [i * PHASE_STEP for i in range(N_DRONES)]

# Expected chord distance between any two *adjacent* drones on the circle
# (all adjacent pairs are equal for a regularly-spaced formation)
ADJACENT_CHORD = 2 * CIRCLE_R * math.sin(PHASE_STEP / 2)

# Entry points on the circle, one per drone
CIRCLE_STARTS = [
    (CIRCLE_CX + CIRCLE_R * math.cos(a), CIRCLE_CY + CIRCLE_R * math.sin(a))
    for a in START_ANGLES
]

# Landing pads — one per drone, spread inside the safe zone
LAND_PADS = [
    (2.2, 2.0),
    (2.0, 3.0),
    (0.8, 3.2),
]

# Radio stagger — offset each drone's go_to calls slightly so packets
# never hit the shared 80/2M channel at exactly the same moment.
RADIO_STAGGER = 0.05  # s between each drone's tick within a cycle

# Mission timeline
T_TAKEOFF_END = 3.5
T_ENTRY_END = 8.0
T_CIRCLE_END = T_ENTRY_END + CIRCLE_DURATION
T_LANDNAV_END = T_CIRCLE_END + 4.5
T_TOTAL = T_LANDNAV_END + 4.5

# Plot palette — one color per drone, extend if you add more drones
COLORS = ['#3266ad', '#c94a2a', '#3a8a5a', '#c4870a', '#8a3ab5']
CG = '#888780'  # neutral gray for overlays

def color_for(i):
    return COLORS[i % len(COLORS)]

# =============================================================================
# SECTION 2 — SHARED STATE & TELEMETRY
# =============================================================================

_state_lock = threading.Lock()
_drone_pos = {uri: np.array([1.0 + i, 2.0 + i * 0.5, 0.0]) for i, uri in enumerate(URIS)}

_telem_lock = threading.Lock()
_telem = {uri: {'t': [], 'ax': [], 'ay': [], 'az': [], 'ex': [], 'ey': [], 'ez': []} for uri in URIS}


def _log_telem(uri, t, actual, expected):
    with _telem_lock:
        d = _telem[uri]
        d['t'].append(float(t))
        d['ax'].append(float(actual[0])); d['ay'].append(float(actual[1])); d['az'].append(float(actual[2]))
        d['ex'].append(float(expected[0])); d['ey'].append(float(expected[1])); d['ez'].append(float(expected[2]))


def _telem_arrays(uri):
    with _telem_lock:
        return {k: np.array(v) for k, v in _telem[uri].items()}


# =============================================================================
# SECTION 3 — LPS HELPERS
# =============================================================================

def _read_lps_position(cf) -> np.ndarray:
    """Wait for Kalman filter to settle, then read absolute LPS position."""
    print(f"    [LPS] Waiting {LPS_SETTLE} s for estimator to settle ...")
    time.sleep(LPS_SETTLE)

    pos = {}; done = [False]

    def _cb(ts, data, lc):
        if not done[0]:
            pos['x'] = data.get('kalman.stateX', 0.0)
            pos['y'] = data.get('kalman.stateY', 0.0)
            pos['z'] = data.get('kalman.stateZ', 0.0)
            done[0] = True

    lc = LogConfig(name='SpawnPos', period_in_ms=100)
    for v in ('kalman.stateX', 'kalman.stateY', 'kalman.stateZ'):
        lc.add_variable(v, 'float')
    cf.log.add_config(lc)
    lc.data_received_cb.add_callback(_cb)
    lc.start()

    t0 = time.time()
    while not done[0] and (time.time() - t0) < 6.0:
        time.sleep(0.05)
    lc.stop()

    if not done[0]:
        raise RuntimeError("LPS position read timed out. Check deck and anchors.")

    return np.array([pos['x'], pos['y'], pos['z']])


def _start_state_logging(cf, uri: str) -> 'LogConfig':
    """Continuously update _drone_pos[uri] from Kalman state."""
    def _cb(ts, data, lc):
        with _state_lock:
            _drone_pos[uri] = np.array([
                data.get('kalman.stateX', _drone_pos[uri][0]),
                data.get('kalman.stateY', _drone_pos[uri][1]),
                data.get('kalman.stateZ', _drone_pos[uri][2]),
            ])

    lc = LogConfig(name='StateLog', period_in_ms=100)
    for v in ('kalman.stateX', 'kalman.stateY', 'kalman.stateZ'):
        lc.add_variable(v, 'float')
    cf.log.add_config(lc)
    lc.data_received_cb.add_callback(_cb)
    lc.start()
    return lc


def _start_telem_logging(cf, uri: str) -> 'LogConfig':
    """Log kalman state + ctrltarget setpoint into _telem[uri]."""
    def _cb(ts, data, lc):
        with _state_lock:
            actual = _drone_pos[uri].copy()
        expected = [
            data.get('ctrltarget.x', actual[0]),
            data.get('ctrltarget.y', actual[1]),
            data.get('ctrltarget.z', actual[2]),
        ]
        _log_telem(uri, time.time(), actual, expected)

    lc = LogConfig(name='TelemLog', period_in_ms=100)
    for v in ('ctrltarget.x', 'ctrltarget.y', 'ctrltarget.z'):
        lc.add_variable(v, 'float')
    cf.log.add_config(lc)
    lc.data_received_cb.add_callback(_cb)
    return lc  # caller does lc.start()


# =============================================================================
# SECTION 4 — CSV SAVE
# =============================================================================

def save_csv(uri, out_dir):
    d = _telem_arrays(uri)
    if len(d['t']) == 0:
        return
    d['t'] = d['t'] - d['t'][0]
    label = uri[-1]
    path = os.path.join(out_dir, f'drone_{label}_telem.csv')
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['t', 'ax', 'ay', 'az', 'ex', 'ey', 'ez'])
        w.writeheader()
        for i in range(len(d['t'])):
            w.writerow({k: float(d[k][i]) for k in d})
    print(f"    [CSV] Saved {path}")


# =============================================================================
# SECTION 5 — PER-DRONE FLIGHT THREAD
# =============================================================================

def _drone_thread(index, uri, ready_evt, go_evt, stop_evt):
    label = uri[-1]
    land_x, land_y = LAND_PADS[index]
    start_angle = START_ANGLES[index]
    start_x, start_y = CIRCLE_STARTS[index]
    stagger = index * RADIO_STAGGER  # each drone's tick offset within a cycle

    print(f"\n[Drone {label}] Connecting ...")
    try:
        with SyncCrazyflie(uri, cf=Crazyflie(rw_cache=f'./cache_{label}')) as scf:
            cf = scf.cf
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
            print(f"[Drone {label}] Spawn: ({spawn[0]:.2f}, {spawn[1]:.2f}, {spawn[2]:.2f})")

            margin = 0.1
            if not (SAFE_X_MIN + margin <= spawn[0] <= SAFE_X_MAX - margin and
                    SAFE_Y_MIN + margin <= spawn[1] <= SAFE_Y_MAX - margin):
                raise RuntimeError(
                    f"Drone {label} spawn ({spawn[0]:.2f},{spawn[1]:.2f}) "
                    f"is outside the safe zone. Aborting.")

            state_log = _start_state_logging(cf, uri)
            telem_log = _start_telem_logging(cf, uri)

            print(f"[Drone {label}] Ready — waiting for formation ...")
            ready_evt.set()
            go_evt.wait()
            telem_log.start()

            # Step 1: Takeoff
            print(f"[Drone {label}] TAKEOFF -> z={FLY_Z} m")
            hlc.takeoff(FLY_Z, TAKEOFF_DUR)
            time.sleep(TAKEOFF_DUR + 1.0)

            # Step 2: Fly to circle entry point
            dist = math.sqrt((start_x - spawn[0]) ** 2 + (start_y - spawn[1]) ** 2)
            entry_dur = max(3.0, dist / 0.5)
            print(f"[Drone {label}] ENTRY -> ({start_x:.2f},{start_y:.2f}) "
                  f"angle={math.degrees(start_angle):.0f} deg dur={entry_dur:.1f}s")
            hlc.go_to(start_x, start_y, FLY_Z, yaw=0.0, duration_s=entry_dur, relative=False)
            time.sleep(entry_dur + 0.5)

            # Step 3: Dense CCW circle loop, staggered radio ticks
            time.sleep(stagger)
            steps = int(CIRCLE_DURATION / CIRCLE_DT)
            theta = start_angle
            print(f"[Drone {label}] CIRCLE r={CIRCLE_R}m w={OMEGA}rad/s "
                  f"laps={CIRCLE_LAPS} steps={steps}")

            for step in range(steps):
                theta += OMEGA * CIRCLE_DT
                wx = CIRCLE_CX + CIRCLE_R * math.cos(theta)
                wy = CIRCLE_CY + CIRCLE_R * math.sin(theta)
                wx = max(SAFE_X_MIN + 0.05, min(SAFE_X_MAX - 0.05, wx))
                wy = max(SAFE_Y_MIN + 0.05, min(SAFE_Y_MAX - 0.05, wy))
                hlc.go_to(wx, wy, FLY_Z, yaw=0.0, duration_s=CIRCLE_DT, relative=False)
                time.sleep(CIRCLE_DT)
                if step % 90 == 0:
                    deg = math.degrees(theta) % 360
                    print(f"[Drone {label}] CIRCLE step {step+1:4d}/{steps} "
                          f"({wx:.2f},{wy:.2f}) {deg:.0f} deg")

            # Step 4: Fly to landing pad
            print(f"[Drone {label}] LAND-NAV -> ({land_x:.2f},{land_y:.2f})")
            hlc.go_to(land_x, land_y, FLY_Z, yaw=0.0, duration_s=4.0, relative=False)
            time.sleep(4.5)

            # Step 5: Land
            print(f"[Drone {label}] LANDING")
            hlc.land(0.0, duration_s=4.0)
            time.sleep(4.5)
            hlc.stop()

            telem_log.stop()
            state_log.stop()
            print(f"[Drone {label}] Landed safely.")

    except Exception as exc:
        print(f"[Drone {label}] ERROR: {exc}")
        stop_evt.set()
        ready_evt.set()
        raise


# =============================================================================
# SECTION 6 — SYNTHETIC DATA GENERATOR (--sim mode)
# =============================================================================

def _lcg(seed):
    s = int(seed) & 0xFFFFFFFF
    while True:
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        yield (s >> 1) / 0x7FFFFFFF - 1.0


def generate_synthetic_mission(seeds=None):
    """
    Simulate the full mission for N_DRONES, each 2*pi/N apart on the circle.
    Returns a list of dicts (one per drone), each with numpy arrays
    (t, ax, ay, az, ex, ey, ez).
    """
    if seeds is None:
        seeds = [42 + 17 * i for i in range(N_DRONES)]
    rngs = [_lcg(s) for s in seeds]

    T = np.arange(0, T_TOTAL + CIRCLE_DT, CIRCLE_DT)

    def clamp(v, lo, hi): return float(np.clip(v, lo, hi))

    def ease(f):
        f = clamp(f, 0, 1)
        return 2 * f ** 2 if f < 0.5 else 1 - (-2 * f + 2) ** 2 / 2

    def lerp(t, t0, t1, v0, v1):
        return v0 + (v1 - v0) * ease((t - t0) / max(t1 - t0, 1e-9))

    def circle_xy(t, start_ang):
        frac = clamp((t - T_ENTRY_END) / CIRCLE_DURATION, 0.0, 1.0)
        theta = start_ang + frac * CIRCLE_LAPS * 2 * math.pi
        return (CIRCLE_CX + CIRCLE_R * math.cos(theta),
                CIRCLE_CY + CIRCLE_R * math.sin(theta))

    def phase(t):
        if t <= T_TAKEOFF_END: return 'takeoff'
        if t <= T_ENTRY_END: return 'entry'
        if t <= T_CIRCLE_END: return 'circle'
        if t <= T_LANDNAV_END: return 'landnav'
        return 'land'

    spawn_xy = [(1.0 + i, 2.0 + i * 0.5) for i in range(N_DRONES)]
    outs = [{k: [] for k in ('t', 'ax', 'ay', 'az', 'ex', 'ey', 'ez')} for _ in range(N_DRONES)]
    lasts = [list(CIRCLE_STARTS[i]) for i in range(N_DRONES)]

    for t in T:
        if t > T_TOTAL:
            break
        ph = phase(t)
        for i in range(N_DRONES):
            n = next(rngs[i])
            sx, sy = spawn_xy[i]

            if ph == 'takeoff':
                ex, ey = sx, sy
                ez = lerp(t, 0, T_TAKEOFF_END, 0.0, FLY_Z)
            elif ph == 'entry':
                ez = FLY_Z
                ex = lerp(t, T_TAKEOFF_END, T_ENTRY_END, sx, CIRCLE_STARTS[i][0])
                ey = lerp(t, T_TAKEOFF_END, T_ENTRY_END, sy, CIRCLE_STARTS[i][1])
            elif ph == 'circle':
                ez = FLY_Z
                ex, ey = circle_xy(t, START_ANGLES[i])
                lasts[i] = [ex, ey]
            elif ph == 'landnav':
                ez = FLY_Z
                ex = lerp(t, T_CIRCLE_END, T_LANDNAV_END, lasts[i][0], LAND_PADS[i][0])
                ey = lerp(t, T_CIRCLE_END, T_LANDNAV_END, lasts[i][1], LAND_PADS[i][1])
            else:  # land
                ex, ey = LAND_PADS[i]
                ez = lerp(t, T_LANDNAV_END, T_TOTAL, FLY_Z, 0.0)

            noise = 0.8
            d = outs[i]
            d['t'].append(float(t))
            d['ex'].append(ex); d['ey'].append(ey); d['ez'].append(ez)
            d['ax'].append(ex + n * 0.035 * noise)
            d['ay'].append(ey + n * 0.028 * noise)
            d['az'].append(ez + n * 0.018)

    return [{k: np.array(v) for k, v in d.items()} for d in outs]


# =============================================================================
# SECTION 7 — METRICS
# =============================================================================

def _mae(actual, expected):
    return float(np.mean(np.abs(actual - expected)))


def _rms(actual, expected):
    return float(np.sqrt(np.mean((actual - expected) ** 2)))


def _rolling_rms_3d(d, win):
    err2 = (d['ax'] - d['ex']) ** 2 + (d['ay'] - d['ey']) ** 2 + (d['az'] - d['ez']) ** 2
    return np.sqrt(np.convolve(err2, np.ones(win) / win, mode='same'))


def print_summary(drones):
    print("\n" + "=" * 52)
    print(" AXIS-WISE TRACKING MAE SUMMARY")
    print("=" * 52)
    for i, d in enumerate(drones):
        mx = _mae(d['ax'], d['ex'])
        my = _mae(d['ay'], d['ey'])
        mz = _mae(d['az'], d['ez'])
        print(f"  Drone {LABELS[i]}  X:{mx*100:.2f}cm  Y:{my*100:.2f}cm  Z:{mz*100:.2f}cm")
    print("=" * 52 + "\n")


# =============================================================================
# SECTION 8 — PLOT HELPERS
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


def _add_safezone(ax):
    ax.add_patch(mpatches.FancyBboxPatch(
        (SAFE_X_MIN, SAFE_Y_MIN),
        SAFE_X_MAX - SAFE_X_MIN, SAFE_Y_MAX - SAFE_Y_MIN,
        boxstyle='square,pad=0', linewidth=1.0,
        edgecolor='#444', facecolor='#eef3f8', zorder=0, alpha=0.4))
    bx = [SAFE_X_MIN, SAFE_X_MAX, SAFE_X_MAX, SAFE_X_MIN, SAFE_X_MIN]
    by = [SAFE_Y_MIN, SAFE_Y_MIN, SAFE_Y_MAX, SAFE_Y_MAX, SAFE_Y_MIN]
    ax.plot(bx, by, 'r--', lw=1.1, alpha=0.5, label='Safety boundary')


def _draw_circle_overlay(ax):
    theta = np.linspace(0, 2 * math.pi, 360)
    ax.plot(CIRCLE_CX + CIRCLE_R * np.cos(theta),
            CIRCLE_CY + CIRCLE_R * np.sin(theta),
            ':', color=CG, lw=1.4, alpha=0.7, zorder=1,
            label=f'Planned circle r={CIRCLE_R}m')
    a16 = np.linspace(0, 2 * math.pi, 17)[:-1]
    ax.scatter(CIRCLE_CX + CIRCLE_R * np.cos(a16),
               CIRCLE_CY + CIRCLE_R * np.sin(a16),
               s=18, color=CG, alpha=0.45, zorder=3)
    ax.scatter(CIRCLE_CX, CIRCLE_CY, marker='+', s=90, color=CG, linewidths=1.4, zorder=4)
    ax.annotate(f'Centre\n({CIRCLE_CX},{CIRCLE_CY})',
                (CIRCLE_CX, CIRCLE_CY),
                textcoords='offset points', xytext=(6, -16), fontsize=7.5, color=CG)


def _phase_vlines(ax, ymax):
    quarter = (2 * math.pi / OMEGA) / 4
    marks = [
        (T_TAKEOFF_END, 'Takeoff'),
        (T_ENTRY_END, 'Circle start'),
        (T_ENTRY_END + quarter, '90deg'),
        (T_ENTRY_END + 2 * quarter, '180deg'),
        (T_ENTRY_END + 3 * quarter, '270deg'),
        (T_ENTRY_END + 4 * quarter, 'Lap 1'),
        (T_ENTRY_END + 8 * quarter, 'Lap 2'),
        (T_CIRCLE_END, 'Land nav'),
    ]
    for tv, name in marks:
        if tv > T_TOTAL:
            continue
        ax.axvline(tv, color=CG, ls=':', lw=0.6, alpha=0.45)
        ax.text(tv + 0.3, ymax * 0.93, name, fontsize=6.5, color=CG)


# =============================================================================
# SECTION 9 — PLOTS
# =============================================================================

def plot_xy(drones, out_dir):
    """Figure 1: XY top-down — actual vs expected with circle overlay."""
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    fig.suptitle(f'Figure 1 — XY Top-Down: Actual vs Expected ({N_DRONES} drones, '
                 f'{math.degrees(PHASE_STEP):.0f} deg phase separation)', fontweight='bold')
    _add_safezone(ax)
    _draw_circle_overlay(ax)

    every = 3
    handles = []
    for i, d in enumerate(drones):
        c = color_for(i)
        ax.plot(d['ex'][::every], d['ey'][::every], '--', color=c, lw=0.9, alpha=0.5)
        ax.plot(d['ax'][::every], d['ay'][::every], '-', color=c, lw=1.4)
        sx, sy = CIRCLE_STARTS[i]
        ax.scatter(sx, sy, marker='>', s=100, color=c, zorder=5)
        ax.annotate(f'D{LABELS[i]} entry\n({math.degrees(START_ANGLES[i]):.0f} deg)',
                    (sx, sy), textcoords='offset points', xytext=(6, 4), fontsize=7.5, color=c)
        lx, ly = LAND_PADS[i]
        ax.scatter(lx, ly, marker='*', s=140, color=c, zorder=5)
        ax.annotate(f'L{LABELS[i]}', (lx, ly), textcoords='offset points', xytext=(5, 5), fontsize=8)
        handles += [
            Line2D([0], [0], color=c, lw=1.4, label=f'Drone {LABELS[i]} actual'),
            Line2D([0], [0], color=c, lw=0.9, ls='--', alpha=0.6, label=f'Drone {LABELS[i]} expected'),
        ]

    handles += [
        Line2D([0], [0], color=CG, lw=1.4, ls=':', alpha=0.7, label=f'Planned circle r={CIRCLE_R}m'),
        Line2D([0], [0], color='r', lw=1.1, ls='--', alpha=0.5, label='Safety boundary'),
    ]
    ax.legend(handles=handles, fontsize=7, loc='upper left', framealpha=0.85, ncol=1)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_xlim(-0.15, 3.15); ax.set_ylim(0.75, 4.25)
    ax.set_aspect('equal')
    ax.set_title(
        f'All {N_DRONES} drones at z={FLY_Z}m | Phase step={math.degrees(PHASE_STEP):.0f} deg | '
        f'c=({CIRCLE_CX},{CIRCLE_CY}) r={CIRCLE_R}m w={OMEGA}rad/s laps={CIRCLE_LAPS}', fontsize=8.5)
    plt.tight_layout()
    path = os.path.join(out_dir, '01_xy_trajectory.png')
    fig.savefig(path); plt.close(fig)
    print(f"    Saved: {path}")
    return path


def plot_axes(drones, out_dir):
    """Figure 2: Per-axis position timelines."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Figure 2 — Axis-wise Position Timeline: Actual vs Expected', fontweight='bold')

    every = 2
    for i, d in enumerate(drones):
        c = color_for(i)
        T = d['t'][::every]
        for k, (ak, ek) in enumerate([('ax', 'ex'), ('ay', 'ey'), ('az', 'ez')]):
            mae = _mae(d[ak], d[ek])
            axes[k].plot(T, d[ek][::every], '--', color=c, lw=1.3, alpha=0.55)
            axes[k].plot(T, d[ak][::every], '-', color=c, lw=1.2,
                         label=f'D{LABELS[i]} (MAE={mae*100:.2f}cm)')

    axes[2].axhline(FLY_Z, color=CG, ls=':', lw=1.0, alpha=0.7,
                     label=f'Cruise z={FLY_Z}m (all drones)')

    configs = [
        ('X position (m)', (SAFE_X_MIN - 0.1, SAFE_X_MAX + 0.1), 'X-Axis Tracking'),
        ('Y position (m)', (SAFE_Y_MIN - 0.1, SAFE_Y_MAX + 0.1), 'Y-Axis Tracking'),
        ('Altitude Z (m)', (-0.05, 1.0), 'Z-Axis (all at same height)'),
    ]
    for ax, (ylabel, ylim, title) in zip(axes, configs):
        _phase_vlines(ax, ylim[1])
        ax.set_ylabel(ylabel); ax.set_ylim(*ylim)
        ax.set_title(title + ' (Dashed = expected / ctrltarget)', fontsize=9)
        ax.legend(fontsize=7, ncol=N_DRONES, loc='upper right', framealpha=0.8)
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    path = os.path.join(out_dir, '02_xyz_axes.png')
    fig.savefig(path); plt.close(fig)
    print(f"    Saved: {path}")
    return path


def plot_rms(drones, out_dir):
    """Figure 3: Per-axis RMS bars + rolling 3-D RMS curve."""
    rms_list = []
    for d in drones:
        r = {a: _rms(d[f'a{a}'], d[f'e{a}']) for a in ('x', 'y', 'z')}
        r['3D'] = math.sqrt(sum(r[a] ** 2 for a in ('x', 'y', 'z')))
        rms_list.append(r)

    win = 50  # 5 s rolling window at 0.1 s/sample
    rolls = [_rolling_rms_3d(d, win) for d in drones]

    fig = plt.figure(figsize=(6 * N_DRONES, 8))
    fig.suptitle('Figure 3 — RMS Tracking Error Analysis', fontweight='bold')
    gs = gridspec.GridSpec(2, N_DRONES, figure=fig, hspace=0.38, wspace=0.35)

    for i, (rms_d, c) in enumerate(zip(rms_list, [color_for(j) for j in range(N_DRONES)])):
        ax = fig.add_subplot(gs[0, i])
        keys = ['x', 'y', 'z', '3D']
        vals = [rms_d[k] for k in keys]
        bars = ax.bar(['X', 'Y', 'Z', '3D'], vals,
                       color=[c + '88'] * 3 + [c], edgecolor=c, linewidth=0.8, width=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0002,
                    f'{val*100:.2f}cm', ha='center', va='bottom', fontsize=8)
        ax.set_title(f'Per-Axis RMS — Drone {LABELS[i]}')
        ax.set_ylabel('RMS error (m)')
        ax.set_ylim(0, max(vals) * 1.4)

    ax_r = fig.add_subplot(gs[1, :])
    every = 3
    max_roll = 0.1
    for i, roll in enumerate(rolls):
        c = color_for(i)
        ax_r.plot(drones[i]['t'][::every], roll[::every], color=c, lw=1.4,
                  label=f'Drone {LABELS[i]} rolling 3D RMS')
        ax_r.fill_between(drones[i]['t'][::every], roll[::every], alpha=0.12, color=c)
        max_roll = max(max_roll, roll.max())

    _phase_vlines(ax_r, max_roll * 1.1)
    ax_r.set_xlabel('Time (s)'); ax_r.set_ylabel('3D RMS error (m)')
    summary = ' '.join(f'D{LABELS[i]}={rms_list[i]["3D"]*100:.2f}cm' for i in range(N_DRONES))
    ax_r.set_title(f'Rolling 3D RMS (win={win*CIRCLE_DT:.0f}s) | {summary}')
    ax_r.legend(fontsize=9)

    path = os.path.join(out_dir, '03_rms_error.png')
    fig.savefig(path); plt.close(fig)
    print(f"    Saved: {path}")
    return path, rms_list


def plot_separation(drones, out_dir):
    """
    Figure 4: Pairwise inter-drone XY distance, for every pair of drones.
    Adjacent pairs on the circle (i, i+1) should track ~ADJACENT_CHORD.
    Non-adjacent pairs (only relevant for N>=4) will differ.
    """
    n = min(len(d['t']) for d in drones)
    T = drones[0]['t'][:n]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(f'Figure 4 — Pairwise Inter-Drone XY Separation '
                 f'(adjacent target ~ {ADJACENT_CHORD:.2f} m)', fontweight='bold')

    every = 3
    max_sep = 0.1
    pair_colors = ['#3266ad', '#c94a2a', '#3a8a5a', '#c4870a', '#8a3ab5', '#555']
    for idx, (i, j) in enumerate(combinations(range(N_DRONES), 2)):
        di, dj = drones[i], drones[j]
        sep = np.sqrt((di['ax'][:n] - dj['ax'][:n]) ** 2 + (di['ay'][:n] - dj['ay'][:n]) ** 2)
        pc = pair_colors[idx % len(pair_colors)]
        ax.plot(T[::every], sep[::every], color=pc, lw=1.3,
                label=f'D{LABELS[i]}<->D{LABELS[j]} (min={sep.min():.2f}m avg={sep.mean():.2f}m)')
        max_sep = max(max_sep, sep.max())

    ax.axhline(ADJACENT_CHORD, color=CG, ls='--', lw=1.2,
               label=f'Expected adjacent chord = {ADJACENT_CHORD:.2f}m')
    ax.axhline(0.5, color='#c94a2a', ls='--', lw=1.0, label='Proximity alert (< 0.5m)')
    ax.axvspan(T_ENTRY_END, T_CIRCLE_END, alpha=0.06, color='#3a8a5a', label='Circle phase')

    _phase_vlines(ax, max_sep * 1.15)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('XY distance (m)')
    ax.set_ylim(0, max_sep * 1.20)
    ax.legend(fontsize=7.5, loc='upper right', ncol=2)
    plt.tight_layout()
    path = os.path.join(out_dir, '04_separation.png')
    fig.savefig(path); plt.close(fig)
    print(f"    Saved: {path}")
    return path


def plot_3d(drones, out_dir):
    """Figure 5: 3-D trajectory — all drones at the same altitude."""
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle('Figure 5 — 3-D Trajectory: Actual vs Expected', fontweight='bold')
    ax3 = fig.add_subplot(111, projection='3d')

    every = 3
    for i, d in enumerate(drones):
        c = color_for(i)
        ax3.plot(d['ax'][::every], d['ay'][::every], d['az'][::every],
                 color=c, lw=1.3, label=f'Drone {LABELS[i]} actual')
        ax3.plot(d['ex'][::every], d['ey'][::every], d['ez'][::every],
                 '--', color=c, lw=0.8, alpha=0.5, label=f'Drone {LABELS[i]} expected')

    theta = np.linspace(0, 2 * math.pi, 300)
    cx_pts = CIRCLE_CX + CIRCLE_R * np.cos(theta)
    cy_pts = CIRCLE_CY + CIRCLE_R * np.sin(theta)
    ax3.plot(cx_pts, cy_pts, np.full_like(cx_pts, FLY_Z),
             ':', color=CG, lw=1.0, alpha=0.5, label=f'Circle at z={FLY_Z}m')

    ax3.set_xlabel('X (m)'); ax3.set_ylabel('Y (m)'); ax3.set_zlabel('Z (m)')
    ax3.set_title(f'All {N_DRONES} drones cruise at z={FLY_Z}m | '
                  f'Phase step={math.degrees(PHASE_STEP):.0f} deg', fontsize=9)
    ax3.legend(fontsize=7.5, loc='upper left')
    plt.tight_layout()
    path = os.path.join(out_dir, '05_trajectory_3d.png')
    fig.savefig(path); plt.close(fig)
    print(f"    Saved: {path}")
    return path


def generate_all_plots(drones, out_dir='.'):
    os.makedirs(out_dir, exist_ok=True)
    print('\n' + '=' * 55)
    print(f' Post-Flight Analysis — {N_DRONES} drones — Generating 5 Plots')
    print('=' * 55)
    print_summary(drones)
    plot_xy(drones, out_dir)
    plot_axes(drones, out_dir)
    _, rms_list = plot_rms(drones, out_dir)
    plot_separation(drones, out_dir)
    plot_3d(drones, out_dir)

    print()
    print(' --- RMS Summary ------------------------------------------')
    for i, rd in enumerate(rms_list):
        print(f'  Drone {LABELS[i]}  X:{rd["x"]*100:.2f}cm  Y:{rd["y"]*100:.2f}cm  '
              f'Z:{rd["z"]*100:.2f}cm  -> 3D: {rd["3D"]*100:.2f}cm')
    print(' ------------------------------------------------------------')
    print(f'\n  All plots saved to "{os.path.abspath(out_dir)}"')


# =============================================================================
# SECTION 10 — REAL FLIGHT MAIN
# =============================================================================

def run_real_flight(out_dir):
    if not CF_AVAILABLE:
        raise RuntimeError(
            "cflib not found. pip install cflib\n"
            "Or simulate: python three_cf_circle_path.py --sim")

    print('=' * 55)
    print(f' {N_DRONES}-Drone Circle — {math.degrees(PHASE_STEP):.0f} deg phase separation')
    print('=' * 55)
    print(f'  All drones at z = {FLY_Z} m (same altitude)')
    for i, uri in enumerate(URIS):
        print(f'  Drone {LABELS[i]} start={CIRCLE_STARTS[i]} land={LAND_PADS[i]} '
              f'angle={math.degrees(START_ANGLES[i]):.0f} deg')
    print(f'  Circle c=({CIRCLE_CX},{CIRCLE_CY}) r={CIRCLE_R}m '
          f'w={OMEGA}rad/s laps={CIRCLE_LAPS} dt={CIRCLE_DT}s')
    print(f'  Safe zone X[{SAFE_X_MIN},{SAFE_X_MAX}] Y[{SAFE_Y_MIN},{SAFE_Y_MAX}]')
    print(f'  Expected adjacent chord distance: {ADJACENT_CHORD:.2f} m')
    print('=' * 55)
    input('\nPress ENTER to verify anchors and start: ')

    cflib.crtp.init_drivers()

    stop_evt = threading.Event()
    go_evt = threading.Event()
    ready_evts = [threading.Event() for _ in URIS]

    threads = [
        threading.Thread(target=_drone_thread, args=(i, uri, ready_evts[i], go_evt, stop_evt), daemon=True)
        for i, uri in enumerate(URIS)
    ]
    for t in threads:
        t.start()

    for evt in ready_evts:
        evt.wait(timeout=90)

    if not all(evt.is_set() for evt in ready_evts) or stop_evt.is_set():
        print('[Main] Pre-flight timeout or error. Aborting.')
        stop_evt.set()
        return

    print('[Main] All drones ready — departing in 3 s ...')
    time.sleep(3.0)
    go_evt.set()

    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print('\n[Main] Abort ...')
        stop_evt.set()
        for t in threads:
            t.join(timeout=5)

    print('\n[Main] Mission complete.')
    os.makedirs(out_dir, exist_ok=True)
    for uri in URIS:
        save_csv(uri, out_dir)

    drones = [_telem_arrays(uri) for uri in URIS]
    generate_all_plots(drones, out_dir)


# =============================================================================
# SECTION 11 — ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=f'{N_DRONES} Crazyflies revolving on the same circle, '
                    f'{math.degrees(PHASE_STEP):.0f} deg phase separation.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sim', action='store_true', help='Simulate without hardware')
    parser.add_argument('--out', metavar='DIR', default='./plots', help='Output directory for PNGs and CSVs')
    parser.add_argument('--show', action='store_true', help='Open interactive windows')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                         help='One RNG seed per drone (defaults to 42,59,76,...)')
    args = parser.parse_args()

    if args.show:
        matplotlib.use('TkAgg')

    if args.sim:
        print('=' * 55)
        print(f' Simulation mode — {N_DRONES} drones')
        print(f'  Circle c=({CIRCLE_CX},{CIRCLE_CY}) r={CIRCLE_R}m w={OMEGA}rad/s laps={CIRCLE_LAPS}')
        for i in range(N_DRONES):
            print(f'  Drone {LABELS[i]} angle={math.degrees(START_ANGLES[i]):.0f} deg '
                  f'entry={CIRCLE_STARTS[i]}')
        print(f'  All at z={FLY_Z}m | Adjacent XY separation ~= {ADJACENT_CHORD:.2f}m')
        print(f'  Total mission: ~{T_TOTAL:.0f}s')
        print('=' * 55)
        drones = generate_synthetic_mission(seeds=args.seeds)
        print(f'  Generated {len(drones[0]["t"])} samples ({drones[0]["t"][-1]:.1f}s) per drone')
        generate_all_plots(drones, args.out)
        if args.show:
            plt.show()
    else:
        run_real_flight(args.out)


if __name__ == '__main__':
    main()
