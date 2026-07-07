#!/usr/bin/env python3
import csv
import math
import time
import threading
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')   # headless-safe; pass --show for interactive windows
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    import cflib.crtp
    from cflib.crazyflie import Crazyflie
    from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
    from cflib.crazyflie.log import LogConfig
    CF_AVAILABLE = True
except ImportError:
    CF_AVAILABLE = False

# =============================================================================
#  SECTION 1 — CONSTANTS
# =============================================================================

URI_7 = 'radio://0/80/2M/E7E7E7E701'
URI_2 = 'radio://0/80/2M/E7E7E7E702'

# Safe zone
SAFE_X_MIN, SAFE_X_MAX = 0.0, 3.0
SAFE_Y_MIN, SAFE_Y_MAX = 1.0, 4.0

# Flight — nominal cruise altitude (both drones oscillate around this)
FLY_Z       = 1.0   # m  reference/mean altitude
TAKEOFF_DUR = 2.5    # s

# LPS
LPS_SETTLE  = 10.0    # s – wait for Kalman filter after connect

# ── Tilted ellipse path ─────────────────────────────────────────────────────
# The XY footprint is an ellipse. Altitude is NOT constant anymore: it swings
# sinusoidally with the path angle theta, so the loop looks like a circular/
# elliptical hoop that's been tilted on an axis — high on one side, low on
# the opposite side. Because the two drones stay pi apart in theta, one is
# always near the high point while the other is near the low point.
CIRCLE_CX   = (SAFE_X_MIN + SAFE_X_MAX) / 2.0   # 1.5 m
CIRCLE_CY   = (SAFE_Y_MIN + SAFE_Y_MAX) / 2.0   # 2.5 m
ELLIPSE_A   = 1.2    # m  semi-axis along X (fits inside safe zone w/ margin)
ELLIPSE_B   = 0.9    # m  semi-axis along Y (fits inside safe zone w/ margin)
TILT_AMP    = 0.35   # m  altitude swing amplitude: z = FLY_Z ± TILT_AMP
TILT_PHASE  = 0.0    # rad  rotates *where* the high/low points sit on the loop
Z_MIN, Z_MAX = 0.35, 1.65   # hard altitude safety clamp (independent of FLY_Z)

OMEGA       = 0.7    # rad/s  →  one full lap ≈ 8.98 s
CIRCLE_DT   = 0.1    # s per waypoint (10 Hz, matches reference script)
CIRCLE_LAPS = 3      # number of complete revolutions

CIRCLE_DURATION = CIRCLE_LAPS * (2 * math.pi / OMEGA)   # total circle time (s)


def _ellipse_point(theta: float):
    """
    Position on the tilted ellipse at path angle theta.
    theta=0            -> +X side, HIGH point  (z = FLY_Z + TILT_AMP)
    theta=pi           -> -X side, LOW point   (z = FLY_Z - TILT_AMP)
    (exact high/low locations rotate with TILT_PHASE)
    """
    x = CIRCLE_CX + ELLIPSE_A * math.cos(theta)
    y = CIRCLE_CY + ELLIPSE_B * math.sin(theta)
    z = FLY_Z + TILT_AMP * math.cos(theta + TILT_PHASE)
    z = max(Z_MIN, min(Z_MAX, z))
    return x, y, z


# Drone entry points (180° apart on the ellipse)
CIRCLE_START_7 = _ellipse_point(0.0)         # angle   0°
CIRCLE_START_2 = _ellipse_point(math.pi)     # angle 180°

# Landing pads
LAND_7 = (2.2, 2.0)
LAND_2 = (2.0, 3.0)

# Radio stagger – offset Drone 2's go_to calls slightly so two packets
# never hit the shared 80/2M channel at exactly the same moment.
RADIO_STAGGER = 0.05   # s

# ── Mission timeline ─────────────────────────────────────────────────────────
T_TAKEOFF_END  = 3.5
T_ENTRY_END    = 8.0                               # fly to ellipse entry point
T_CIRCLE_END   = T_ENTRY_END + CIRCLE_DURATION     # ≈ 34.9 s  (3 laps)
T_LANDNAV_END  = T_CIRCLE_END + 4.5
T_TOTAL        = T_LANDNAV_END + 4.5               # ≈ 43.9 s

# Plot palette
C7  = '#3266ad'   # Drone 7 — blue
C2  = '#c94a2a'   # Drone 2 — coral
CG  = '#888780'   # neutral
CA  = '#3a8a5a'   # green
CB  = '#c4870a'   # amber

# =============================================================================
#  SECTION 2 — SHARED STATE & TELEMETRY
# =============================================================================

_state_lock = threading.Lock()
_drone_pos  = {
    URI_7: np.array([1.0, 2.0, 0.0]),
    URI_2: np.array([2.0, 3.0, 0.0]),
}

_telem_lock = threading.Lock()
_telem = {
    URI_7: {'t': [], 'ax': [], 'ay': [], 'az': [], 'ex': [], 'ey': [], 'ez': []},
    URI_2: {'t': [], 'ax': [], 'ay': [], 'az': [], 'ex': [], 'ey': [], 'ez': []},
}


def _log_telem(uri, t, actual, expected):
    with _telem_lock:
        d = _telem[uri]
        d['t'].append(float(t))
        d['ax'].append(float(actual[0]));   d['ay'].append(float(actual[1]));   d['az'].append(float(actual[2]))
        d['ex'].append(float(expected[0])); d['ey'].append(float(expected[1])); d['ez'].append(float(expected[2]))


def _telem_arrays(uri):
    with _telem_lock:
        return {k: np.array(v) for k, v in _telem[uri].items()}

# =============================================================================
#  SECTION 3 — LPS HELPERS
# =============================================================================

def _read_lps_position(cf) -> np.ndarray:
    """Wait for Kalman filter to settle, then read absolute LPS position."""
    print(f"  [LPS] Waiting {LPS_SETTLE} s for estimator to settle ...")
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
    """
    Log kalman state + ctrltarget setpoint into _telem[uri].
    ctrltarget.x/y/z is the firmware's current position setpoint —
    the most accurate record of what the drone was told to do.
    """
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
    return lc   # caller does lc.start()

# =============================================================================
#  SECTION 4 — CSV SAVE
# =============================================================================

def save_csv(uri, out_dir):
    d = _telem_arrays(uri)
    if len(d['t']) == 0:
        return
    # Normalise timestamps to seconds from mission start
    d['t'] = d['t'] - d['t'][0]
    label = uri[-1]
    path  = os.path.join(out_dir, f'drone_{label}_telem.csv')
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['t','ax','ay','az','ex','ey','ez'])
        w.writeheader()
        for i in range(len(d['t'])):
            w.writerow({k: float(d[k][i]) for k in d})
    print(f"  [CSV] Saved {path}")

# =============================================================================
#  SECTION 5 — PER-DRONE FLIGHT THREAD
# =============================================================================

def _drone_thread(uri, other_uri, ready_evt, go_evt, stop_evt):
    label   = uri[-1]
    land_x, land_y = LAND_7 if uri == URI_7 else LAND_2

    # Phase offset: Drone 7 at 0°, Drone 2 at 180°
    start_angle = 0.0 if uri == URI_7 else math.pi
    start_x, start_y, start_z = CIRCLE_START_7 if uri == URI_7 else CIRCLE_START_2

    print(f"\n[Drone {label}] Connecting ...")
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

            # Read absolute spawn position from LPS
            spawn = _read_lps_position(cf)
            with _state_lock:
                _drone_pos[uri] = spawn
            print(f"[Drone {label}] Spawn: ({spawn[0]:.2f}, {spawn[1]:.2f}, {spawn[2]:.2f})")

            # Safety check
            margin = 0.1
            if not (SAFE_X_MIN + margin <= spawn[0] <= SAFE_X_MAX - margin and
                    SAFE_Y_MIN + margin <= spawn[1] <= SAFE_Y_MAX - margin):
                raise RuntimeError(
                    f"Drone {label} spawn ({spawn[0]:.2f},{spawn[1]:.2f}) "
                    f"is outside the safe zone. Aborting.")

            state_log = _start_state_logging(cf, uri)
            telem_log = _start_telem_logging(cf, uri)

            print(f"[Drone {label}] Ready — waiting for partner ...")
            ready_evt.set()
            go_evt.wait()

            telem_log.start()   # start recording from takeoff

            # ── Step 1: Takeoff ──────────────────────────────────────────────
            print(f"[Drone {label}] TAKEOFF → z={FLY_Z} m")
            hlc.takeoff(FLY_Z, TAKEOFF_DUR)
            time.sleep(TAKEOFF_DUR + 1.0)

            # ── Step 2: Fly to ellipse entry point (at its tilted height) ────
            dist = math.sqrt((start_x - spawn[0])**2 + (start_y - spawn[1])**2)
            entry_dur = max(3.0, dist / 0.5)
            print(f"[Drone {label}] ENTRY → ({start_x:.2f},{start_y:.2f},{start_z:.2f})  "
                  f"angle={math.degrees(start_angle):.0f}°  dur={entry_dur:.1f}s")
            hlc.go_to(start_x, start_y, start_z, yaw=0.0,
                      duration_s=entry_dur, relative=False)
            time.sleep(entry_dur + 0.5)

            # ── Step 3: Dense CCW tilted-ellipse loop ────────────────────────
            # Drone 2 starts RADIO_STAGGER seconds after Drone 7 within each tick
            # so both radios never transmit at the exact same instant.
            if uri == URI_2:
                time.sleep(RADIO_STAGGER)

            steps = int(CIRCLE_DURATION / CIRCLE_DT)
            theta = start_angle

            print(f"[Drone {label}] ELLIPSE  a={ELLIPSE_A}m b={ELLIPSE_B}m  "
                  f"tilt=±{TILT_AMP}m  ω={OMEGA}rad/s  laps={CIRCLE_LAPS}  steps={steps}")

            for step in range(steps):
                theta += OMEGA * CIRCLE_DT   # advance angle each tick

                wx, wy, wz = _ellipse_point(theta)

                # Hard clamp inside safe zone / altitude (safety net for edge cases)
                wx = max(SAFE_X_MIN + 0.05, min(SAFE_X_MAX - 0.05, wx))
                wy = max(SAFE_Y_MIN + 0.05, min(SAFE_Y_MAX - 0.05, wy))
                wz = max(Z_MIN, min(Z_MAX, wz))

                hlc.go_to(wx, wy, wz, yaw=0.0,
                          duration_s=CIRCLE_DT, relative=False)
                time.sleep(CIRCLE_DT)

                if step % 90 == 0:
                    deg = math.degrees(theta) % 360
                    print(f"[Drone {label}] ELLIPSE step {step+1:4d}/{steps}  "
                          f"({wx:.2f},{wy:.2f},{wz:.2f})  {deg:.0f}°")

            # ── Step 4: Fly to landing pad (settle back to FLY_Z) ────────────
            print(f"[Drone {label}] LAND-NAV → ({land_x:.2f},{land_y:.2f})")
            hlc.go_to(land_x, land_y, FLY_Z, yaw=0.0,
                      duration_s=4.0, relative=False)
            time.sleep(4.5)

            # ── Step 5: Land ─────────────────────────────────────────────────
            print(f"[Drone {label}] LANDING ↓")
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
#  SECTION 6 — SYNTHETIC DATA GENERATOR  (--sim mode)
# =============================================================================

def _lcg(seed):
    s = int(seed) & 0xFFFFFFFF
    while True:
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        yield (s >> 1) / 0x7FFFFFFF - 1.0


def generate_synthetic_mission(seed7=42, seed2=77):
    """
    Simulate the full mission. Drones stay 180° apart in theta on a shared
    tilted ellipse, so their altitudes are always mirror images of each other:
    when Drone 7 is near the high point, Drone 2 is near the low point.
    Returns d7, d2 — dicts of numpy arrays (t, ax, ay, az, ex, ey, ez).
    """
    rng7 = _lcg(seed7)
    rng2 = _lcg(seed2)
    T    = np.arange(0, T_TOTAL + CIRCLE_DT, CIRCLE_DT)

    def clamp(v, lo, hi): return float(np.clip(v, lo, hi))

    def ease(f):
        f = clamp(f, 0, 1)
        return 2*f**2 if f < 0.5 else 1 - (-2*f + 2)**2 / 2

    def lerp(t, t0, t1, v0, v1):
        return v0 + (v1 - v0) * ease((t - t0) / max(t1 - t0, 1e-9))

    def ellipse_xyz(t, start_ang):
        """Position on the tilted ellipse at time t, starting from start_ang, CCW."""
        frac  = clamp((t - T_ENTRY_END) / CIRCLE_DURATION, 0.0, 1.0)
        theta = start_ang + frac * CIRCLE_LAPS * 2 * math.pi
        return _ellipse_point(theta)

    def phase(t):
        if t <= T_TAKEOFF_END:  return 'takeoff'
        if t <= T_ENTRY_END:    return 'entry'
        if t <= T_CIRCLE_END:   return 'circle'
        if t <= T_LANDNAV_END:  return 'landnav'
        return 'land'

    out7 = {k: [] for k in ('t','ax','ay','az','ex','ey','ez')}
    out2 = {k: [] for k in ('t','ax','ay','az','ex','ey','ez')}
    last7 = list(CIRCLE_START_7)
    last2 = list(CIRCLE_START_2)

    for t in T:
        if t > T_TOTAL:
            break
        ph = phase(t)
        n7 = next(rng7)
        n2 = next(rng2)

        if ph == 'takeoff':
            ex7, ey7 = 1.0, 2.0
            ez7 = lerp(t, 0, T_TAKEOFF_END, 0.0, FLY_Z)
            ex2, ey2 = 2.0, 3.0
            ez2 = lerp(t, 0, T_TAKEOFF_END, 0.0, FLY_Z)

        elif ph == 'entry':
            ex7 = lerp(t, T_TAKEOFF_END, T_ENTRY_END, 1.0, CIRCLE_START_7[0])
            ey7 = lerp(t, T_TAKEOFF_END, T_ENTRY_END, 2.0, CIRCLE_START_7[1])
            ez7 = lerp(t, T_TAKEOFF_END, T_ENTRY_END, FLY_Z, CIRCLE_START_7[2])
            ex2 = lerp(t, T_TAKEOFF_END, T_ENTRY_END, 2.0, CIRCLE_START_2[0])
            ey2 = lerp(t, T_TAKEOFF_END, T_ENTRY_END, 3.0, CIRCLE_START_2[1])
            ez2 = lerp(t, T_TAKEOFF_END, T_ENTRY_END, FLY_Z, CIRCLE_START_2[2])

        elif ph == 'circle':
            ex7, ey7, ez7 = ellipse_xyz(t, 0.0)         # Drone 7 at θ
            ex2, ey2, ez2 = ellipse_xyz(t, math.pi)     # Drone 2 at θ + π
            last7 = [ex7, ey7, ez7]
            last2 = [ex2, ey2, ez2]

        elif ph == 'landnav':
            ex7 = lerp(t, T_CIRCLE_END, T_LANDNAV_END, last7[0], LAND_7[0])
            ey7 = lerp(t, T_CIRCLE_END, T_LANDNAV_END, last7[1], LAND_7[1])
            ez7 = lerp(t, T_CIRCLE_END, T_LANDNAV_END, last7[2], FLY_Z)
            ex2 = lerp(t, T_CIRCLE_END, T_LANDNAV_END, last2[0], LAND_2[0])
            ey2 = lerp(t, T_CIRCLE_END, T_LANDNAV_END, last2[1], LAND_2[1])
            ez2 = lerp(t, T_CIRCLE_END, T_LANDNAV_END, last2[2], FLY_Z)

        else:  # land
            ex7, ey7 = LAND_7
            ez7 = lerp(t, T_LANDNAV_END, T_TOTAL, FLY_Z, 0.0)
            ex2, ey2 = LAND_2
            ez2 = lerp(t, T_LANDNAV_END, T_TOTAL, FLY_Z, 0.0)

        noise = 0.8   # scale for LPS-like noise
        for d, ex, ey, ez, n in [(out7, ex7, ey7, ez7, n7),
                                  (out2, ex2, ey2, ez2, n2)]:
            d['t'].append(float(t))
            d['ex'].append(ex);  d['ey'].append(ey);  d['ez'].append(ez)
            d['ax'].append(ex + n * 0.035 * noise)
            d['ay'].append(ey + n * 0.028 * noise)
            d['az'].append(ez + n * 0.018)

    def to_np(d): return {k: np.array(v) for k, v in d.items()}
    return to_np(out7), to_np(out2)

# =============================================================================
#  SECTION 7 — METRICS
# =============================================================================

def _mae(actual, expected):
    return float(np.mean(np.abs(actual - expected)))

def _rms(actual, expected):
    return float(np.sqrt(np.mean((actual - expected) ** 2)))

def _rolling_rms_3d(d, win):
    err2 = (d['ax']-d['ex'])**2 + (d['ay']-d['ey'])**2 + (d['az']-d['ez'])**2
    return np.sqrt(np.convolve(err2, np.ones(win)/win, mode='same'))

def print_summary(d7, d2):
    print("\n" + "="*52)
    print("   AXIS-WISE TRACKING MAE SUMMARY")
    print("="*52)
    for lbl, d in [('Drone 7', d7), ('Drone 2', d2)]:
        mx = _mae(d['ax'], d['ex'])
        my = _mae(d['ay'], d['ey'])
        mz = _mae(d['az'], d['ez'])
        print(f"  {lbl}  X:{mx*100:.2f}cm  Y:{my*100:.2f}cm  Z:{mz*100:.2f}cm")
    print("="*52 + "\n")

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


def _add_safezone(ax):
    ax.add_patch(mpatches.FancyBboxPatch(
        (SAFE_X_MIN, SAFE_Y_MIN),
        SAFE_X_MAX-SAFE_X_MIN, SAFE_Y_MAX-SAFE_Y_MIN,
        boxstyle='square,pad=0', linewidth=1.0,
        edgecolor='#444', facecolor='#eef3f8', zorder=0, alpha=0.4))
    bx = [SAFE_X_MIN, SAFE_X_MAX, SAFE_X_MAX, SAFE_X_MIN, SAFE_X_MIN]
    by = [SAFE_Y_MIN, SAFE_Y_MIN, SAFE_Y_MAX, SAFE_Y_MAX, SAFE_Y_MIN]
    ax.plot(bx, by, 'r--', lw=1.1, alpha=0.5, label='Safety boundary')


def _draw_ellipse_overlay(ax):
    theta = np.linspace(0, 2*math.pi, 360)
    ax.plot(CIRCLE_CX + ELLIPSE_A*np.cos(theta),
            CIRCLE_CY + ELLIPSE_B*np.sin(theta),
            ':', color=CG, lw=1.4, alpha=0.7, zorder=1,
            label=f'Planned ellipse  a={ELLIPSE_A}m b={ELLIPSE_B}m')
    # 16 evenly-spaced reference tick marks
    a16 = np.linspace(0, 2*math.pi, 17)[:-1]
    ax.scatter(CIRCLE_CX + ELLIPSE_A*np.cos(a16),
               CIRCLE_CY + ELLIPSE_B*np.sin(a16),
               s=18, color=CG, alpha=0.45, zorder=3)
    ax.scatter(CIRCLE_CX, CIRCLE_CY, marker='+', s=90,
               color=CG, linewidths=1.4, zorder=4)
    ax.annotate(f'Centre\n({CIRCLE_CX},{CIRCLE_CY})',
                (CIRCLE_CX, CIRCLE_CY),
                textcoords='offset points', xytext=(6, -16),
                fontsize=7.5, color=CG)


def _phase_vlines(ax, ymax):
    quarter = (2*math.pi / OMEGA) / 4
    marks = [
        (T_TAKEOFF_END,                 'Takeoff'),
        (T_ENTRY_END,                   'Ellipse start'),
        (T_ENTRY_END + quarter,         '90°'),
        (T_ENTRY_END + 2*quarter,       '180°'),
        (T_ENTRY_END + 3*quarter,       '270°'),
        (T_ENTRY_END + 4*quarter,       'Lap 1'),
        (T_ENTRY_END + 8*quarter,       'Lap 2'),
        (T_CIRCLE_END,                  'Land nav'),
    ]
    for tv, name in marks:
        if tv > T_TOTAL:
            continue
        ax.axvline(tv, color=CG, ls=':', lw=0.6, alpha=0.45)
        ax.text(tv + 0.3, ymax * 0.93, name, fontsize=6.5, color=CG)

# =============================================================================
#  SECTION 9 — PLOTS
# =============================================================================

def plot_xy(d7, d2, out_dir):
    """Figure 1: XY top-down — actual vs expected with ellipse overlay."""
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.suptitle('Figure 1 — XY Top-Down: Actual vs Expected  (π phase separation)',
                 fontweight='bold')
    _add_safezone(ax)
    _draw_ellipse_overlay(ax)

    every = 3
    for d, color, label in [(d7, C7, 'Drone 7'), (d2, C2, 'Drone 2')]:
        ax.plot(d['ex'][::every], d['ey'][::every], '--', color=color, lw=0.9, alpha=0.5)
        ax.plot(d['ax'][::every], d['ay'][::every], '-',  color=color, lw=1.4,
                label=f'{label} actual')

    # Entry points (always diametrically opposite in theta)
    ax.scatter(CIRCLE_START_7[0], CIRCLE_START_7[1], marker='>', s=100, color=C7, zorder=5)
    ax.scatter(CIRCLE_START_2[0], CIRCLE_START_2[1], marker='<', s=100, color=C2, zorder=5)
    ax.annotate(f'D7 entry\n(0°, HIGH)', (CIRCLE_START_7[0], CIRCLE_START_7[1]),
                textcoords='offset points', xytext=(6, 4), fontsize=8, color=C7)
    ax.annotate(f'D2 entry\n(180°, LOW)', (CIRCLE_START_2[0], CIRCLE_START_2[1]),
                textcoords='offset points', xytext=(-58, 4), fontsize=8, color=C2)

    # Landing pads
    ax.scatter(*LAND_7, marker='*', s=140, color=C7, zorder=5)
    ax.scatter(*LAND_2, marker='*', s=140, color=C2, zorder=5)
    ax.annotate('L7', LAND_7, textcoords='offset points', xytext=(5, 5), fontsize=8)
    ax.annotate('L2', LAND_2, textcoords='offset points', xytext=(5, 5), fontsize=8)

    handles = [
        Line2D([0],[0], color=C7, lw=1.4,  label='Drone 7 actual'),
        Line2D([0],[0], color=C7, lw=0.9, ls='--', alpha=0.6, label='Drone 7 expected'),
        Line2D([0],[0], color=C2, lw=1.4,  label='Drone 2 actual'),
        Line2D([0],[0], color=C2, lw=0.9, ls='--', alpha=0.6, label='Drone 2 expected'),
        Line2D([0],[0], color=CG, lw=1.4, ls=':', alpha=0.7,
               label=f'Planned ellipse a={ELLIPSE_A}m b={ELLIPSE_B}m'),
        Line2D([0],[0], color='r', lw=1.1, ls='--', alpha=0.5, label='Safety boundary'),
        Line2D([0],[0], marker='*', color=C7, lw=0, markersize=9, label='Land pad 7'),
        Line2D([0],[0], marker='*', color=C2, lw=0, markersize=9, label='Land pad 2'),
    ]
    ax.legend(handles=handles, fontsize=7.5, loc='upper left', framealpha=0.85)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_xlim(-0.15, 3.15); ax.set_ylim(0.75, 4.25)
    ax.set_aspect('equal')
    ax.set_title(
        f'z tilts ±{TILT_AMP}m around {FLY_Z}m as drones go around  |  Phase Δ=π  |  '
        f'c=({CIRCLE_CX},{CIRCLE_CY})  a={ELLIPSE_A}m b={ELLIPSE_B}m  ω={OMEGA}rad/s  '
        f'laps={CIRCLE_LAPS}', fontsize=8.5)

    plt.tight_layout()
    path = os.path.join(out_dir, '01_xy_trajectory.png')
    fig.savefig(path); plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_axes(d7, d2, out_dir):
    """Figure 2: Per-axis position timelines."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Figure 2 — Axis-wise Position Timeline: Actual vs Expected',
                 fontweight='bold')
    every = 2

    for d, color, label in [(d7, C7, 'Drone 7'), (d2, C2, 'Drone 2')]:
        T = d['t'][::every]
        for i, (ak, ek) in enumerate([('ax','ex'), ('ay','ey'), ('az','ez')]):
            mae = _mae(d[ak], d[ek])
            axes[i].plot(T, d[ek][::every], '--', color=color, lw=1.3, alpha=0.55)
            axes[i].plot(T, d[ak][::every], '-',  color=color, lw=1.2,
                         label=f'{label}  (MAE={mae*100:.2f}cm)')

    # Reference altitude band (tilt swings between these two lines)
    axes[2].axhline(FLY_Z + TILT_AMP, color=CG, ls=':', lw=1.0, alpha=0.6,
                    label=f'High side z={FLY_Z+TILT_AMP:.2f}m')
    axes[2].axhline(FLY_Z - TILT_AMP, color=CG, ls=':', lw=1.0, alpha=0.6,
                    label=f'Low side z={FLY_Z-TILT_AMP:.2f}m')

    configs = [
        ('X position (m)', (SAFE_X_MIN-0.1, SAFE_X_MAX+0.1), 'X-Axis Tracking'),
        ('Y position (m)', (SAFE_Y_MIN-0.1, SAFE_Y_MAX+0.1), 'Y-Axis Tracking'),
        ('Altitude Z (m)', (Z_MIN-0.15, Z_MAX+0.15),          'Z-Axis (tilted — swaps high/low as drones orbit)'),
    ]
    for ax, (ylabel, ylim, title) in zip(axes, configs):
        _phase_vlines(ax, ylim[1])
        ax.set_ylabel(ylabel); ax.set_ylim(*ylim)
        ax.set_title(title + '  (Dashed = expected / ctrltarget)', fontsize=9)
        ax.legend(fontsize=7.5, ncol=2, loc='upper right', framealpha=0.8)

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    path = os.path.join(out_dir, '02_xyz_axes.png')
    fig.savefig(path); plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_rms(d7, d2, out_dir):
    """Figure 3: Per-axis RMS bars + rolling 3-D RMS curve."""
    rms7 = {a: _rms(d7[f'a{a}'], d7[f'e{a}']) for a in ('x','y','z')}
    rms2 = {a: _rms(d2[f'a{a}'], d2[f'e{a}']) for a in ('x','y','z')}
    rms7['3D'] = math.sqrt(sum(rms7[a]**2 for a in ('x','y','z')))
    rms2['3D'] = math.sqrt(sum(rms2[a]**2 for a in ('x','y','z')))

    win   = 50   # 5 s rolling window at 0.1 s/sample
    roll7 = _rolling_rms_3d(d7, win)
    roll2 = _rolling_rms_3d(d2, win)

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle('Figure 3 — RMS Tracking Error Analysis', fontweight='bold')
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.3)

    for idx, (lbl, rms_d, color) in enumerate([('7', rms7, C7), ('2', rms2, C2)]):
        ax = fig.add_subplot(gs[0, idx])
        keys = ['x','y','z','3D']
        vals = [rms_d[k] for k in keys]
        bars = ax.bar(['X','Y','Z','3D'], vals,
                      color=[color+'88']*3 + [color],
                      edgecolor=color, linewidth=0.8, width=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.0002,
                    f'{val*100:.2f}cm', ha='center', va='bottom', fontsize=8)
        ax.set_title(f'Per-Axis RMS — Drone {lbl}')
        ax.set_ylabel('RMS error (m)')
        ax.set_ylim(0, max(vals) * 1.4)

    ax_r = fig.add_subplot(gs[1, :])
    every = 3
    ax_r.plot(d7['t'][::every], roll7[::every], color=C7, lw=1.4,
              label='Drone 7  rolling 3D RMS')
    ax_r.plot(d2['t'][::every], roll2[::every], color=C2, lw=1.4,
              label='Drone 2  rolling 3D RMS')
    ax_r.fill_between(d7['t'][::every], roll7[::every], alpha=0.12, color=C7)
    ax_r.fill_between(d2['t'][::every], roll2[::every], alpha=0.12, color=C2)
    _phase_vlines(ax_r, max(roll7.max(), roll2.max()) * 1.1 or 0.1)
    ax_r.set_xlabel('Time (s)'); ax_r.set_ylabel('3D RMS error (m)')
    ax_r.set_title(
        f'Rolling 3D RMS (win={win*CIRCLE_DT:.0f}s) | '
        f'Drone7={rms7["3D"]*100:.2f}cm  Drone2={rms2["3D"]*100:.2f}cm')
    ax_r.legend(fontsize=9)

    path = os.path.join(out_dir, '03_rms_error.png')
    fig.savefig(path); plt.close(fig)
    print(f"  Saved: {path}")
    return path, rms7, rms2


def plot_separation(d7, d2, out_dir):
    """
    Figure 4: Inter-drone 3D distance.
    On a flat circle this would sit near 2R at all times; on the tilted
    ellipse it now oscillates, since the two drones are also offset in Z
    (one near the high point, one near the low point) as well as XY.
    """
    n = min(len(d7['t']), len(d2['t']))
    sep_xy = np.sqrt((d7['ax'][:n]-d2['ax'][:n])**2 +
                     (d7['ay'][:n]-d2['ay'][:n])**2)
    sep_3d = np.sqrt((d7['ax'][:n]-d2['ax'][:n])**2 +
                      (d7['ay'][:n]-d2['ay'][:n])**2 +
                      (d7['az'][:n]-d2['az'][:n])**2)
    T   = d7['t'][:n]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle('Figure 4 — Inter-Drone Separation (XY vs full 3D)',
                 fontweight='bold')

    every = 3
    ax.plot(T[::every], sep_xy[::every], color=C7, lw=1.3, alpha=0.7, label='XY distance')
    ax.plot(T[::every], sep_3d[::every], color=CA, lw=1.6, label='3D distance')
    ax.axhline(0.5, color=C2, ls='--', lw=1.0, label='Proximity alert (< 0.5m)')

    # Shade under the proximity threshold
    ax.fill_between(T[::every], 0, np.minimum(sep_3d[::every], 0.5),
                    alpha=0.2, color=C2)

    # Shade the ellipse phase
    ax.axvspan(T_ENTRY_END, T_CIRCLE_END, alpha=0.06, color=CA,
               label='Ellipse phase')

    _phase_vlines(ax, sep_3d.max() * 1.15)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Distance (m)')
    ax.set_ylim(0, sep_3d.max() * 1.20)
    ax.legend(fontsize=8.5, loc='upper right')
    ax.set_title(
        f'3D min={sep_3d.min():.3f}m  avg={sep_3d.mean():.3f}m  '
        f'(tilt adds a Z component on top of the XY separation)', fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, '04_separation.png')
    fig.savefig(path); plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_3d(d7, d2, out_dir):
    """Figure 5: 3-D trajectory — the tilted ellipse itself."""
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle('Figure 5 — 3-D Trajectory: Actual vs Expected', fontweight='bold')
    ax3 = fig.add_subplot(111, projection='3d')

    every = 3
    for d, color, label in [(d7, C7, 'Drone 7'), (d2, C2, 'Drone 2')]:
        ax3.plot(d['ax'][::every], d['ay'][::every], d['az'][::every],
                 color=color, lw=1.3, label=f'{label} actual')
        ax3.plot(d['ex'][::every], d['ey'][::every], d['ez'][::every],
                 '--', color=color, lw=0.8, alpha=0.5, label=f'{label} expected')

    # Draw the tilted-ellipse skeleton itself
    theta = np.linspace(0, 2*math.pi, 300)
    skel = np.array([_ellipse_point(th) for th in theta])
    ax3.plot(skel[:,0], skel[:,1], skel[:,2],
             ':', color=CG, lw=1.1, alpha=0.6, label='Tilted ellipse path')

    ax3.set_xlabel('X (m)'); ax3.set_ylabel('Y (m)'); ax3.set_zlabel('Z (m)')
    ax3.set_title(
        f'Tilted ellipse: a={ELLIPSE_A}m b={ELLIPSE_B}m  z=FLY_Z±{TILT_AMP}m  |  '
        f'Phase Δ=π  →  drones sit on opposite sides of the tilt', fontsize=9)
    ax3.legend(fontsize=8, loc='upper left')

    plt.tight_layout()
    path = os.path.join(out_dir, '05_trajectory_3d.png')
    fig.savefig(path); plt.close(fig)
    print(f"  Saved: {path}")
    return path


def generate_all_plots(d7, d2, out_dir='.'):
    os.makedirs(out_dir, exist_ok=True)
    print('\n' + '='*55)
    print('  Post-Flight Analysis — Generating 5 Plots')
    print('='*55)
    print_summary(d7, d2)

    plot_xy(d7, d2, out_dir)
    plot_axes(d7, d2, out_dir)
    _, rms7, rms2 = plot_rms(d7, d2, out_dir)
    plot_separation(d7, d2, out_dir)
    plot_3d(d7, d2, out_dir)

    print()
    print('  ─── RMS Summary ─────────────────────────────────────────')
    for lbl, rd in [('Drone 7', rms7), ('Drone 2', rms2)]:
        print(f'  {lbl}  X:{rd["x"]*100:.2f}cm  Y:{rd["y"]*100:.2f}cm  '
              f'Z:{rd["z"]*100:.2f}cm  →  3D: {rd["3D"]*100:.2f}cm')
    print('  ─────────────────────────────────────────────────────────')
    print(f'\n  ✓ All plots saved to "{os.path.abspath(out_dir)}"')

# =============================================================================
#  SECTION 10 — REAL FLIGHT MAIN
# =============================================================================

def run_real_flight(out_dir):
    if not CF_AVAILABLE:
        raise RuntimeError(
            "cflib not found.  pip install cflib\n"
            "Or simulate:  python real_boids_swarm_tilted_ellipse.py --sim")

    print('='*55)
    print('  Dual-Drone Tilted Ellipse  —  π phase separation')
    print('='*55)
    print(f'  Nominal altitude z = {FLY_Z} m,  tilt swing = ±{TILT_AMP} m')
    print(f'  Drone 7  start={CIRCLE_START_7}  land={LAND_7}')
    print(f'  Drone 2  start={CIRCLE_START_2}  land={LAND_2}')
    print(f'  Ellipse  c=({CIRCLE_CX},{CIRCLE_CY})  a={ELLIPSE_A}m b={ELLIPSE_B}m  '
          f'ω={OMEGA}rad/s  laps={CIRCLE_LAPS}  dt={CIRCLE_DT}s')
    print(f'  Safe zone  X[{SAFE_X_MIN},{SAFE_X_MAX}]  Y[{SAFE_Y_MIN},{SAFE_Y_MAX}]  '
          f'Z[{Z_MIN},{Z_MAX}]')
    print('='*55)
    input('\nPress ENTER to verify anchors and start: ')

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
        print('[Main] Pre-flight timeout or error. Aborting.')
        stop_evt.set()
        return

    print('[Main] Both drones ready — departing in 3 s ...')
    time.sleep(3.0)
    go_evt.set()

    try:
        t7.join(); t2.join()
    except KeyboardInterrupt:
        print('\n[Main] Abort ...')
        stop_evt.set()
        t7.join(timeout=5); t2.join(timeout=5)

    print('\n[Main] Mission complete.')
    os.makedirs(out_dir, exist_ok=True)
    save_csv(URI_7, out_dir)
    save_csv(URI_2, out_dir)
    d7 = _telem_arrays(URI_7)
    d2 = _telem_arrays(URI_2)
    generate_all_plots(d7, d2, out_dir)

# =============================================================================
#  SECTION 11 — ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Two Crazyflies revolving on a shared tilted ellipse with π phase separation.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sim',   action='store_true', help='Simulate without hardware')
    parser.add_argument('--out',   metavar='DIR', default='./plots',
                        help='Output directory for PNGs and CSVs')
    parser.add_argument('--show',  action='store_true', help='Open interactive windows')
    parser.add_argument('--seed7', type=int, default=42)
    parser.add_argument('--seed2', type=int, default=77)
    args = parser.parse_args()

    if args.show:
        matplotlib.use('TkAgg')

    if args.sim:
        print('='*55)
        print('  Simulation mode')
        print(f'  Ellipse  c=({CIRCLE_CX},{CIRCLE_CY})  a={ELLIPSE_A}m b={ELLIPSE_B}m  '
              f'ω={OMEGA}rad/s  laps={CIRCLE_LAPS}')
        print(f'  Tilt     z = {FLY_Z} ± {TILT_AMP} m  (phase={TILT_PHASE} rad)')
        print(f'  Drone 7  angle=  0°  entry={CIRCLE_START_7}')
        print(f'  Drone 2  angle=180°  entry={CIRCLE_START_2}  (π offset → opposite tilt side)')
        print(f'  Total mission: ~{T_TOTAL:.0f}s')
        print('='*55)
        d7, d2 = generate_synthetic_mission(seed7=args.seed7, seed2=args.seed2)
        print(f'  Generated {len(d7["t"])} samples  ({d7["t"][-1]:.1f}s)')
        generate_all_plots(d7, d2, args.out)
        if args.show:
            plt.show()
    else:
        run_real_flight(args.out)

if __name__ == '__main__':
    main()
