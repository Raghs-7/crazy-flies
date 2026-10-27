#!/usr/bin/env python3
"""
leader_follower_square.py - leader flies a SQUARE, follower replays it TAU seconds later.

LEADER   : takes off, flies to the first corner of a square, then flies the square
           (default 2 laps) at LEADER_SPEED. Its LPS position (kalman.stateX/Y/Z, 10 Hz)
           is read and timestamped on the PC.
FOLLOWER : takes off, then every 100 ms flies to the leader's position from
           TAU seconds ago (rolling buffer), but never closer than D_MIN
           (horizontal) to the leader's CURRENT position.   (logic unchanged)

    reference   D(t)  = p_leader(t - TAU)
    keep-out    if |D - p_leader(t)|_xy < D_MIN  ->  push D out to D_MIN from the leader
    target      geofence-clamped, speed-limited, sent with go_to()

LIVE PLOT window (top-down XY + separation + X(t) + Y(t)) is shown while flying.
Closing the window while flying = emergency landing of both drones.

Run the no-risk version first (NO radios, NO motors, leader and follower are both
simulated, so you can check the square, the live plot and the report):
    python leader_follower_square.py --dry-run
Then the real thing:
    python leader_follower_square.py

Useful options:
    --laps 2  --side 1.4  --center-x 1.5 --center-y 2.5  --leader-speed 0.4  --cw
    --tau 3.0 --dmin 0.5  --vmax 0.8  --follow-z  --no-live

Ctrl+C at any time -> both drones land.
"""

import argparse
import csv
import os
import sys
import threading
import time
import traceback
from contextlib import ExitStack

import numpy as np
import matplotlib
if '--no-live' in sys.argv:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from matplotlib.patches import Circle

try:
    import cflib.crtp
    from cflib.crazyflie import Crazyflie
    from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
    from cflib.crazyflie.log import LogConfig
    CF_AVAILABLE = True
except ImportError:
    CF_AVAILABLE = False

NON_GUI_BACKENDS = ('agg', 'pdf', 'svg', 'ps', 'cairo', 'template', 'pgf')
LIVE_OK = matplotlib.get_backend().lower() not in NON_GUI_BACKENDS

# =============================================================================
#  CONSTANTS
# =============================================================================

LEADER_URI   = 'radio://0/80/2M/E7E7E7E703'     # Drone 7: flies the square
FOLLOWER_URI = 'radio://0/80/2M/E7E7E7E701'     # Drone 2: flies the delayed copy

SAFE_X = (0.0, 3.0)
SAFE_Y = (1.0, 4.0)
GEO_MARGIN = 0.10          # follower targets stay this far inside the safe zone
ZONE_BREACH = 0.15         # a drone measured this far OUTSIDE the zone -> land both

FLY_Z = 1.0                # follower altitude (m)
LEADER_Z = 1.0             # leader altitude (m)
Z_MIN, Z_MAX = 0.5, 1.5    # altitude limits when --follow-z is used

# ---- leader square ----------------------------------------------------------
SQ_CENTER = (1.5, 2.5)     # m   centre of the square
SQ_SIDE   = 1.4            # m   side length  -> x: 0.8..2.2, y: 1.8..3.2
SQ_SPEED  = 0.4            # m/s leader speed along the square
SQ_LAPS   = 2
SQ_RAMP_S = 1.5            # s   leader speed ramps up 0 -> SQ_SPEED over this time
SQ_MARGIN = 0.3            # m   the whole square must stay this far inside the safe zone
FINISH_EXTRA_S = 2.0       # s   after the last lap the leader hovers TAU + this, then both land
LEADER_DEV_MAX = 0.6       # m   leader measured this far from its own command -> land both

TAU    = 3.0               # s   time buffer (follower uses leader position from TAU ago)
D_MIN  = 0.5               # m   minimum horizontal distance to the leader
V_MAX  = 0.8               # m/s follower speed limit
V_ESCAPE = 1.2             # m/s speed limit while inside the keep-out circle (move away faster)

DT           = 0.1         # s   control period (10 Hz)
TAKEOFF_DUR  = 2.5
LAND_DUR     = 3.0
LPS_SETTLE   = 10.0        # s   wait for the Kalman filters after connecting
STALE_HOLD_S = 0.5         # leader data older than this -> follower holds
LEADER_LOST_S = 3.0        # leader data older than this -> land both
MAX_FLIGHT_S = 120.0       # follow phase is cut off after this long

# Pass criteria from the progress document
RATE_MIN_HZ, GAP_MAX_S = 9.0, 0.5
LAG_TOL_S = 0.3
SEP_TOL_M = 0.05           # tolerance below D_MIN when checking "separation >= D_MIN"

ROW_KEYS = ['t', 'phase', 'lphase', 'lx', 'ly', 'lz', 'leader_age', 'lcx', 'lcy', 'lcz',
            'rx', 'ry', 'rz', 'tx', 'ty', 'tz', 'cx', 'cy', 'cz', 'fx', 'fy', 'fz',
            'dist_xy', 'dist_ref', 'ex', 'ey', 'ez']

# =============================================================================
#  TIMESTAMPED POSITION BUFFER
# =============================================================================

class PosStream:
    """Thread-safe history of (PC time, xyz) for one drone."""

    def __init__(self):
        self._lock = threading.Lock()
        self.t, self.p = [], []

    def push(self, t, p):
        with self._lock:
            self.t.append(float(t))
            self.p.append((float(p[0]), float(p[1]), float(p[2])))

    def latest(self):
        with self._lock:
            if not self.t:
                return None
            return self.t[-1], np.array(self.p[-1])

    def span(self):
        with self._lock:
            return self.t[-1] - self.t[0] if len(self.t) > 1 else 0.0

    def at(self, tq):
        """Linear interpolation of the position at PC time tq (None if the
        history does not reach back that far)."""
        with self._lock:
            if len(self.t) < 2 or tq < self.t[0]:
                return None
            ts = np.array(self.t[-300:])
            ps = np.array(self.p[-300:])
        if tq < ts[0]:
            return None
        return np.array([np.interp(tq, ts, ps[:, k]) for k in range(3)])

    def arrays(self):
        with self._lock:
            return np.array(self.t), np.array(self.p)


def start_pos_logging(cf, name, stream):
    lc = LogConfig(name=name, period_in_ms=100)
    for v in ('kalman.stateX', 'kalman.stateY', 'kalman.stateZ'):
        lc.add_variable(v, 'float')

    def _cb(ts, data, _lc):
        stream.push(time.time(),
                    (data['kalman.stateX'], data['kalman.stateY'], data['kalman.stateZ']))

    cf.log.add_config(lc)
    lc.data_received_cb.add_callback(_cb)
    lc.start()
    return lc

# =============================================================================
#  GEOMETRY: reference -> target -> command   (follower logic, unchanged)
# =============================================================================

def in_zone(xy, margin):
    return (SAFE_X[0] + margin <= xy[0] <= SAFE_X[1] - margin and
            SAFE_Y[0] + margin <= xy[1] <= SAFE_Y[1] - margin)


def compute_target(leader_now, ref, follower_xy, d_min, z):
    """Delayed reference, pushed out of the keep-out circle around the leader's
    current position, then kept inside the geofence. None -> hold."""
    des = ref[:2].copy()
    v = des - leader_now[:2]
    d = np.linalg.norm(v)
    if d < d_min:
        if d < 1e-3:                       # reference sits on the leader:
            v = follower_xy - leader_now[:2]   # approach from the follower's side
            d = np.linalg.norm(v)
            if d < 1e-3:
                v, d = np.array([1.0, 0.0]), 1.0
        des = leader_now[:2] + v / d * d_min

    lo = np.array([SAFE_X[0] + GEO_MARGIN, SAFE_Y[0] + GEO_MARGIN])
    hi = np.array([SAFE_X[1] - GEO_MARGIN, SAFE_Y[1] - GEO_MARGIN])
    xy = np.clip(des, lo, hi)

    if np.linalg.norm(xy - leader_now[:2]) < d_min - 0.02:
        # clamping pushed it back into the keep-out circle (leader near a wall):
        # choose the point ON the circle, inside the zone, closest to the wish
        ang = np.deg2rad(np.arange(0, 360, 15))
        cand = leader_now[:2] + d_min * np.column_stack([np.cos(ang), np.sin(ang)])
        ok = [c for c in cand if in_zone(c, GEO_MARGIN)]
        if not ok:
            return None
        xy = min(ok, key=lambda c: np.linalg.norm(c - des))
    return np.array([xy[0], xy[1], z])


def slew(cmd, tgt, leader_now, d_min, dt, vmax, vesc):
    """Move the command toward the target by at most v*dt."""
    inside = np.linalg.norm(cmd[:2] - leader_now[:2]) < d_min
    lim = (vesc if inside else vmax) * dt
    step = tgt - cmd
    n = np.linalg.norm(step)
    return tgt.copy() if n <= lim else cmd + step * (lim / n)


def step_toward(cmd, tgt, vmax, dt):
    """Plain speed-limited move toward a point (used by the leader)."""
    step = tgt - cmd
    n = np.linalg.norm(step)
    lim = vmax * dt
    return tgt.copy() if n <= lim else cmd + step * (lim / n)

# =============================================================================
#  LEADER SQUARE TRAJECTORY
# =============================================================================

class SquarePath:
    """Closed square, corners p0 -> p1 -> p2 -> p3 -> p0 (counter-clockwise by default)."""

    def __init__(self, cx, cy, side, ccw=True):
        h = side / 2.0
        pts = [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)]
        if not ccw:
            pts = [pts[0], pts[3], pts[2], pts[1]]
        self.pts = np.array(pts, dtype=float)
        self.side = float(side)
        self.perimeter = 4.0 * self.side

    def point(self, s):
        """XY point at arc length s (wraps around every perimeter)."""
        s = s % self.perimeter
        i = min(int(s // self.side), 3)
        f = (s - i * self.side) / self.side
        a, b = self.pts[i], self.pts[(i + 1) % 4]
        return a + (b - a) * f

    def outline(self):
        return np.vstack([self.pts, self.pts[:1]])

    def fits(self, margin):
        return all(in_zone(p, margin) for p in self.pts)


class LeaderSquare:
    """Leader command generator:  goto_start -> square (N laps) -> finish (hover) -> done."""

    def __init__(self, path, z, speed, laps, tau, start_cmd):
        self.path, self.z, self.speed, self.laps = path, z, speed, laps
        self.total = laps * path.perimeter
        self.hold_after = tau + FINISH_EXTRA_S
        self.cmd = np.array(start_cmd, dtype=float)
        self.phase = 'goto_start'
        self.s = 0.0
        self.t_sq0 = None
        self.t_fin = None

    def lap_text(self):
        if self.phase != 'square':
            return self.phase
        lap = min(int(self.s // self.path.perimeter) + 1, self.laps)
        return f"square lap {lap}/{self.laps}"

    def update(self, now, dt):
        """Advance one control tick. Returns (command xyz, finished?)."""
        done = False
        if self.phase == 'goto_start':
            tgt = np.array([self.path.pts[0][0], self.path.pts[0][1], self.z])
            self.cmd = step_toward(self.cmd, tgt, self.speed, dt)
            if np.linalg.norm(self.cmd - tgt) < 1e-6:
                self.phase, self.s, self.t_sq0 = 'square', 0.0, now
                print("[Leader] at start corner -> flying the SQUARE")
        elif self.phase == 'square':
            ramp = min(1.0, (now - self.t_sq0) / SQ_RAMP_S)
            self.s = min(self.total, self.s + self.speed * ramp * dt)
            p = self.path.point(self.s)
            self.cmd = np.array([p[0], p[1], self.z])
            if self.s >= self.total - 1e-9:
                p0 = self.path.pts[0]
                self.cmd = np.array([p0[0], p0[1], self.z])
                self.phase, self.t_fin = 'finish', now
                print(f"[Leader] {self.laps} lap(s) done -> hovering "
                      f"{self.hold_after:.1f} s so the follower can finish the path")
        else:  # finish
            done = (now - self.t_fin) >= self.hold_after
        return self.cmd, done

# =============================================================================
#  ROWS / ANALYSIS
# =============================================================================

def make_row(t, lphase, lat, age, ref, tgt, cmd, f, lcmd):
    nan = np.full(3, np.nan)
    ref_ = nan if ref is None else ref
    tgt_ = nan if tgt is None else tgt
    d = {'t': float(t), 'phase': 'follow', 'lphase': lphase, 'leader_age': float(age)}
    for name, v in (('l', lat), ('lc', lcmd), ('r', ref_), ('t', tgt_), ('c', cmd), ('f', f)):
        d[name + 'x'], d[name + 'y'], d[name + 'z'] = (float(v[0]), float(v[1]), float(v[2]))
    d['dist_xy'] = float(np.hypot(f[0] - lat[0], f[1] - lat[1]))
    d['dist_ref'] = float(np.hypot(f[0] - ref_[0], f[1] - ref_[1]))
    d['ex'], d['ey'], d['ez'] = (float(f[0] - tgt_[0]), float(f[1] - tgt_[1]), float(f[2] - tgt_[2]))
    return d


def estimate_lag(Lt, Lp, Ft, Fp, t_from=None, max_lag=6.0, step=0.1):
    """Lag L that best explains follower(t) ~ leader(t - L) in XY. None if the
    leader barely moved (flat curve -> lag not observable)."""
    if np.ptp(Lp[:, 0]) + np.ptp(Lp[:, 1]) < 0.8:
        return None
    lags = np.arange(0.0, max_lag + 1e-9, step)
    t0 = (Lt[0] + max_lag) if t_from is None else t_from
    m = Ft >= t0
    if m.sum() < 30:
        return None
    best, best_err = None, np.inf
    for lag in lags:
        tq = Ft[m] - lag
        lx = np.interp(tq, Lt, Lp[:, 0])
        ly = np.interp(tq, Lt, Lp[:, 1])
        err = np.mean(np.hypot(Fp[m, 0] - lx, Fp[m, 1] - ly))
        if err < best_err:
            best, best_err = lag, err
    return float(best)


def analyse(rows, L, args, dry):
    follow = [r for r in rows if r['phase'] == 'follow']
    if len(follow) < 5:
        print("\n  [analysis] too few follow samples.")
        return None
    col = lambda k: np.array([r[k] for r in follow])
    t = col('t')

    Lt, Lp = L.arrays()
    dts = np.diff(Lt)
    rate = 1.0 / np.mean(dts) if len(dts) else float('nan')
    max_gap = float(np.max(dts)) if len(dts) else float('nan')

    dist = col('dist_xy')
    e3 = np.sqrt(np.nansum(np.column_stack([col('ex'), col('ey'), col('ez')]) ** 2, axis=1))
    e3 = e3[np.isfinite(col('tx'))]
    track_rms = float(np.sqrt(np.mean(e3 ** 2))) if len(e3) else float('nan')

    Fp = np.column_stack([col('fx'), col('fy'), col('fz')])
    # only judge the lag once the follower has had time to reach the square part
    sq_t = [r['t'] for r in follow if r['lphase'] in ('square', 'finish')]
    t_from = (sq_t[0] + args.tau + 1.0) if sq_t else None
    lag = estimate_lag(Lt, Lp, t, Fp, t_from)

    below = float(np.mean(dist < args.dmin - SEP_TOL_M) * 100)
    s = dict(n=len(follow), duration=float(t[-1] - t[0]), rate=rate, max_gap=max_gap,
             min_sep=float(dist.min()), mean_sep=float(dist.mean()),
             pct_below=below, track_rms=track_rms, lag=lag)

    ok_rate = rate >= RATE_MIN_HZ and max_gap <= GAP_MAX_S
    ok_lag = None if lag is None else abs(lag - args.tau) <= LAG_TOL_S
    ok_sep = s['min_sep'] >= args.dmin - SEP_TOL_M
    tag = lambda ok: 'n/a (leader barely moved)' if ok is None else ('PASS' if ok else 'CHECK')

    print("\n" + "=" * 62)
    print("  LEADER-FOLLOWER REPORT" + ("   (DRY RUN: both drones simulated)" if dry else ""))
    print("=" * 62)
    print(f"  leader path : square {args.side} m, {args.laps} lap(s), {args.leader_speed} m/s "
          f"({'CW' if args.cw else 'CCW'})")
    print(f"  follow phase: {s['n']} ticks, {s['duration']:.1f} s")
    print(f"  leader data : {rate:.1f} Hz, longest gap {max_gap:.2f} s"
          f"          -> {tag(ok_rate)}")
    lag_txt = 'not observable' if lag is None else f'{lag:.1f} s'
    print(f"  delay       : estimated {lag_txt} (target {args.tau:.1f} s +/- {LAG_TOL_S})"
          f"   -> {tag(ok_lag)}")
    print(f"  separation  : min {s['min_sep']:.3f} m, mean {s['mean_sep']:.3f} m "
          f"(D_MIN {args.dmin} m), {below:.1f}% of time below D_MIN-{SEP_TOL_M}"
          f"  -> {tag(ok_sep)}")
    print(f"  tracking    : RMS |follower - target| = {track_rms:.3f} m")
    print("=" * 62)
    return s


def save_csv(rows, path):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(ROW_KEYS)
        for r in rows:
            w.writerow([r[k] for k in ROW_KEYS])
    print(f"  Saved {path}")


def save_plots(rows, args, prefix, sq_path):
    follow = [r for r in rows if r['phase'] == 'follow']
    if len(follow) < 5:
        return
    col = lambda k: np.array([r[k] for r in follow])
    t = col('t') - follow[0]['t']

    # ---- XY ----  (Figure() directly: no GUI window pops up while saving)
    fig = Figure(figsize=(7, 7))
    ax = fig.subplots()
    bx = [SAFE_X[0], SAFE_X[1], SAFE_X[1], SAFE_X[0], SAFE_X[0]]
    by = [SAFE_Y[0], SAFE_Y[0], SAFE_Y[1], SAFE_Y[1], SAFE_Y[0]]
    ax.plot(bx, by, 'r--', lw=1.1, alpha=0.6, label='Safety boundary')
    o = sq_path.outline()
    ax.plot(o[:, 0], o[:, 1], ':', color='#888780', lw=1.0, label='Leader square (planned)')
    ax.plot(col('lx'), col('ly'), color='#3266ad', lw=1.6, label='Leader (flown)')
    ax.plot(col('tx'), col('ty'), '--', color='#3a8a5a', lw=1.1,
            label=f'Follower target (t-{args.tau:.0f}s, keep-out)')
    ax.plot(col('fx'), col('fy'), color='#c94a2a', lw=1.4, label='Follower actual')
    ax.plot(col('lx')[0], col('ly')[0], 'o', color='#3266ad')
    ax.plot(col('fx')[0], col('fy')[0], 'o', color='#c94a2a')
    ax.add_patch(Circle((col('lx')[-1], col('ly')[-1]), args.dmin, fill=False,
                        ls=':', color='#888780', label=f'D_MIN={args.dmin} m (at end)'))
    ax.set_aspect('equal')
    ax.set_xlim(SAFE_X[0] - 0.2, SAFE_X[1] + 0.2)
    ax.set_ylim(SAFE_Y[0] - 0.2, SAFE_Y[1] + 0.2)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_title('Top-down: leader square vs delayed follower'); ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc='upper right')
    fig.savefig(prefix + 'xy.png', dpi=130, bbox_inches='tight')

    # ---- time series ----
    fig = Figure(figsize=(11, 9))
    axs = fig.subplots(3, 1, sharex=True)
    dist = col('dist_xy')
    axs[0].plot(t, dist, color='#3266ad', label='XY distance follower-leader')
    axs[0].axhline(args.dmin, color='#c94a2a', ls='--', label=f'D_MIN = {args.dmin} m')
    axs[0].fill_between(t, 0, np.minimum(dist, args.dmin), color='#c94a2a', alpha=0.15)
    axs[0].set_ylabel('Separation (m)'); axs[0].legend(fontsize=8)
    axs[0].set_title(f"min {dist.min():.3f} m   mean {dist.mean():.3f} m")
    for a, ck, tk, fk, name in ((axs[1], 'lx', 'tx', 'fx', 'X'), (axs[2], 'ly', 'ty', 'fy', 'Y')):
        a.plot(t, col(ck), color='#3266ad', label='Leader')
        a.plot(t, col(tk), '--', color='#3a8a5a', label=f'Target (t-{args.tau:.0f}s)')
        a.plot(t, col(fk), color='#c94a2a', label='Follower')
        a.set_ylabel(f'{name} (m)'); a.legend(fontsize=8, ncol=3, loc='upper right')
    axs[2].set_xlabel('Time since follow start (s)')
    for a in axs:
        a.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(prefix + 'time.png', dpi=130)
    print(f"  Saved {prefix}xy.png, {prefix}time.png")

# =============================================================================
#  LIVE PLOT  (runs in the main thread; the flight runs in a worker thread)
# =============================================================================

class LivePlot:
    def __init__(self, rows, args, sq_path, stop_evt, done_evt):
        self.rows, self.args = rows, args
        self.stop_evt, self.done_evt = stop_evt, done_evt

        self.fig = plt.figure(figsize=(14, 8))
        try:
            self.fig.canvas.manager.set_window_title('Leader-follower live')
        except Exception:
            pass
        gs = self.fig.add_gridspec(3, 2, width_ratios=[1, 1.2])
        self.ax_xy = self.fig.add_subplot(gs[:, 0])
        self.ax_d = self.fig.add_subplot(gs[0, 1])
        self.ax_x = self.fig.add_subplot(gs[1, 1], sharex=self.ax_d)
        self.ax_y = self.fig.add_subplot(gs[2, 1], sharex=self.ax_d)
        plt.setp(self.ax_d.get_xticklabels(), visible=False)
        plt.setp(self.ax_x.get_xticklabels(), visible=False)

        # ---- top-down ----
        a = self.ax_xy
        bx = [SAFE_X[0], SAFE_X[1], SAFE_X[1], SAFE_X[0], SAFE_X[0]]
        by = [SAFE_Y[0], SAFE_Y[0], SAFE_Y[1], SAFE_Y[1], SAFE_Y[0]]
        a.plot(bx, by, 'r--', lw=1.1, alpha=0.6, label='Safety boundary')
        o = sq_path.outline()
        a.plot(o[:, 0], o[:, 1], ':', color='#888780', lw=1.0, label='Leader square (planned)')
        self.ln_l, = a.plot([], [], color='#3266ad', lw=1.6, label='Leader')
        self.ln_t, = a.plot([], [], '--', color='#3a8a5a', lw=1.1,
                            label=f'Follower target (t-{args.tau:.0f}s)')
        self.ln_f, = a.plot([], [], color='#c94a2a', lw=1.4, label='Follower')
        self.pt_l, = a.plot([], [], 'o', color='#3266ad', ms=9)
        self.pt_f, = a.plot([], [], 's', color='#c94a2a', ms=8)
        self.keep = Circle((0, 0), args.dmin, fill=False, ls=':', ec='#888780',
                           label=f'D_MIN = {args.dmin} m')
        a.add_patch(self.keep)
        a.set_aspect('equal')
        a.set_xlim(SAFE_X[0] - 0.2, SAFE_X[1] + 0.2)
        a.set_ylim(SAFE_Y[0] - 0.2, SAFE_Y[1] + 0.2)
        a.set_xlabel('X (m)'); a.set_ylabel('Y (m)')
        a.set_title('Top-down'); a.grid(alpha=0.25)
        a.legend(fontsize=8, loc='lower right')
        self.info = a.text(0.02, 0.98, '', transform=a.transAxes, va='top', fontsize=9,
                           family='monospace',
                           bbox=dict(boxstyle='round', fc='white', ec='#cccccc', alpha=0.85))

        # ---- time series ----
        self.ln_d, = self.ax_d.plot([], [], color='#3266ad', label='XY separation')
        self.ax_d.axhline(args.dmin, color='#c94a2a', ls='--', label=f'D_MIN = {args.dmin} m')
        self.ax_d.set_ylabel('Separation (m)'); self.ax_d.legend(fontsize=8, loc='upper right')
        self.ln_lx, = self.ax_x.plot([], [], color='#3266ad', label='Leader')
        self.ln_tx, = self.ax_x.plot([], [], '--', color='#3a8a5a', label='Target')
        self.ln_fx, = self.ax_x.plot([], [], color='#c94a2a', label='Follower')
        self.ax_x.set_ylabel('X (m)'); self.ax_x.legend(fontsize=8, ncol=3, loc='upper right')
        self.ax_x.set_ylim(SAFE_X[0] - 0.2, SAFE_X[1] + 0.2)
        self.ln_ly, = self.ax_y.plot([], [], color='#3266ad', label='Leader')
        self.ln_ty, = self.ax_y.plot([], [], '--', color='#3a8a5a', label='Target')
        self.ln_fy, = self.ax_y.plot([], [], color='#c94a2a', label='Follower')
        self.ax_y.set_ylabel('Y (m)'); self.ax_y.set_xlabel('Time since follow start (s)')
        self.ax_y.legend(fontsize=8, ncol=3, loc='upper right')
        self.ax_y.set_ylim(SAFE_Y[0] - 0.2, SAFE_Y[1] + 0.2)
        for ax in (self.ax_d, self.ax_x, self.ax_y):
            ax.grid(alpha=0.25)

        self.fig.tight_layout()
        self.fig.canvas.mpl_connect('close_event', self._on_close)
        self.ani = FuncAnimation(self.fig, self.update, interval=200, cache_frame_data=False)

    def _on_close(self, _evt):
        if not self.done_evt.is_set():
            print("\n[Live] window closed -> landing")
            self.stop_evt.set()

    def update(self, _frame):
        snap = list(self.rows)
        if len(snap) >= 2:
            col = lambda k: np.array([r[k] for r in snap])
            t = col('t') - snap[0]['t']
            lx, ly, fx, fy = col('lx'), col('ly'), col('fx'), col('fy')
            tx, ty, dist = col('tx'), col('ty'), col('dist_xy')

            self.ln_l.set_data(lx, ly)
            self.ln_t.set_data(tx, ty)
            self.ln_f.set_data(fx, fy)
            self.pt_l.set_data([lx[-1]], [ly[-1]])
            self.pt_f.set_data([fx[-1]], [fy[-1]])
            self.keep.center = (lx[-1], ly[-1])

            self.ln_d.set_data(t, dist)
            self.ln_lx.set_data(t, lx); self.ln_tx.set_data(t, tx); self.ln_fx.set_data(t, fx)
            self.ln_ly.set_data(t, ly); self.ln_ty.set_data(t, ty); self.ln_fy.set_data(t, fy)
            self.ax_d.set_xlim(0, max(t[-1], 20.0) * 1.02)
            self.ax_d.set_ylim(0, max(1.5, float(np.nanmax(dist)) * 1.15))

            last = snap[-1]
            self.info.set_text(f"t = {t[-1]:5.1f} s\n"
                               f"leader  : {last['lphase']}\n"
                               f"sep     : {dist[-1]:.2f} m  (min {dist.min():.2f})\n"
                               f"age     : {last['leader_age']:.2f} s")
        if self.done_evt.is_set():
            self.fig.suptitle('FLIGHT FINISHED - close this window (final PNGs get saved)',
                              color='#c94a2a')
        return []

# =============================================================================
#  SESSION
# =============================================================================

def prepare(args, L, F, dry):
    """Runs in the main thread BEFORE take-off. Returns initial leader / follower commands."""
    if dry:
        lcmd0 = np.array([SAFE_X[0] + 0.5, SAFE_Y[0] + 0.5, args.leader_z])
        fcmd0 = np.array([SAFE_X[1] - 0.5, SAFE_Y[0] + 0.5, FLY_Z])
        print(f"[Leader]   DRY RUN - virtual leader   starts at ({lcmd0[0]:.2f}, {lcmd0[1]:.2f})")
        print(f"[Follower] DRY RUN - virtual follower starts at ({fcmd0[0]:.2f}, {fcmd0[1]:.2f})")
    else:
        print(f"\n[LPS] Waiting {LPS_SETTLE:.0f} s for the estimators to settle ...")
        time.sleep(LPS_SETTLE)
        lat, fl = L.latest(), F.latest()
        if lat is None:
            raise RuntimeError("No position from the LEADER. Check its LPS deck / radio.")
        if fl is None:
            raise RuntimeError("No position from the FOLLOWER. Check its LPS deck / radio.")
        print(f"[Leader]   at ({lat[1][0]:.2f}, {lat[1][1]:.2f}, {lat[1][2]:.2f})")
        print(f"[Follower] at ({fl[1][0]:.2f}, {fl[1][1]:.2f}, {fl[1][2]:.2f})")
        if not in_zone(lat[1][:2], 0.1):
            raise RuntimeError("Leader spawn is outside / too close to the safe zone. Aborting.")
        if not in_zone(fl[1][:2], 0.1):
            raise RuntimeError("Follower spawn is outside / too close to the safe zone. Aborting.")
        lcmd0 = np.array([lat[1][0], lat[1][1], args.leader_z])
        fcmd0 = np.array([fl[1][0], fl[1][1], FLY_Z])

    input("\nPlace both drones on their spawn points, then press ENTER to "
          + ("start (dry run) ... " if dry else "TAKE OFF ... "))

    if dry:   # fill the leader history with 'sitting at the spawn' so the TAU buffer is ready
        now = time.time()
        for k in range(int((args.tau + 2.0) / DT), -1, -1):
            L.push(now - k * DT, lcmd0)
    while L.span() < args.tau + 1.0:        # buffer must reach back TAU seconds
        time.sleep(0.1)
    return lcmd0, fcmd0


def flight(args, dry, lcf, fcf, L, F, rows, ctrl, fcmd, stop_evt, done_evt):
    """Worker thread: take off both, run the 10 Hz loop, land both."""
    lhc = None if dry else lcf.high_level_commander
    fhc = None if dry else fcf.high_level_commander
    airborne = False
    try:
        if not dry:
            for cf in (lcf, fcf):
                cf.param.set_value('commander.enHighLevel', '1')
            time.sleep(0.3)
            for cf in (lcf, fcf):
                try:
                    cf.supervisor.send_arming_request(True)
                except AttributeError:
                    pass
            time.sleep(0.5)
            print(f"[Both] TAKEOFF -> leader z={args.leader_z} m, follower z={FLY_Z} m")
            lhc.takeoff(args.leader_z, TAKEOFF_DUR)
            fhc.takeoff(FLY_Z, TAKEOFF_DUR)
            airborne = True
            stop_evt.wait(TAKEOFF_DUR + 1.0)

        print(f"[Run] FOLLOWING  tau={args.tau} s  D_MIN={args.dmin} m  "
              f"follower vmax={args.vmax} m/s  leader {args.leader_speed} m/s  "
              f"(Ctrl+C or closing the plot window = land)")
        t_start = time.time()
        t_prev = t_start
        tick = 0
        while not stop_evt.is_set():
            t_loop = time.time()
            dt_real = DT if tick == 0 else min(t_loop - t_prev, 3 * DT)
            t_prev = t_loop

            if t_loop - t_start > args.duration:
                print("[Run] max duration reached -> landing")
                break

            # ---------- leader: read, check, command ----------
            lat = L.latest()
            age = t_loop - lat[0]
            if age > LEADER_LOST_S:
                print(f"[Leader] data lost for {age:.1f} s -> landing both")
                break
            lp = lat[1]
            if not dry:
                if not (SAFE_X[0] - ZONE_BREACH <= lp[0] <= SAFE_X[1] + ZONE_BREACH and
                        SAFE_Y[0] - ZONE_BREACH <= lp[1] <= SAFE_Y[1] + ZONE_BREACH):
                    print(f"[Leader] left the safe zone at ({lp[0]:.2f}, {lp[1]:.2f}) -> landing both")
                    break
                dev = float(np.hypot(lp[0] - ctrl.cmd[0], lp[1] - ctrl.cmd[1]))
                if dev > LEADER_DEV_MAX:
                    print(f"[Leader] {dev:.2f} m away from its own command -> landing both")
                    break

            lcmd, ldone = ctrl.update(t_loop, dt_real)
            if dry:
                L.push(t_loop, lcmd)            # virtual leader perfectly tracks its command
                lat = L.latest()
                age = t_loop - lat[0]
            else:
                lhc.go_to(lcmd[0], lcmd[1], lcmd[2], yaw=0.0, duration_s=DT, relative=False)

            # ---------- follower (logic unchanged) ----------
            if dry:
                f_act = fcmd.copy()
            else:
                f_act = F.latest()[1]
                if not (SAFE_X[0] - ZONE_BREACH <= f_act[0] <= SAFE_X[1] + ZONE_BREACH and
                        SAFE_Y[0] - ZONE_BREACH <= f_act[1] <= SAFE_Y[1] + ZONE_BREACH):
                    print(f"[Follower] left the safe zone at ({f_act[0]:.2f}, {f_act[1]:.2f}) "
                          f"-> landing both")
                    break

            ref = L.at(t_loop - args.tau)
            tgt = None
            if age <= STALE_HOLD_S and ref is not None:
                z = FLY_Z if not args.follow_z else float(np.clip(ref[2], Z_MIN, Z_MAX))
                tgt = compute_target(lat[1], ref, f_act[:2], args.dmin, z)
                if tgt is not None:
                    fcmd = slew(fcmd, tgt, lat[1], args.dmin, DT, args.vmax, V_ESCAPE)
            # (tgt is None -> hold the last command)

            if not dry:
                fhc.go_to(fcmd[0], fcmd[1], fcmd[2], yaw=0.0, duration_s=DT, relative=False)

            rows.append(make_row(t_loop, ctrl.phase, lat[1], age, ref, tgt, fcmd, f_act, lcmd))

            if tick % 10 == 0:
                r = rows[-1]
                print(f"  t={t_loop - t_start:5.1f}s  [{ctrl.lap_text()}]  "
                      f"leader=({r['lx']:.2f},{r['ly']:.2f})  "
                      f"target=({r['tx']:.2f},{r['ty']:.2f})  follower=({r['fx']:.2f},{r['fy']:.2f})  "
                      f"sep={r['dist_xy']:.2f} m  leader_age={age:.2f}s"
                      + ("  [HOLD]" if tgt is None else ""))
            tick += 1

            if ldone:
                print("[Run] leader finished and follower caught up -> landing")
                break
            stop_evt.wait(max(0.0, DT - (time.time() - t_loop)))

        if stop_evt.is_set():
            print("[Run] stop requested -> landing")

    except Exception:
        print("\n[Run] ERROR in flight loop -> landing")
        traceback.print_exc()
    finally:
        if airborne:
            print("[Both] LANDING")
            for name, h in (('Leader', lhc), ('Follower', fhc)):
                try:
                    h.land(0.0, duration_s=LAND_DUR)
                except Exception as e:
                    print(f"[{name}] landing command error: {e}")
            time.sleep(LAND_DUR + 0.5)
            for name, h in (('Leader', lhc), ('Follower', fhc)):
                try:
                    h.stop()
                except Exception as e:
                    print(f"[{name}] stop command error: {e}")
        done_evt.set()

# =============================================================================
#  MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description='Leader flies a square, delayed follower.')
    ap.add_argument('--leader-uri', default=LEADER_URI)
    ap.add_argument('--follower-uri', default=FOLLOWER_URI)
    ap.add_argument('--tau', type=float, default=TAU, help='time buffer in seconds')
    ap.add_argument('--dmin', type=float, default=D_MIN, help='minimum XY distance to leader (m)')
    ap.add_argument('--vmax', type=float, default=V_MAX, help='follower speed limit (m/s)')
    ap.add_argument('--duration', type=float, default=MAX_FLIGHT_S, help='max follow time (s)')
    ap.add_argument('--follow-z', action='store_true', help='copy the leader altitude (clipped)')
    # leader square
    ap.add_argument('--laps', type=int, default=SQ_LAPS, help='number of square laps')
    ap.add_argument('--side', type=float, default=SQ_SIDE, help='square side length (m)')
    ap.add_argument('--center-x', type=float, default=SQ_CENTER[0])
    ap.add_argument('--center-y', type=float, default=SQ_CENTER[1])
    ap.add_argument('--leader-speed', type=float, default=SQ_SPEED, help='leader speed (m/s)')
    ap.add_argument('--leader-z', type=float, default=LEADER_Z, help='leader altitude (m)')
    ap.add_argument('--cw', action='store_true', help='fly the square clockwise (default CCW)')
    # modes
    ap.add_argument('--dry-run', action='store_true',
                    help='no radios, no motors: leader AND follower are simulated')
    ap.add_argument('--no-live', action='store_true', help='disable the live plot window')
    ap.add_argument('--out', default='./lf_runs')
    args = ap.parse_args()
    dry = args.dry_run

    if not dry:
        if not CF_AVAILABLE:
            raise SystemExit("cflib not found.  pip install cflib")
        cflib.crtp.init_drivers()

    path = SquarePath(args.center_x, args.center_y, args.side, ccw=not args.cw)
    if not path.fits(SQ_MARGIN):
        raise SystemExit(f"The square does not fit inside the safe zone with a {SQ_MARGIN} m margin "
                         f"(corners: {path.pts.tolist()}). Change --center-x/--center-y/--side.")
    if args.laps < 1:
        raise SystemExit("--laps must be >= 1")
    if args.leader_speed > args.vmax:
        print(f"[warn] leader speed {args.leader_speed} m/s > follower vmax {args.vmax} m/s: "
              f"the follower cannot keep up with the delayed path.")

    live = LIVE_OK and not args.no_live
    if not live:
        print("[info] live plot disabled "
              f"({'--no-live' if args.no_live else 'no GUI matplotlib backend'}); "
              "final PNGs are still saved.")

    os.makedirs(args.out, exist_ok=True)
    prefix = os.path.join(args.out, f"lf_{time.strftime('%Y%m%d_%H%M%S')}_")

    L, F, rows = PosStream(), PosStream(), []
    stop_evt, done_evt = threading.Event(), threading.Event()
    try:
        with ExitStack() as stack:
            lcf = fcf = None
            blocks = []
            if not dry:
                lscf = stack.enter_context(
                    SyncCrazyflie(args.leader_uri, cf=Crazyflie(rw_cache='./cache_leader')))
                lcf = lscf.cf
                blocks.append(start_pos_logging(lcf, 'LeaderPos', L))
                fscf = stack.enter_context(
                    SyncCrazyflie(args.follower_uri, cf=Crazyflie(rw_cache='./cache_follower')))
                fcf = fscf.cf
                blocks.append(start_pos_logging(fcf, 'FollowerPos', F))

            lcmd0, fcmd0 = prepare(args, L, F, dry)
            ctrl = LeaderSquare(path, args.leader_z, args.leader_speed, args.laps, args.tau, lcmd0)

            worker = threading.Thread(
                target=flight, daemon=True,
                args=(args, dry, lcf, fcf, L, F, rows, ctrl, fcmd0, stop_evt, done_evt))
            worker.start()

            try:
                if live:
                    lp = LivePlot(rows, args, path, stop_evt, done_evt)   # keep a reference
                    plt.show()                                           # blocks until closed
                while worker.is_alive():
                    worker.join(0.2)
            except KeyboardInterrupt:
                print("\n[Run] Ctrl+C -> landing")
                stop_evt.set()
                while worker.is_alive():
                    try:
                        worker.join(0.2)
                    except KeyboardInterrupt:
                        print("[Run] landing in progress, please wait ...")

            for lc in blocks:
                try:
                    lc.stop()
                except Exception:
                    pass
    except KeyboardInterrupt:
        print("\n[Run] aborted before take-off.")
    finally:
        if rows:
            save_csv(rows, prefix + 'log.csv')
            analyse(rows, L, args, dry)
            save_plots(rows, args, prefix, path)


if __name__ == '__main__':
    main()
