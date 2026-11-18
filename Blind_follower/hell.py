#!/usr/bin/env python3
"""
leader_follower_square.py - leader flies a SQUARE, follower replays it TAU seconds later
using TG-DPAC (Time-to-Go Delayed Pursuit with Arrival Confirmation).

LEADER   : takes off, flies to the first corner of a square, then flies the square
           (default 2 laps) at LEADER_SPEED. Its LPS position (kalman.stateX/Y/Z, 10 Hz)
           and world-frame velocity (stateEstimate.vx/vy/vz) are read and timestamped
           on the PC.
FOLLOWER : takes off, then every 100 ms runs the TG-DPAC pipeline:

  Stage 1  delayed reference + safety shaping
             D(t)   = p_leader(t - TAU)                       (linear interpolation)
             keep-out: if |D - p_leader(t)|_xy < D_MIN -> push D out to D_MIN
             geofence: clip to the safe zone (+ candidate-point fallback)   -> D''
  Stage 2  time-to-go velocity law
             v_ref  = v_leader(t - TAU)                       (feed-forward)
             v_des  = v_ref + (D'' - p_F) / T_g               (error correction)
  Stage 3  acceleration + speed limits
             dv     = clip(v_des - v_prev, -a_max*dt, +a_max*dt)
             |v| <= V_MAX  (V_ESCAPE while inside the keep-out circle)
  Stage 4  anti-runaway setpoint
             the commanded position stays within R_RUNAWAY (0.3 m) of the MEASURED
             follower position, then it is sent with go_to()
  Stage 5  arrival confirmation
             PASS at every square corner (outer 0.30 m / inner 0.20 m, >=2 samples,
             time gate)  and  STOP at the final hover (0.15 m, 0.10 m/s, 0.3 s dwell,
             APPROACH -> IN_ZONE -> CONFIRMED, LATE flag).

  Evaluation (never fed back into the controller): continuous timing monitor
  (closest-path time), position / velocity / formation (long, lat, vert) errors,
  target-tracking error, checkpoint timing error, separation metrics, and the
  weighted objective  J = Jp + Jv + Je + Ju + Js,  reward r = -J,  return G.

Use --baseline to fly the OLD speed-limited slew() follower for an A/B comparison.

REFERENCE MODE (--ref-mode):
  time  (default) D(t) = p_leader(t - TAU): the reference moves with the clock, so a
                  lagging follower SKIPS the waypoints it has not reached yet.
  fifo            leader waypoints go into a FIFO queue; the reference may only advance
                  past a waypoint once the follower has actually REACHED it (within
                  WP_RADIUS). Nothing is skipped; when the follower keeps up it behaves
                  like 'time', when it lags it walks through x_6, x_7, ... in order.
Every run appends one line to <out>/runs_summary.csv (mean J, components, path coverage,
...) so two approaches can be compared with numbers.

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
    --tg 0.8 --tg-final 0.3 --amax 1.5 --runaway 0.3
    --stale-hold 0.3 --leader-lost 2.0   --baseline

Ctrl+C at any time -> both drones land.
"""

import argparse
import csv
import os
import sys
import threading
import time
import traceback
from collections import deque
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

LEADER_URI   = 'radio://0/80/2M/E7E7E7E701'     # Drone 7: flies the square
FOLLOWER_URI = 'radio://0/80/2M/E7E7E7E711'     # Drone 2: flies the delayed copy

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
MAX_FLIGHT_S = 120.0       # follow phase is cut off after this long

# ---- stale-state policy (TG-DPAC document values) ---------------------------
#   data older than STALE_HOLD_S  -> follower decelerates smoothly and holds
#   data older than LEADER_LOST_S -> land both   (applies to leader AND follower data)
STALE_HOLD_S  = 0.3
LEADER_LOST_S = 2.0

# ---- TG-DPAC controller parameters ------------------------------------------
TG_CRUISE  = 0.8           # s   time-to-go while tracking the moving reference
TG_FINAL   = 0.3           # s   time-to-go during the final approach (STOP)
A_MAX      = 1.5           # m/s^2 acceleration limit per axis   (tunable)
R_RUNAWAY  = 0.3           # m   commanded setpoint stays within this of the measured follower
VEL_WIN_S  = 0.25          # s   half-window for the fallback velocity estimate (position slope)

# ---- arrival confirmation ----------------------------------------------------
PASS_R_OUT, PASS_R_IN = 0.30, 0.20   # m   PASS outer / inner radius
PASS_MIN_SAMPLES = 2                 #     samples needed inside the outer radius
PASS_GATE_S = 2.0                    # s   PASS only judged within +/- this of the expected time
STOP_POS_TOL = 0.15                  # m
STOP_VEL_TOL = 0.10                  # m/s
STOP_DWELL_S = 0.3                   # s
STOP_SLACK_S = 1.5                   # s   STOP deadline = (leader finish + TAU) + this -> else LATE

# ---- continuous timing monitor -----------------------------------------------
TIMING_WIN_S = 1.5         # s   search window around t - TAU for the closest path point
TIMING_GRID_S = 0.02       # s   search resolution
TIMING_MIN_EXTENT = 0.3    # m   leader path must move this much in the window (else unobservable)
HEADING_MIN_SPEED = 0.05   # m/s minimum delayed speed to define the leader frame heading
FF_SHAPED_TOL = 0.03       # m   if keep-out/geofence moved the target farther than this from the
                           #     delayed reference, the feed-forward velocity is switched off

# ---- FIFO waypoint mode (--ref-mode fifo) and path-coverage metric ------------
WP_SPACING = 0.10          # m   leader samples closer than this to the last queued waypoint are not queued
WP_RADIUS  = 0.12          # m   follower must come this close to a waypoint to 'reach' it
COV_RADIUS = 0.15          # m   coverage metric: waypoint counts as visited within this distance
FIFO_DRAIN_WAIT_S = 3.0    # s   after the queue is empty wait at most this long for STOP confirmation
FIFO_STALL_S = 8.0         # s   watchdog: a waypoint stuck at the head this long is force-popped (counted)
HIST_N = 600               # samples kept for interpolation (60 s at 10 Hz)

# ---- weighted objective (evaluation only; tunable weights) -------------------
W_P, W_V, W_E, W_U, W_S = 1.0, 0.5, 0.5, 0.05, 100.0
GAMMA = 0.99               # discount factor for G = sum gamma^t r_t

# Pass criteria from the progress document
RATE_MIN_HZ, GAP_MAX_S = 9.0, 0.5
LAG_TOL_S = 0.3
SEP_TOL_M = 0.05           # tolerance below D_MIN when checking "separation >= D_MIN"

ROW_KEYS = ['t', 'phase', 'lphase', 'lx', 'ly', 'lz', 'leader_age', 'follower_age',
            'lcx', 'lcy', 'lcz',
            'rx', 'ry', 'rz', 'tx', 'ty', 'tz', 'cx', 'cy', 'cz', 'fx', 'fy', 'fz',
            'dist_xy', 'dist_ref', 'ex', 'ey', 'ez',
            'vrx', 'vry', 'vrz',            # delayed leader velocity (feed-forward)
            'vfx', 'vfy', 'vfz',            # follower measured velocity
            'vcx', 'vcy', 'vcz',            # commanded velocity
            'acc', 'runaway', 'tg',
            'erx', 'ery', 'erz',            # follower - ORIGINAL delayed reference
            'e_long', 'e_lat', 'e_vert',    # same error in the leader frame
            'tau_hat', 'tau_err', 'path_err',
            'Jp', 'Jv', 'Je', 'Ju', 'Js', 'J', 'reward', 'arrival']

# =============================================================================
#  TIMESTAMPED POSITION / VELOCITY BUFFER
# =============================================================================

def _slope(ts, ps, t0, t1):
    """Least-squares slope (velocity) of ps(ts) inside [t0, t1]. None if < 3 samples."""
    m = (ts >= t0) & (ts <= t1)
    if m.sum() < 3:
        return None
    tt = ts[m] - ts[m].mean()
    den = float(np.dot(tt, tt))
    if den < 1e-9:
        return None
    return (tt[:, None] * (ps[m] - ps[m].mean(axis=0))).sum(axis=0) / den


class PosStream:
    """Thread-safe history of (PC time, xyz) and optionally (PC time, vxyz) for one drone."""

    def __init__(self):
        self._lock = threading.Lock()
        self.t, self.p = [], []
        self.tv, self.v = [], []

    def push(self, t, p, v=None):
        with self._lock:
            self.t.append(float(t))
            self.p.append((float(p[0]), float(p[1]), float(p[2])))
            if v is not None:
                self.tv.append(float(t))
                self.v.append((float(v[0]), float(v[1]), float(v[2])))

    def push_vel(self, t, v):
        with self._lock:
            self.tv.append(float(t))
            self.v.append((float(v[0]), float(v[1]), float(v[2])))

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
            ts = np.array(self.t[-HIST_N:])
            ps = np.array(self.p[-HIST_N:])
        if tq < ts[0]:
            return None
        return np.array([np.interp(tq, ts, ps[:, k]) for k in range(3)])

    def at_many(self, tqs):
        """Vectorised position interpolation at the (sorted) times tqs. None if the
        history does not reach back to tqs[0]."""
        with self._lock:
            if len(self.t) < 2:
                return None
            ts = np.array(self.t[-HIST_N:])
            ps = np.array(self.p[-HIST_N:])
        if tqs[0] < ts[0]:
            return None
        return np.column_stack([np.interp(tqs, ts, ps[:, k]) for k in range(3)])

    def samples_after(self, t0):
        """All (times, positions) with time > t0."""
        with self._lock:
            ts = np.array(self.t)
            ps = np.array(self.p)
        if len(ts) == 0:
            return ts, ps
        m = ts > t0
        return ts[m], ps[m]

    def vel_at(self, tq):
        """Velocity at PC time tq: the LOGGED velocity (interpolated) when available,
        otherwise the least-squares slope of the position history around tq."""
        with self._lock:
            have_v = len(self.tv) >= 2
            if have_v:
                tv = np.array(self.tv[-HIST_N:])
                vv = np.array(self.v[-HIST_N:])
            tp = np.array(self.t[-HIST_N:])
            pp = np.array(self.p[-HIST_N:])
        if have_v and tq >= tv[0]:
            return np.array([np.interp(tq, tv, vv[:, k]) for k in range(3)])
        s = _slope(tp, pp, tq - VEL_WIN_S, tq + VEL_WIN_S) if len(tp) >= 3 else None
        return np.zeros(3) if s is None else s

    def vel_latest(self):
        """Most recent velocity (logged if available, else slope over the last 0.4 s)."""
        with self._lock:
            if self.v:
                return np.array(self.v[-1])
            tp = np.array(self.t[-10:])
            pp = np.array(self.p[-10:])
        if len(tp) < 3:
            return np.zeros(3)
        s = _slope(tp, pp, tp[-1] - 0.4, tp[-1])
        return np.zeros(3) if s is None else s

    def arrays(self):
        with self._lock:
            return np.array(self.t), np.array(self.p)


def start_pos_logging(cf, name, stream):
    """Start the 10 Hz position log (and, if the firmware offers it, a velocity log).
    Returns the list of started LogConfigs."""
    started = []

    lc = LogConfig(name=name, period_in_ms=100)
    for v in ('kalman.stateX', 'kalman.stateY', 'kalman.stateZ'):
        lc.add_variable(v, 'float')

    def _cb(ts, data, _lc):
        stream.push(time.time(),
                    (data['kalman.stateX'], data['kalman.stateY'], data['kalman.stateZ']))

    cf.log.add_config(lc)
    lc.data_received_cb.add_callback(_cb)
    lc.start()
    started.append(lc)

    # world-frame velocity from the on-board estimator (feed-forward / STOP speed).
    # If this block cannot be created the code falls back to a position-slope estimate.
    try:
        lv = LogConfig(name=name + 'Vel', period_in_ms=100)
        for v in ('stateEstimate.vx', 'stateEstimate.vy', 'stateEstimate.vz'):
            lv.add_variable(v, 'float')

        def _cbv(ts, data, _lc):
            stream.push_vel(time.time(), (data['stateEstimate.vx'],
                                          data['stateEstimate.vy'],
                                          data['stateEstimate.vz']))

        lv.error_cb.add_callback(
            lambda lc_, msg: print(f"[{name}] velocity log error: {msg} (using position slope)"))
        cf.log.add_config(lv)
        lv.data_received_cb.add_callback(_cbv)
        lv.start()
        started.append(lv)
    except Exception as e:
        print(f"[{name}] velocity log unavailable ({e}) -> velocity from position slope")
    return started

# =============================================================================
#  GEOMETRY: reference -> target     (Stage 1: keep-out + geofence)
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
    """OLD baseline: move the command toward the target by at most v*dt
    (speed-limited position update; used only with --baseline)."""
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
#  FIFO WAYPOINT REFERENCE   (--ref-mode fifo)
# =============================================================================

class WaypointQueue:
    """FIFO of leader waypoints (t_i, p_i), thinned to WP_SPACING.

    The 'time' reference is D(t) = p_L(t - TAU). Here the reference time is capped at the
    time of the first waypoint the follower has NOT reached yet:

        t_ref = t - TAU                   if the head waypoint is not due yet (follower keeps up)
        t_ref = t_head                    if the head is due but not reached (follower lags)

    A due waypoint is popped only when the (keep-out shaped) waypoint is within WP_RADIUS of
    the measured follower, so x_6, x_7, ... are all visited in order before x_11 is used.
    """

    def __init__(self, spacing, radius):
        self.q = deque()
        self.spacing, self.radius = spacing, radius
        self.fed_t = -np.inf
        self.last = None
        self.n_pushed = self.n_popped = self.n_forced = 0
        self._head_id = None
        self._head_since = None

    def feed(self, L):
        """Queue every new leader sample that is >= spacing away from the last queued one."""
        ts, ps = L.samples_after(self.fed_t)
        for t, p in zip(ts, ps):
            self.fed_t = float(t)
            if self.last is None or np.linalg.norm(p[:2] - self.last[:2]) >= self.spacing:
                self.q.append((float(t), np.array(p, dtype=float)))
                self.last = np.array(p, dtype=float)
                self.n_pushed += 1

    def _reached(self, leader_now, f_xy, d_min):
        """Head waypoint visited?  Yes if the follower is within `radius` of it, OR it has
        already passed it along the path (projection on the segment head->next is >= 0)
        while staying within `radius` of that segment (handles corner cutting / overshoot)."""
        a = compute_target(leader_now, self.q[0][1], f_xy, d_min, FLY_Z)
        if a is None:
            return False
        d = f_xy - a[:2]
        if np.linalg.norm(d) <= self.radius:
            return True
        if len(self.q) > 1:
            b = compute_target(leader_now, self.q[1][1], f_xy, d_min, FLY_Z)
            if b is not None:
                seg = b[:2] - a[:2]
                l2 = float(seg @ seg)
                if l2 > 1e-9:
                    u = float(d @ seg) / l2
                    if u >= 0.0 and np.linalg.norm(f_xy - (a[:2] + min(u, 1.0) * seg)) <= self.radius:
                        return True
        return False

    def reference(self, now, tau, leader_now, f_xy, d_min):
        """Returns (t_ref, ref_position or None). None -> caller interpolates L.at(t_ref)."""
        t_due = now - tau
        while self.q:
            t_i, p_i = self.q[0]
            if t_i > t_due:                      # head not due yet: follower keeps up
                return t_due, None
            if self._head_id != t_i:             # new head: start the watchdog
                self._head_id, self._head_since = t_i, now
            if self._reached(leader_now, f_xy, d_min):
                self.q.popleft()                 # reached -> next waypoint
                self.n_popped += 1
                continue
            if now - self._head_since > FIFO_STALL_S:      # watchdog: never deadlock
                self.q.popleft()
                self.n_popped += 1
                self.n_forced += 1
                print(f"[FIFO] waypoint at t={t_i:.1f} not reached for {FIFO_STALL_S:.0f} s "
                      f"-> forced skip #{self.n_forced}")
                continue
            return t_i, p_i.copy()               # due but not reached: do NOT skip it
        return t_due, None

# =============================================================================
#  TG-DPAC CONTROLLER   (Stages 2, 3, 4)
# =============================================================================

class TGDPAC:
    """Time-to-go velocity law + acceleration/speed limiter + anti-runaway setpoint.

    State: self.cmd (commanded position, sent via go_to) and self.v (commanded velocity).
    """

    def __init__(self, cmd0, args):
        self.cmd = np.array(cmd0, dtype=float)
        self.v = np.zeros(3)
        self.amax, self.vmax, self.vesc = args.amax, args.vmax, V_ESCAPE
        self.R, self.dmin, self.follow_z = args.runaway, args.dmin, args.follow_z

    def step(self, p_f, tgt, v_ref, leader_xy, tg, dt, v_lead=None):
        """One control tick.  p_f = MEASURED follower position, tgt = D'' (or None -> hold).
        Returns dict(acc=vector, runaway=0/1)."""
        # ---- Stage 2: v_des = v_ref + (D'' - p_F) / T_g -----------------------------
        if tgt is None:                       # stale data / no valid target:
            v_des = np.zeros(3)               #   decelerate smoothly (through the limiter)
        else:
            v_des = v_ref + (tgt - p_f) / tg
            if not self.follow_z:
                v_des[2] = 0.0

        # keep-out guard (braking curve): the speed at which the follower may CLOSE IN on the
        # leader is limited to sqrt(2 * 0.8*a_max * (distance - D_MIN)), so the acceleration
        # limiter can always stop it at the circle; zero inside the circle. The leader's own
        # velocity is taken into account (closing speed is relative).
        rel = p_f[:2] - leader_xy
        dist = float(np.linalg.norm(rel))
        inside = dist < self.dmin
        if dist > 1e-6:
            n = rel / dist
            vl = np.zeros(2) if v_lead is None else np.asarray(v_lead)[:2]
            v_allow = float(np.sqrt(2.0 * 0.8 * self.amax * max(0.0, dist - self.dmin)))
            vn = float((v_des[:2] - vl) @ n)
            if vn < -v_allow:
                v_des[:2] = v_des[:2] + (-v_allow - vn) * n

        # ---- Stage 3: acceleration limiter, then speed limiter ----------------------
        dv = np.clip(v_des - self.v, -self.amax * dt, self.amax * dt)
        v = self.v + dv
        lim = self.vesc if inside else self.vmax
        sp = float(np.linalg.norm(v))
        if sp > lim:
            v = v * (lim / sp)

        # integrate the commanded position, keep it inside the geofence
        new = self.cmd + v * dt
        lo = np.array([SAFE_X[0] + GEO_MARGIN, SAFE_Y[0] + GEO_MARGIN])
        hi = np.array([SAFE_X[1] - GEO_MARGIN, SAFE_Y[1] - GEO_MARGIN])
        new[:2] = np.clip(new[:2], lo, hi)
        if self.follow_z:
            new[2] = float(np.clip(new[2], Z_MIN, Z_MAX))
        else:
            new[2] = tgt[2] if tgt is not None else self.cmd[2]

        # ---- Stage 4: anti-runaway setpoint (stay within R of the measured follower) -
        off = new - p_f
        if not self.follow_z:
            off[2] = 0.0
        n_off = float(np.linalg.norm(off))
        runaway = 0
        if n_off > self.R:
            new = new - off + off * (self.R / n_off)
            runaway = 1

        v_eff = (new - self.cmd) / dt         # what the setpoint really does (anti-windup)
        acc = (v_eff - self.v) / dt
        self.v, self.cmd = v_eff, new
        return dict(acc=acc, runaway=runaway)

    def step_baseline(self, tgt, leader_xy, dt):
        """OLD behaviour (--baseline): speed-limited slew toward the target."""
        old = self.cmd.copy()
        if tgt is not None:
            self.cmd = slew(self.cmd, tgt, np.array([leader_xy[0], leader_xy[1], 0.0]),
                            self.dmin, dt, self.vmax, self.vesc)
        v_eff = (self.cmd - old) / dt
        acc = (v_eff - self.v) / dt
        self.v = v_eff
        return dict(acc=acc, runaway=0)

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
    """Leader command generator:  goto_start -> square (N laps) -> finish (hover) -> done.
    Records the time at which the leader passes each corner (cp_times[k], k = 0 .. 4*laps);
    the follower should pass corner k at cp_times[k] + TAU."""

    def __init__(self, path, z, speed, laps, tau, start_cmd):
        self.path, self.z, self.speed, self.laps = path, z, speed, laps
        self.total = laps * path.perimeter
        self.hold_after = tau + FINISH_EXTRA_S
        self.cmd = np.array(start_cmd, dtype=float)
        self.phase = 'goto_start'
        self.s = 0.0
        self.t_sq0 = None
        self.t_fin = None
        self.cp_times = []
        self.next_cp = 1

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
                self.cp_times.append(now)                       # corner 0
                print("[Leader] at start corner -> flying the SQUARE")
        elif self.phase == 'square':
            ramp = min(1.0, (now - self.t_sq0) / SQ_RAMP_S)
            self.s = min(self.total, self.s + self.speed * ramp * dt)
            p = self.path.point(self.s)
            self.cmd = np.array([p[0], p[1], self.z])
            while (self.next_cp <= 4 * self.laps and
                   self.s >= self.next_cp * self.path.side - 1e-9):
                self.cp_times.append(now)                       # corner next_cp
                self.next_cp += 1
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
#  STAGE 5: ARRIVAL CONFIRMATION  (PASS at corners, STOP at the final hover)
# =============================================================================

PASS_FINAL_STATES = ('CONFIRMED', 'NEAR', 'MISSED')


class PassDetector:
    """Corner checkpoint. The follower is judged only inside a time gate around the
    expected passing time (prevents early confirmation). It tracks the CLOSEST approach
    and needs >= PASS_MIN_SAMPLES samples inside the outer radius.
        CONFIRMED : closest approach <= inner radius
        NEAR      : closest approach inside the outer radius only
        MISSED    : never (enough samples) inside the outer radius
    """

    def __init__(self, idx, pos, t_expect):
        self.idx, self.pos, self.t_expect = idx, np.asarray(pos, float), float(t_expect)
        self.state = 'WAIT'
        self.n_in = 0
        self.d_best = np.inf
        self.t_best = None

    def _finish(self):
        if self.n_in >= PASS_MIN_SAMPLES:
            self.state = 'CONFIRMED' if self.d_best <= PASS_R_IN else 'NEAR'
        else:
            self.state = 'MISSED'

    def update(self, now, f_xy):
        if self.state in PASS_FINAL_STATES:
            return
        if now < self.t_expect - PASS_GATE_S:            # time gate: too early
            return
        if now > self.t_expect + PASS_GATE_S:            # gate closed: decide
            self._finish()
            return
        d = float(np.linalg.norm(np.asarray(f_xy[:2]) - self.pos))
        if d <= PASS_R_OUT:
            self.n_in += 1
            self.state = 'TRACK'
            if d < self.d_best:
                self.d_best, self.t_best = d, now
        elif self.state == 'TRACK':                      # left the outer circle: closest approach known
            self._finish()

    @property
    def timing_err(self):
        return None if self.t_best is None else self.t_best - self.t_expect


class StopDetector:
    """Final hover: IDLE -> APPROACH -> IN_ZONE -> CONFIRMED.
    CONFIRMED needs position error <= 0.15 m AND speed <= 0.10 m/s for 0.3 s (dwell).
    LATE is flagged when the deadline passes before confirmation."""

    def __init__(self, follow_z):
        self.k = 3 if follow_z else 2
        self.state = 'IDLE'
        self.t_arm = self.deadline = self.t_in = self.t_conf = None
        self.late = False
        self.final_err = self.final_speed = float('nan')

    def arm(self, t_arm):
        if self.t_arm is None:
            self.t_arm = float(t_arm)
            self.deadline = self.t_arm + STOP_SLACK_S

    def update(self, now, f_act, f_vel, tgt):
        if (self.t_arm is None or now < self.t_arm or tgt is None
                or self.state == 'CONFIRMED'):
            return
        if self.state == 'IDLE':
            self.state = 'APPROACH'
        err = float(np.linalg.norm((np.asarray(f_act) - np.asarray(tgt))[:self.k]))
        spd = float(np.linalg.norm(np.asarray(f_vel)[:self.k]))
        self.final_err, self.final_speed = err, spd
        if err <= STOP_POS_TOL and spd <= STOP_VEL_TOL:
            if self.state != 'IN_ZONE':
                self.state, self.t_in = 'IN_ZONE', now
            elif now - self.t_in >= STOP_DWELL_S:
                self.state, self.t_conf = 'CONFIRMED', now
        else:
            self.state = 'APPROACH'
        if now > self.deadline:
            self.late = True


class ArrivalMonitor:
    """Owns one PassDetector per square corner (corners 0 .. 4*laps-1) and one StopDetector."""

    def __init__(self, path, args):
        self.path, self.tau = path, args.tau
        self.n_cp = 4 * args.laps
        self.passes = []
        self.stop = StopDetector(args.follow_z)

    def update(self, now, ctrl, f_act, f_vel, tgt):
        n_known = min(len(ctrl.cp_times), self.n_cp)
        while len(self.passes) < n_known:                    # leader has passed corner k
            k = len(self.passes)
            self.passes.append(PassDetector(k, self.path.pts[k % 4], ctrl.cp_times[k] + self.tau))
        for p in self.passes:
            p.update(now, f_act[:2])
        if ctrl.t_fin is not None:                           # final corner -> STOP, not PASS
            self.stop.arm(ctrl.t_fin + self.tau)
        self.stop.update(now, f_act, f_vel, tgt)

    def status_text(self):
        ok = sum(p.state == 'CONFIRMED' for p in self.passes)
        judged = sum(p.state in PASS_FINAL_STATES for p in self.passes)
        s = f"PASS {ok}/{self.n_cp} ok ({judged} judged) | STOP {self.stop.state}"
        return s + (' LATE' if self.stop.late else '')

# =============================================================================
#  CONTINUOUS TIMING MONITOR + OBJECTIVE
# =============================================================================

def timing_monitor(L, f_xy, t_now, tau, t_center=None):
    """Closest-path-time estimate: at which leader time t* was the leader closest (XY) to
    the follower's CURRENT position, searched around t_now - tau.
        tau_hat = t_now - t*      (should stay close to tau)
        path_err = distance follower <-> leader's past path
    tau_hat is NaN when the delay is not observable (leader path barely moves, or the
    minimum sits on the edge of the search window)."""
    tc = (t_now - tau) if t_center is None else t_center   # centre on the reference actually tracked
    ts = np.arange(tc - TIMING_WIN_S, tc + TIMING_WIN_S + 1e-9, TIMING_GRID_S)
    P = L.at_many(ts)
    if P is None:
        return np.nan, np.nan
    d = np.hypot(P[:, 0] - f_xy[0], P[:, 1] - f_xy[1])
    i = int(np.argmin(d))
    path_err = float(d[i])
    extent = float(np.ptp(P[:, 0]) + np.ptp(P[:, 1]))
    if extent < TIMING_MIN_EXTENT or i in (0, len(ts) - 1):
        return np.nan, path_err
    return float(t_now - ts[i]), path_err


def evaluate(f_act, f_vel, ref, v_ref, dist_xy, tau_err, acc, heading, args):
    """Evaluation quantities for one tick (NOT used by the controller).
        e_p = follower - ORIGINAL delayed reference, e_v = v_F - v_ref,
        (long, lat, vert) = e_p in the leader frame (heading of the delayed velocity),
        J = Jp + Jv + Je + Ju + Js,  reward r = -J."""
    mask = np.array([1.0, 1.0, 1.0 if args.follow_z else 0.0])
    nan = float('nan')
    if ref is None:
        e_raw = np.full(3, nan)
        e_p, e_v = np.zeros(3), np.zeros(3)
        e_long = e_lat = e_vert = nan
    else:
        e_raw = f_act - ref
        e_p = e_raw * mask
        e_v = (f_vel - v_ref) * mask
        if heading is None:
            e_long = e_lat = nan
        else:
            e_long = float(e_p[:2] @ heading)
            e_lat = float(-e_p[0] * heading[1] + e_p[1] * heading[0])
        e_vert = float(e_raw[2])
    te = 0.0 if not np.isfinite(tau_err) else float(tau_err)
    Jp = W_P * float(e_p @ e_p)
    Jv = W_V * float(e_v @ e_v)
    Je = W_E * te * te
    Ju = W_U * float(acc @ acc)
    Js = W_S * max(0.0, args.dmin - dist_xy) ** 2
    J = Jp + Jv + Je + Ju + Js
    return dict(erx=float(e_raw[0]), ery=float(e_raw[1]), erz=float(e_raw[2]),
                e_long=e_long, e_lat=e_lat, e_vert=e_vert,
                Jp=Jp, Jv=Jv, Je=Je, Ju=Ju, Js=Js, J=J, reward=-J)

# =============================================================================
#  ROWS / ANALYSIS
# =============================================================================

def make_row(t, lphase, lat, age, ref, tgt, cmd, f, lcmd, **extra):
    nan = np.full(3, np.nan)
    ref_ = nan if ref is None else ref
    tgt_ = nan if tgt is None else tgt
    d = {'t': float(t), 'phase': 'follow', 'lphase': lphase, 'leader_age': float(age)}
    for name, v in (('l', lat), ('lc', lcmd), ('r', ref_), ('t', tgt_), ('c', cmd), ('f', f)):
        d[name + 'x'], d[name + 'y'], d[name + 'z'] = (float(v[0]), float(v[1]), float(v[2]))
    d['dist_xy'] = float(np.hypot(f[0] - lat[0], f[1] - lat[1]))
    d['dist_ref'] = float(np.hypot(f[0] - ref_[0], f[1] - ref_[1]))
    d['ex'], d['ey'], d['ez'] = (float(f[0] - tgt_[0]), float(f[1] - tgt_[1]), float(f[2] - tgt_[2]))
    d.update(extra)
    for k in ROW_KEYS:
        d.setdefault(k, float('nan'))
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


def _rms(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else float('nan')


def mode_name(args):
    return ('slew' if args.baseline else 'tg') + '-' + args.ref_mode


def path_coverage(Lt, Lp, t_f, Fp, args, t_from):
    """Path fidelity, independent of timing:
        coverage  = share of the leader's waypoints (thinned to WP_SPACING) that the follower
                    came within COV_RADIUS of at some moment (skipped waypoints = not covered)
        cross-track = distance of every follower sample (after t_from) to the leader's path.
    Waypoints inside the keep-out circle of the leader's final position are excluded
    (the follower is not allowed there)."""
    m = Lt <= t_f[-1]
    Lp = Lp[m][:, :2]
    wp = [Lp[0]]
    for p in Lp:
        if np.linalg.norm(p - wp[-1]) >= WP_SPACING:
            wp.append(p)
    wp = np.array(wp)
    wp = wp[np.linalg.norm(wp - Lp[-1], axis=1) >= args.dmin - COV_RADIUS]
    F2 = Fp[:, :2]
    d = np.linalg.norm(wp[:, None, :] - F2[None, :, :], axis=2)          # waypoints x samples
    visited = d.min(axis=1) <= COV_RADIUS
    sel = (t_f >= t_from) if t_from is not None else np.ones(len(t_f), bool)
    if sel.sum() > 0:
        xt = np.linalg.norm(F2[sel][:, None, :] - Lp[None, :, :], axis=2).min(axis=1)
        xt_rms, xt_max = float(np.sqrt(np.mean(xt ** 2))), float(xt.max())
    else:
        xt_rms = xt_max = float('nan')
    return dict(n_wp=int(len(wp)), n_visited=int(visited.sum()),
                n_skipped=int((~visited).sum()),
                cov_pct=float(visited.mean() * 100), xt_rms=xt_rms, xt_max=xt_max)


SUMMARY_KEYS = ['time', 'mode', 'dry', 'laps', 'leader_speed', 'vmax', 'tau', 'tg', 'amax',
                'J_mean', 'J_int', 'G', 'Jp', 'Jv', 'Je', 'Ju', 'Js',
                'cov_pct', 'n_skipped', 'n_wp', 'xt_rms', 'xt_max',
                'tau_mean', 'tau_rms_err', 'track_rms', 'vel_rms', 'long_rms', 'lat_rms',
                'min_sep', 'pass_ok', 'pass_total', 'stop_state', 'stop_late', 'duration', 'log']


def append_summary(out_dir, args, s, dry, log_path):
    """One line per run -> <out>/runs_summary.csv (compare approaches with numbers)."""
    path = os.path.join(out_dir, 'runs_summary.csv')
    new = not os.path.exists(path)
    row = dict(s)
    row.update(time=time.strftime('%Y-%m-%d %H:%M:%S'), mode=mode_name(args), dry=int(dry),
               laps=args.laps, leader_speed=args.leader_speed, vmax=args.vmax, tau=args.tau,
               tg=args.tg, amax=args.amax, log=os.path.basename(log_path))
    with open(path, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(SUMMARY_KEYS)
        w.writerow([('%.5g' % row[k]) if isinstance(row.get(k), float) else row.get(k, '')
                    for k in SUMMARY_KEYS])
    print(f"  Appended run to {path}  (mode = {mode_name(args)})")


def analyse(rows, L, args, dry, mon=None):
    follow = [r for r in rows if r['phase'] == 'follow']
    if len(follow) < 5:
        print("\n  [analysis] too few follow samples.")
        return None
    col = lambda k: np.array([r[k] for r in follow], dtype=float)
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

    # ---- TG-DPAC metrics ----
    vr = np.column_stack([col('vrx'), col('vry')])
    vf = np.column_stack([col('vfx'), col('vfy')])
    vel_rms = _rms(np.linalg.norm(vf - vr, axis=1))
    pos_ref_rms = _rms(np.hypot(col('erx'), col('ery')))
    long_rms, lat_rms, vert_rms = _rms(col('e_long')), _rms(col('e_lat')), _rms(col('e_vert'))
    th = col('tau_hat')
    th = th[np.isfinite(th)]
    tau_mean = float(np.mean(th)) if len(th) else float('nan')
    tau_std = float(np.std(th)) if len(th) else float('nan')
    tau_rms_err = _rms(col('tau_err'))
    acc = col('acc')
    acc_max = float(np.nanmax(acc)) if np.any(np.isfinite(acc)) else float('nan')
    run_pct = float(np.nanmean(col('runaway')) * 100)

    J = col('J')
    dtv = np.diff(t, prepend=t[0] - DT)
    J_int = float(np.sum(J * dtv))
    J_mean = J_int / float(t[-1] - t[0] + DT)
    G = float(np.sum((GAMMA ** np.arange(len(J))) * (-J)))
    comp = {k: float(np.mean(col(k))) for k in ('Jp', 'Jv', 'Je', 'Ju', 'Js')}
    cov = path_coverage(Lt, Lp, t, Fp, args, t_from)
    s.update(cov, **comp)
    s.update(vel_rms=vel_rms, pos_ref_rms=pos_ref_rms, long_rms=long_rms, lat_rms=lat_rms,
             vert_rms=vert_rms, tau_mean=tau_mean, tau_std=tau_std, tau_rms_err=tau_rms_err,
             acc_max=acc_max, J_int=J_int, J_mean=J_mean, G=G)

    print("\n" + "=" * 66)
    print("  LEADER-FOLLOWER REPORT"
          + ("   (DRY RUN: both drones simulated)" if dry else "")
          + f"   [{'BASELINE slew' if args.baseline else 'TG-DPAC'} | ref={args.ref_mode}]")
    print("=" * 66)
    print(f"  leader path : square {args.side} m, {args.laps} lap(s), {args.leader_speed} m/s "
          f"({'CW' if args.cw else 'CCW'})")
    print(f"  follow phase: {s['n']} ticks, {s['duration']:.1f} s")
    print(f"  leader data : {rate:.1f} Hz, longest gap {max_gap:.2f} s"
          f"          -> {tag(ok_rate)}")
    lag_txt = 'not observable' if lag is None else f'{lag:.1f} s'
    print(f"  delay (fit) : estimated {lag_txt} (target {args.tau:.1f} s +/- {LAG_TOL_S})"
          f"   -> {tag(ok_lag)}")
    print(f"  delay (live): closest-path tau_hat mean {tau_mean:.2f} s, std {tau_std:.2f} s, "
          f"RMS timing err {tau_rms_err:.2f} s")
    print(f"  separation  : min {s['min_sep']:.3f} m, mean {s['mean_sep']:.3f} m "
          f"(D_MIN {args.dmin} m), {below:.1f}% of time below D_MIN-{SEP_TOL_M}"
          f"  -> {tag(ok_sep)}")
    print(f"  tracking    : RMS |follower - commanded target| = {track_rms:.3f} m")
    print(f"  vs delayed D: RMS position err {pos_ref_rms:.3f} m, RMS velocity err {vel_rms:.3f} m/s")
    print(f"  formation   : RMS long {long_rms:.3f} m, lat {lat_rms:.3f} m, vert {vert_rms:.3f} m")
    print(f"  control     : max |a_cmd| {acc_max:.2f} m/s^2 (limit {args.amax} per axis), "
          f"anti-runaway active {run_pct:.1f}% of ticks")
    print(f"  objective   : mean J {J_mean:.4f}, integrated {J_int:.3f}, "
          f"discounted return G {G:.3f} (gamma {GAMMA})")
    print("                mean components  "
          + "  ".join(f"{k}={v:.4f}" for k, v in comp.items()))

    print(f"  path fidelity: visited {cov['n_visited']}/{cov['n_wp']} leader waypoints "
          f"({cov['cov_pct']:.1f}%), skipped {cov['n_skipped']}; cross-track RMS "
          f"{cov['xt_rms']:.3f} m, max {cov['xt_max']:.3f} m")

    if mon is not None:
        print("  " + "-" * 62)
        print("  ARRIVAL CONFIRMATION")
        n_ok = 0
        for p in mon.passes:
            st = p.state if p.state in PASS_FINAL_STATES else 'UNRESOLVED'
            n_ok += st == 'CONFIRMED'
            if p.t_best is None:
                print(f"    PASS corner {p.idx + 1:2d}/{mon.n_cp}: {st:10s} never inside {PASS_R_OUT} m")
            else:
                print(f"    PASS corner {p.idx + 1:2d}/{mon.n_cp}: {st:10s} closest {p.d_best:.3f} m "
                      f"(in {PASS_R_IN}/{PASS_R_OUT}), timing err {p.timing_err:+.2f} s, "
                      f"{p.n_in} samples")
        missing = mon.n_cp - len(mon.passes)
        if missing > 0:
            print(f"    ({missing} corner(s) never reached by the leader - run ended early)")
        st = mon.stop
        if st.state == 'CONFIRMED':
            txt = 'CONFIRMED' + (' (LATE)' if st.late else '')
        else:
            txt = f"NOT CONFIRMED (state {st.state}" + (', LATE' if st.late else '') + ')'
        print(f"    STOP final hover : {txt}; last pos err {st.final_err:.3f} m "
              f"(tol {STOP_POS_TOL}), speed {st.final_speed:.3f} m/s (tol {STOP_VEL_TOL})")
        s.update(pass_ok=n_ok, pass_total=mon.n_cp, stop_state=st.state, stop_late=st.late)
    print("=" * 66)
    return s


def save_csv(rows, path):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(ROW_KEYS)
        for r in rows:
            w.writerow([r[k] for k in ROW_KEYS])
    print(f"  Saved {path}")


def save_plots(rows, args, prefix, sq_path, mon=None):
    follow = [r for r in rows if r['phase'] == 'follow']
    if len(follow) < 5:
        return
    col = lambda k: np.array([r[k] for r in follow], dtype=float)
    t = col('t') - follow[0]['t']

    # ---- XY ----  (Figure() directly: no GUI window pops up while saving)
    fig = Figure(figsize=(7, 7))
    ax = fig.subplots()
    bx = [SAFE_X[0], SAFE_X[1], SAFE_X[1], SAFE_X[0], SAFE_X[0]]
    by = [SAFE_Y[0], SAFE_Y[0], SAFE_Y[1], SAFE_Y[1], SAFE_Y[0]]
    ax.plot(bx, by, 'r--', lw=1.1, alpha=0.6, label='Safety boundary')
    o = sq_path.outline()
    ax.plot(o[:, 0], o[:, 1], ':', color='#888780', lw=1.0, label='Leader square (planned)')
    for k, c in enumerate(sq_path.pts):          # PASS checkpoint radii
        ax.add_patch(Circle(c, PASS_R_OUT, fill=False, ls='-', lw=0.6, ec='#b0b0b0',
                            label=f'PASS outer {PASS_R_OUT} m' if k == 0 else None))
        ax.add_patch(Circle(c, PASS_R_IN, fill=False, ls='-', lw=0.8, ec='#3a8a5a',
                            label=f'PASS inner {PASS_R_IN} m' if k == 0 else None))
    ax.plot(col('lx'), col('ly'), color='#3266ad', lw=1.6, label='Leader (flown)')
    ax.plot(col('tx'), col('ty'), '--', color='#3a8a5a', lw=1.1,
            label=f'Follower target D\'\' (t-{args.tau:.0f}s, keep-out)')
    ax.plot(col('fx'), col('fy'), color='#c94a2a', lw=1.4, label='Follower actual')
    ax.plot(col('lx')[0], col('ly')[0], 'o', color='#3266ad')
    ax.plot(col('fx')[0], col('fy')[0], 'o', color='#c94a2a')
    ax.add_patch(Circle((col('lx')[-1], col('ly')[-1]), args.dmin, fill=False,
                        ls=':', color='#888780', label=f'D_MIN={args.dmin} m (at end)'))
    ax.set_aspect('equal')
    ax.set_xlim(SAFE_X[0] - 0.2, SAFE_X[1] + 0.2)
    ax.set_ylim(SAFE_Y[0] - 0.2, SAFE_Y[1] + 0.2)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    title = 'Top-down: leader square vs delayed follower'
    if mon is not None:
        title += f"\n{mon.status_text()}"
    ax.set_title(title, fontsize=10); ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc='upper right')
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

    # ---- TG-DPAC internals ----
    fig = Figure(figsize=(11, 11))
    axs = fig.subplots(4, 1, sharex=True)
    axs[0].plot(t, np.hypot(col('vrx'), col('vry')), color='#3266ad', label='|v_ref| (delayed leader)')
    axs[0].plot(t, np.hypot(col('vcx'), col('vcy')), color='#c94a2a', label='|v_cmd| (commanded)')
    axs[0].plot(t, np.hypot(col('vfx'), col('vfy')), color='#3a8a5a', lw=1.0, label='|v_F| (measured)')
    axs[0].axhline(args.vmax, color='k', ls=':', lw=0.8, label=f'V_MAX = {args.vmax} m/s')
    axs[0].set_ylabel('Speed (m/s)'); axs[0].legend(fontsize=8, ncol=4, loc='upper right')
    axs[0].set_title('TG-DPAC: time-to-go law, limiters, timing monitor, objective')
    axs[1].plot(t, col('acc'), color='#c94a2a', label='|a_cmd|')
    axs[1].axhline(args.amax, color='k', ls=':', lw=0.8, label=f'a_max = {args.amax} m/s^2 (per axis)')
    rw = col('runaway')
    axs[1].fill_between(t, 0, rw * max(1.0, np.nanmax(col('acc'))), color='#888780', alpha=0.2,
                        step='post', label='anti-runaway active')
    axs[1].set_ylabel('Accel (m/s^2)'); axs[1].legend(fontsize=8, ncol=3, loc='upper right')
    axs[2].plot(t, col('tau_hat'), '.', ms=3, color='#3266ad', label='tau_hat (closest-path time)')
    axs[2].axhline(args.tau, color='#c94a2a', ls='--', label=f'tau = {args.tau} s')
    axs[2].axhspan(args.tau - LAG_TOL_S, args.tau + LAG_TOL_S, color='#c94a2a', alpha=0.08)
    axs[2].set_ylabel('Delay (s)'); axs[2].legend(fontsize=8, loc='upper right')
    for k, c in (('Jp', '#3266ad'), ('Jv', '#3a8a5a'), ('Je', '#d09a2a'),
                 ('Ju', '#888780'), ('Js', '#c94a2a')):
        axs[3].plot(t, col(k), lw=0.8, color=c, label=k)
    axs[3].plot(t, col('J'), 'k', lw=1.4, label='J total')
    axs[3].set_ylabel('Cost J'); axs[3].legend(fontsize=8, ncol=6, loc='upper right')
    axs[3].set_xlabel('Time since follow start (s)')
    for a in axs:
        a.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(prefix + 'tgdpac.png', dpi=130)
    print(f"  Saved {prefix}xy.png, {prefix}time.png, {prefix}tgdpac.png")

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
                               f"age     : {last['leader_age']:.2f} s\n"
                               f"tau_hat : {last['tau_hat']:.2f} s  (T_g {last['tg']:.1f} s)\n"
                               f"{last['arrival']}")
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
            L.push(now - k * DT, lcmd0, (0.0, 0.0, 0.0))
    while L.span() < args.tau + 1.0:        # buffer must reach back TAU seconds
        time.sleep(0.1)
    return lcmd0, fcmd0


def flight(args, dry, lcf, fcf, L, F, rows, ctrl, fcmd0, mon, stop_evt, done_evt):
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

        mode = f"ref={args.ref_mode} | " + ("BASELINE slew" if args.baseline else f"TG-DPAC (T_g={args.tg}/{args.tg_final} s, a_max={args.amax})")
        print(f"[Run] FOLLOWING  tau={args.tau} s  D_MIN={args.dmin} m  "
              f"follower vmax={args.vmax} m/s  leader {args.leader_speed} m/s  "
              f"controller: {mode}  (Ctrl+C or closing the plot window = land)")
        tg = TGDPAC(fcmd0, args)
        wq = WaypointQueue(WP_SPACING, WP_RADIUS) if args.ref_mode == 'fifo' else None
        t_drained = None
        prev_lcmd = ctrl.cmd.copy()
        heading = None
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
            if age > args.leader_lost:
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
            lcmd = np.array(lcmd, dtype=float)
            if dry:
                lvel = (lcmd - prev_lcmd) / max(dt_real, 1e-3)
                L.push(t_loop, lcmd, lvel)      # virtual leader perfectly tracks its command
                lat = L.latest()
                age = t_loop - lat[0]
            else:
                lhc.go_to(lcmd[0], lcmd[1], lcmd[2], yaw=0.0, duration_s=DT, relative=False)
            prev_lcmd = lcmd.copy()

            # ---------- follower: measured state ----------
            if dry:
                f_act = tg.cmd.copy()
                f_vel = tg.v.copy()
                fage = 0.0
            else:
                fl = F.latest()
                fage = t_loop - fl[0]
                if fage > args.leader_lost:
                    print(f"[Follower] data lost for {fage:.1f} s -> landing both")
                    break
                f_act = fl[1]
                f_vel = F.vel_latest()
                if not (SAFE_X[0] - ZONE_BREACH <= f_act[0] <= SAFE_X[1] + ZONE_BREACH and
                        SAFE_Y[0] - ZONE_BREACH <= f_act[1] <= SAFE_Y[1] + ZONE_BREACH):
                    print(f"[Follower] left the safe zone at ({f_act[0]:.2f}, {f_act[1]:.2f}) "
                          f"-> landing both")
                    break

            # ---------- Stage 1: delayed reference (+ velocity) and safety shaping ----------
            t_ref = t_loop - args.tau
            ref = None
            if wq is not None:                   # FIFO: never skip an unvisited waypoint
                wq.feed(L)
                t_ref, ref = wq.reference(t_loop, args.tau, lat[1], f_act[:2], args.dmin)
            if ref is None:
                ref = L.at(t_ref)
            v_ref = np.zeros(3)
            if ref is not None:
                vr = L.vel_at(t_ref)
                if vr is not None:
                    v_ref = np.array(vr, dtype=float)
            if not args.follow_z:
                v_ref[2] = 0.0
            sp_ref = float(np.linalg.norm(v_ref[:2]))
            if sp_ref > HEADING_MIN_SPEED:
                heading = v_ref[:2] / sp_ref

            tgt = None
            if age <= args.stale_hold and fage <= args.stale_hold and ref is not None:
                z = FLY_Z if not args.follow_z else float(np.clip(ref[2], Z_MIN, Z_MAX))
                tgt = compute_target(lat[1], ref, f_act[:2], args.dmin, z)
            # (tgt is None -> controller decelerates smoothly and holds)

            # feed-forward only while the target really moves like the delayed reference
            v_ff = v_ref
            if tgt is not None and ref is not None and np.linalg.norm(tgt[:2] - ref[:2]) > FF_SHAPED_TOL:
                v_ff = np.zeros(3)

            # ---------- Stages 2-4: time-to-go law, limiters, anti-runaway ----------
            final_phase = ctrl.t_fin is not None and t_ref >= ctrl.t_fin
            tg_now = args.tg_final if final_phase else args.tg
            if args.baseline:
                info = tg.step_baseline(tgt, lat[1][:2], DT)
            else:
                info = tg.step(f_act, tgt, v_ff, lat[1][:2], tg_now, dt_real, L.vel_latest())
            fcmd = tg.cmd

            if not dry:
                fhc.go_to(fcmd[0], fcmd[1], fcmd[2], yaw=0.0, duration_s=DT, relative=False)

            # ---------- Stage 5 + evaluation (never fed back into the controller) ----------
            mon.update(t_loop, ctrl, f_act, f_vel, tgt)
            tau_hat, path_err = timing_monitor(L, f_act[:2], t_loop, args.tau, t_ref)
            tau_err = np.nan if np.isnan(tau_hat) else tau_hat - args.tau
            dist_xy = float(np.hypot(f_act[0] - lat[1][0], f_act[1] - lat[1][1]))
            ev = evaluate(f_act, f_vel, ref, v_ref, dist_xy, tau_err, info['acc'], heading, args)

            rows.append(make_row(
                t_loop, ctrl.phase, lat[1], age, ref, tgt, fcmd, f_act, lcmd,
                follower_age=float(fage),
                vrx=float(v_ref[0]), vry=float(v_ref[1]), vrz=float(v_ref[2]),
                vfx=float(f_vel[0]), vfy=float(f_vel[1]), vfz=float(f_vel[2]),
                vcx=float(tg.v[0]), vcy=float(tg.v[1]), vcz=float(tg.v[2]),
                acc=float(np.linalg.norm(info['acc'])), runaway=int(info['runaway']),
                tg=float(tg_now), tau_hat=float(tau_hat), tau_err=float(tau_err),
                path_err=float(path_err), arrival=mon.status_text(), **ev))

            if tick % 10 == 0:
                r = rows[-1]
                print(f"  t={t_loop - t_start:5.1f}s  [{ctrl.lap_text()}]  "
                      f"leader=({r['lx']:.2f},{r['ly']:.2f})  "
                      f"target=({r['tx']:.2f},{r['ty']:.2f})  follower=({r['fx']:.2f},{r['fy']:.2f})  "
                      f"sep={r['dist_xy']:.2f} m  |v|={np.hypot(r['vcx'], r['vcy']):.2f}  "
                      f"tau_hat={r['tau_hat']:.2f}  leader_age={age:.2f}s  {r['arrival']}"
                      + (f"  queue={len(wq.q)}" if wq is not None else "")
                      + ("  [HOLD]" if tgt is None else ""))
            tick += 1

            if ldone:
                if wq is None:
                    print("[Run] leader finished and follower caught up -> landing")
                    break
                if not wq.q:                     # FIFO: wait until every waypoint was visited
                    t_drained = t_loop if t_drained is None else t_drained
                    if mon.stop.state == 'CONFIRMED' or t_loop - t_drained > FIFO_DRAIN_WAIT_S:
                        print("[Run] leader finished and the waypoint queue is empty -> landing")
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
    ap = argparse.ArgumentParser(description='Leader flies a square, TG-DPAC delayed follower.')
    ap.add_argument('--leader-uri', default=LEADER_URI)
    ap.add_argument('--follower-uri', default=FOLLOWER_URI)
    ap.add_argument('--tau', type=float, default=TAU, help='time buffer in seconds')
    ap.add_argument('--dmin', type=float, default=D_MIN, help='minimum XY distance to leader (m)')
    ap.add_argument('--vmax', type=float, default=V_MAX, help='follower speed limit (m/s)')
    ap.add_argument('--duration', type=float, default=MAX_FLIGHT_S, help='max follow time (s)')
    ap.add_argument('--follow-z', action='store_true', help='copy the leader altitude (clipped)')
    # TG-DPAC controller
    ap.add_argument('--tg', type=float, default=TG_CRUISE, help='time-to-go while tracking (s)')
    ap.add_argument('--tg-final', type=float, default=TG_FINAL, help='time-to-go for the final approach (s)')
    ap.add_argument('--amax', type=float, default=A_MAX, help='acceleration limit per axis (m/s^2)')
    ap.add_argument('--runaway', type=float, default=R_RUNAWAY,
                    help='anti-runaway radius: setpoint stays this close to the measured follower (m)')
    ap.add_argument('--stale-hold', type=float, default=STALE_HOLD_S,
                    help='data older than this -> follower holds (s)')
    ap.add_argument('--leader-lost', type=float, default=LEADER_LOST_S,
                    help='data older than this -> land both (s)')
    ap.add_argument('--ref-mode', choices=('time', 'fifo'), default='time',
                    help="'time': D(t)=p_L(t-tau) (may skip waypoints when lagging); "
                         "'fifo': visit every leader waypoint in order, never skip")
    ap.add_argument('--baseline', action='store_true',
                    help='fly the OLD speed-limited slew() follower (for A/B comparison)')
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
    if args.tg <= 0 or args.tg_final <= 0:
        raise SystemExit("--tg and --tg-final must be > 0")
    if args.amax <= 0 or args.runaway <= 0:
        raise SystemExit("--amax and --runaway must be > 0")
    if args.stale_hold >= args.leader_lost:
        raise SystemExit("--stale-hold must be smaller than --leader-lost")
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
    mon = ArrivalMonitor(path, args)
    stop_evt, done_evt = threading.Event(), threading.Event()
    try:
        with ExitStack() as stack:
            lcf = fcf = None
            blocks = []
            if not dry:
                lscf = stack.enter_context(
                    SyncCrazyflie(args.leader_uri, cf=Crazyflie(rw_cache='./cache_leader')))
                lcf = lscf.cf
                blocks.extend(start_pos_logging(lcf, 'LeaderPos', L))
                fscf = stack.enter_context(
                    SyncCrazyflie(args.follower_uri, cf=Crazyflie(rw_cache='./cache_follower')))
                fcf = fscf.cf
                blocks.extend(start_pos_logging(fcf, 'FollowerPos', F))

            lcmd0, fcmd0 = prepare(args, L, F, dry)
            ctrl = LeaderSquare(path, args.leader_z, args.leader_speed, args.laps, args.tau, lcmd0)

            worker = threading.Thread(
                target=flight, daemon=True,
                args=(args, dry, lcf, fcf, L, F, rows, ctrl, fcmd0, mon, stop_evt, done_evt))
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
            summ = analyse(rows, L, args, dry, mon)
            if summ:
                append_summary(args.out, args, summ, dry, prefix + 'log.csv')
            save_plots(rows, args, prefix, path, mon)


if __name__ == '__main__':
    main()
