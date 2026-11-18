#!/usr/bin/env python3
"""
leader_follower_queue.py - leader flies a SQUARE, follower replays it TAU seconds later.

DIFFERENCE TO THE CORNER VERSION
  Every position sample the leader produces (10 Hz) is pushed into a FIFO queue and
  becomes a CHECKPOINT. There are no square corners any more, no thinning, no PASS/STOP
  detector, and a checkpoint is NEVER skipped or force-popped: the follower must reach
  checkpoint 0, then 1, then 2, ... in order.

HOW A CHECKPOINT WORKS
  leader sample i at time t_i   ->   queue entry i
  expected time at the follower     t_due(i) = t_i + TAU
  the follower REACHES it when it is within R_NEAR of it, or has PASSED it along the path
  (beyond it in the leader's travel direction, inside a narrow corridor)
  timing error = t_reach - t_due        (+ = late, - = early)
      LATE   : error >  +TOL
      EARLY  : error <  -TOL
      TIMELY : otherwise

STATUS IS REPORT-ONLY
  EARLY / TIMELY / LATE is logged and printed, it is NEVER fed back into the controller and
  it NEVER stops or lands a drone. A late follower simply keeps walking through the queue
  (reference = the oldest unreached checkpoint) until it has caught up.
  Only real safety conditions end a flight: Ctrl+C, radio/position data lost, a drone
  leaving the safe zone, the --duration cap, or
  leader moves to PARK_X/PARK_Y/PARK_Z and lands after feeding its final samples; follower drains the queue.

CONTROLLER (unchanged TG-DPAC)
  reference  : p_leader(t - TAU)               while the head checkpoint is not due yet
               head checkpoint position        when the head is due but not reached yet
  Stage 1    : keep-out circle (D_MIN) around the leader + geofence      -> target D''
  Stage 2    : v_des = v_ref + (D'' - p_F) / T_g
  Stage 3    : acceleration limit, speed limit, keep-out braking guard
  Stage 4    : anti-runaway setpoint (R_RUNAWAY around the measured follower) -> go_to()

COST  J = Jp + Jv + Je + Ju + Js      (same weights as the corner version, logging only)
  Jp : |follower - schedule p_L(t-TAU)|^2          Jv : |v_F - v_schedule|^2
  Je : (queue-head lag in seconds)^2               Ju : |a_cmd|^2     Js : keep-out violation
  reward r = -J, running integral J_cum, discounted return G_cum.

OUTPUT in <out>/ (one set per run)
  lfq_<stamp>_log.csv          one row per 100 ms tick: positions, targets, velocities, queue
                               state, tick status, every cost term, running totals
  lfq_<stamp>_checkpoints.csv  one row per checkpoint: when due, when reached, error, status,
                               cost at that moment (+ UNVISITED rows if the run ended early)
  queue_runs_summary.csv       one line per run (counts, timing error, cost)

Dry run (no radios, no motors, both drones simulated):
    python leader_follower_queue.py --dry-run
Real flight:
    python leader_follower_queue.py

Options:
    --laps 2 --side 1.4 --center-x 1.5 --center-y 2.5 --leader-speed 0.4 --cw
    --tau 3.0 --dmin 0.5 --vmax 0.8 --tol 0.3 --duration 180 --follow-z
    --tg 0.8 --tg-final 0.3 --amax 1.5 --runaway 0.3 --stale-hold 0.3 --leader-lost 2.0
"""

import argparse
import csv
import os
import threading
import time
import traceback
from collections import deque
from contextlib import ExitStack

import numpy as np

try:
    import cflib.crtp
    from cflib.crazyflie import Crazyflie
    from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
    from cflib.crazyflie.log import LogConfig
    CF_AVAILABLE = True
except ImportError:
    CF_AVAILABLE = False

# =============================================================================
#  CONSTANTS
# =============================================================================

LEADER_URI   = 'radio://0/80/2M/E7E7E7E701'
FOLLOWER_URI = 'radio://0/80/2M/E7E7E7E711'

SAFE_X = (0.0, 3.0)
SAFE_Y = (1.0, 4.0)
GEO_MARGIN = 0.10
ZONE_BREACH = 0.15

FLY_Z = 0.6
LEADER_Z = 0.6
# ---- leader parking point (EDIT THESE IN THE CODE BEFORE REAL FLIGHT) ----
PARK_X = 0.2       # m; replace with your verified clear parking X coordinate
PARK_Y = 1.2       # m; replace with your verified clear parking Y coordinate
PARK_Z = LEADER_Z  # m; altitude at which the leader moves to the parking point
Z_MIN, Z_MAX = 0.5, 1.2

# ---- leader square ------------------------------------------------------------
SQ_CENTER = (1.5, 2.5)
SQ_SIDE   = 1.4
SQ_SPEED  = 0.4
SQ_LAPS   = 2
SQ_RAMP_S = 1.5
SQ_MARGIN = 0.3
FINISH_EXTRA_S = 2.0

TAU    = 3.0
D_MIN  = 0.5
V_MAX  = 0.8
V_ESCAPE = 1.2

DT           = 0.1
TAKEOFF_DUR  = 2.5
LAND_DUR     = 3.0
LPS_SETTLE   = 10.0
MAX_FLIGHT_S = 180.0       # safety cap only; a late follower is NOT stopped before this

STALE_HOLD_S  = 0.3
LEADER_LOST_S = 2.0

# ---- TG-DPAC ---------------------------------------------------------------------
TG_CRUISE  = 0.8
TG_FINAL   = 0.3
A_MAX      = 1.5
R_RUNAWAY  = 0.3
VEL_WIN_S  = 0.25
HEADING_MIN_SPEED = 0.05
FF_SHAPED_TOL = 0.03
HIST_N = 600

# ---- checkpoint queue ------------------------------------------------------------
TIMELY_TOL_S = 0.3         # |timing error| <= this -> TIMELY
R_NEAR     = 0.05          # m  follower this close to a checkpoint -> reached
R_SHAPED   = 0.10          # m  same, but for a checkpoint that keep-out/geofence moved
SHAPED_TOL = 0.03          # m  target moved farther than this from the raw point = shaped
R_CORR     = 0.15          # m  'passed' test: max sideways distance from the checkpoint
ALONG_MAX  = 0.60          # m  'passed' test: max distance beyond the checkpoint along the path
FEED_TAIL_S = 1.0          # s  leader hover samples after its finish that are still queued
STALL_WARN_S = 8.0         # s  print a warning (ONLY a warning) if one checkpoint sits at the head this long

# ---- cost (logging only) ---------------------------------------------------------
W_P, W_V, W_E, W_U, W_S = 1.0, 0.5, 0.5, 0.05, 100.0
GAMMA = 0.99

TICK_KEYS = [
    't', 't_rel', 'lphase', 'leader_age', 'follower_age',
    'lx', 'ly', 'lz',                 # leader (measured)
    'fx', 'fy', 'fz',                 # follower (measured)
    'sx', 'sy', 'sz',                 # schedule  p_L(t - TAU)
    'rx', 'ry', 'rz',                 # reference used by the controller (head checkpoint or schedule)
    'tx', 'ty', 'tz',                 # target D'' after keep-out + geofence
    'cx', 'cy', 'cz',                 # commanded setpoint
    'dist_xy', 'dist_sched', 'dist_target',
    'vsx', 'vsy', 'vsz',              # schedule velocity
    'vfx', 'vfy', 'vfz',              # follower measured velocity
    'vcx', 'vcy', 'vcz',              # commanded velocity
    'acc', 'runaway', 'tg',
    'q_len', 'head_idx', 'head_t_rel', 'n_reached', 'n_early', 'n_timely', 'n_late',
    'lag_head', 'status',
    'e_long', 'e_lat', 'e_vert',
    'Jp', 'Jv', 'Je', 'Ju', 'Js', 'J', 'reward', 'J_cum', 'G_cum',
]

CP_KEYS = ['idx', 'leader_s', 'x', 'y', 'z', 'target_x', 'target_y', 'shaped',
           'due_s', 'reach_s', 'timing_err', 'status', 'reach_dist', 'queue_len_after',
           'J_tick', 'J_cum']

# =============================================================================
#  TIMESTAMPED POSITION / VELOCITY BUFFER
# =============================================================================

def _slope(ts, ps, t0, t1):
    m = (ts >= t0) & (ts <= t1)
    if m.sum() < 3:
        return None
    tt = ts[m] - ts[m].mean()
    den = float(np.dot(tt, tt))
    if den < 1e-9:
        return None
    return (tt[:, None] * (ps[m] - ps[m].mean(axis=0))).sum(axis=0) / den


class PosStream:
    """Thread-safe history of (PC time, xyz) and (PC time, vxyz) for one drone."""

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
        """Linear interpolation of the position at PC time tq (None if history is too short)."""
        with self._lock:
            if len(self.t) < 2 or tq < self.t[0]:
                return None
            ts = np.array(self.t[-HIST_N:])
            ps = np.array(self.p[-HIST_N:])
        if tq < ts[0]:
            return None
        return np.array([np.interp(tq, ts, ps[:, k]) for k in range(3)])

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
        """Logged velocity (interpolated) if available, else position-slope estimate."""
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
        with self._lock:
            if self.v:
                return np.array(self.v[-1])
            tp = np.array(self.t[-10:])
            pp = np.array(self.p[-10:])
        if len(tp) < 3:
            return np.zeros(3)
        s = _slope(tp, pp, tp[-1] - 0.4, tp[-1])
        return np.zeros(3) if s is None else s


def start_pos_logging(cf, name, stream):
    """10 Hz position log (+ velocity log if the firmware offers it)."""
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
#  GEOMETRY: reference -> target     (keep-out + geofence)
# =============================================================================

def in_zone(xy, margin):
    return (SAFE_X[0] + margin <= xy[0] <= SAFE_X[1] - margin and
            SAFE_Y[0] + margin <= xy[1] <= SAFE_Y[1] - margin)


def compute_target(leader_now, ref, follower_xy, d_min, z):
    """Reference pushed out of the keep-out circle around the leader's CURRENT position,
    then kept inside the geofence. None -> hold."""
    des = ref[:2].copy()
    v = des - leader_now[:2]
    d = np.linalg.norm(v)
    if d < d_min:
        if d < 1e-3:
            v = follower_xy - leader_now[:2]
            d = np.linalg.norm(v)
            if d < 1e-3:
                v, d = np.array([1.0, 0.0]), 1.0
        des = leader_now[:2] + v / d * d_min

    lo = np.array([SAFE_X[0] + GEO_MARGIN, SAFE_Y[0] + GEO_MARGIN])
    hi = np.array([SAFE_X[1] - GEO_MARGIN, SAFE_Y[1] - GEO_MARGIN])
    xy = np.clip(des, lo, hi)

    if np.linalg.norm(xy - leader_now[:2]) < d_min - 0.02:
        ang = np.deg2rad(np.arange(0, 360, 15))
        cand = leader_now[:2] + d_min * np.column_stack([np.cos(ang), np.sin(ang)])
        ok = [c for c in cand if in_zone(c, GEO_MARGIN)]
        if not ok:
            return None
        xy = min(ok, key=lambda c: np.linalg.norm(c - des))
    return np.array([xy[0], xy[1], z])


def step_toward(cmd, tgt, vmax, dt):
    step = tgt - cmd
    n = np.linalg.norm(step)
    lim = vmax * dt
    return tgt.copy() if n <= lim else cmd + step * (lim / n)

# =============================================================================
#  CHECKPOINT QUEUE   (every leader sample, FIFO, never skipped)
# =============================================================================

def classify(err, tol):
    """err = t_reach - t_due.  + = late."""
    if err > tol:
        return 'LATE'
    if err < -tol:
        return 'EARLY'
    return 'TIMELY'


class CheckpointQueue:
    """FIFO of ALL leader samples. Entries leave the queue only when the follower reaches
    them, in order. Nothing is thinned, skipped, or force-popped."""

    def __init__(self, tol, dmin):
        self.q = deque()
        self.tol, self.dmin = tol, dmin
        self.fed_t = None
        self.t0 = 0.0
        self.n_fed = 0
        self.done = []                                   # one record per reached checkpoint
        self.counts = {'EARLY': 0, 'TIMELY': 0, 'LATE': 0}
        self._head_idx = None
        self._head_since = None
        self._warned = None

    def start(self, t_feed_from, t0):
        self.fed_t, self.t0 = float(t_feed_from), float(t0)

    def feed(self, L, t_max=np.inf):
        """Queue EVERY new leader sample (with its velocity)."""
        ts, ps = L.samples_after(self.fed_t)
        for t, p in zip(ts, ps):
            if t > t_max:
                break
            self.fed_t = float(t)
            v = np.array(L.vel_at(float(t)), dtype=float)
            sp = float(np.linalg.norm(v[:2]))
            self.q.append(dict(idx=self.n_fed, t=float(t), p=np.array(p, dtype=float), v=v,
                               hdg=(v[:2] / sp if sp > HEADING_MIN_SPEED else None)))
            self.n_fed += 1

    def _check(self, rec, leader_now, f_xy):
        """-> (reached, shaped target or None, shaped?, distance follower-target)."""
        a = compute_target(leader_now, rec['p'], f_xy, self.dmin, FLY_Z)
        if a is None:
            return False, None, False, float('nan')
        d = f_xy - a[:2]
        dist = float(np.linalg.norm(d))
        shaped = bool(np.linalg.norm(a[:2] - rec['p'][:2]) > SHAPED_TOL)
        if shaped:                                   # keep-out / geofence moved it: plain distance
            return dist <= R_SHAPED, a, True, dist
        if dist <= R_NEAR:
            return True, a, False, dist
        h = rec['hdg']
        if h is not None:                            # PASSED it along the path (corner cutting,
            along = float(d @ h)                     # overshoot) while staying in the corridor
            lat = abs(float(-d[0] * h[1] + d[1] * h[0]))
            if 0.0 <= along <= ALONG_MAX and lat <= R_CORR:
                return True, a, False, dist
        return False, a, False, dist

    def pop_reached(self, now, tau, leader_now, f_act):
        """Pop every head checkpoint the follower has reached (several per tick if it has
        run through a bunch). Returns the new records."""
        out = []
        while self.q:
            rec = self.q[0]
            if rec['idx'] != self._head_idx:
                self._head_idx, self._head_since = rec['idx'], now
            ok, a, shaped, dist = self._check(rec, leader_now, f_act[:2])
            if not ok:
                if (now - self._head_since > STALL_WARN_S) and self._warned != rec['idx']:
                    self._warned = rec['idx']
                    print(f"[Queue] WARNING: checkpoint {rec['idx']} has been at the head for "
                          f"{now - self._head_since:.0f} s (not skipped, still pulling the follower to it)")
                break
            self.q.popleft()
            err = now - (rec['t'] + tau)
            st = classify(err, self.tol)
            self.counts[st] += 1
            r = dict(idx=rec['idx'], leader_s=rec['t'] - self.t0,
                     x=rec['p'][0], y=rec['p'][1], z=rec['p'][2],
                     target_x=a[0], target_y=a[1], shaped=int(shaped),
                     due_s=rec['t'] + tau - self.t0, reach_s=now - self.t0,
                     timing_err=err, status=st, reach_dist=dist,
                     queue_len_after=len(self.q), J_tick=float('nan'), J_cum=float('nan'))
            self.done.append(r)
            out.append(r)
        return out

    def reference(self, now, tau):
        """-> (t_ref, head record if it is due-but-unreached else None, head_lag).
        head_lag = (now - tau) - t_head : + = the follower is that many seconds behind."""
        t_due = now - tau
        if not self.q:
            return t_due, None, 0.0
        h = self.q[0]
        lag = t_due - h['t']
        if h['t'] <= t_due:
            return h['t'], h, lag                    # due but not reached: do NOT skip it
        return t_due, None, lag                      # not due yet: follow the clock

# =============================================================================
#  TG-DPAC CONTROLLER
# =============================================================================

class TGDPAC:
    def __init__(self, cmd0, args):
        self.cmd = np.array(cmd0, dtype=float)
        self.v = np.zeros(3)
        self.amax, self.vmax, self.vesc = args.amax, args.vmax, V_ESCAPE
        self.R, self.dmin, self.follow_z = args.runaway, args.dmin, args.follow_z

    def step(self, p_f, tgt, v_ref, leader_xy, tg, dt, v_lead=None):
        if tgt is None:
            v_des = np.zeros(3)
        else:
            v_des = v_ref + (tgt - p_f) / tg
            if not self.follow_z:
                v_des[2] = 0.0

        # keep-out braking guard
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

        dv = np.clip(v_des - self.v, -self.amax * dt, self.amax * dt)
        v = self.v + dv
        lim = self.vesc if inside else self.vmax
        sp = float(np.linalg.norm(v))
        if sp > lim:
            v = v * (lim / sp)

        new = self.cmd + v * dt
        lo = np.array([SAFE_X[0] + GEO_MARGIN, SAFE_Y[0] + GEO_MARGIN])
        hi = np.array([SAFE_X[1] - GEO_MARGIN, SAFE_Y[1] - GEO_MARGIN])
        new[:2] = np.clip(new[:2], lo, hi)
        if self.follow_z:
            new[2] = float(np.clip(new[2], Z_MIN, Z_MAX))
        else:
            new[2] = tgt[2] if tgt is not None else self.cmd[2]

        off = new - p_f
        if not self.follow_z:
            off[2] = 0.0
        n_off = float(np.linalg.norm(off))
        runaway = 0
        if n_off > self.R:
            new = new - off + off * (self.R / n_off)
            runaway = 1

        v_eff = (new - self.cmd) / dt
        acc = (v_eff - self.v) / dt
        self.v, self.cmd = v_eff, new
        return dict(acc=acc, runaway=runaway)

# =============================================================================
#  LEADER SQUARE TRAJECTORY
# =============================================================================

class SquarePath:
    def __init__(self, cx, cy, side, ccw=True):
        h = side / 2.0
        pts = [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)]
        if not ccw:
            pts = [pts[0], pts[3], pts[2], pts[1]]
        self.pts = np.array(pts, dtype=float)
        self.side = float(side)
        self.perimeter = 4.0 * self.side

    def point(self, s):
        s = s % self.perimeter
        i = min(int(s // self.side), 3)
        f = (s - i * self.side) / self.side
        a, b = self.pts[i], self.pts[(i + 1) % 4]
        return a + (b - a) * f

    def fits(self, margin):
        return all(in_zone(p, margin) for p in self.pts)


class LeaderSquare:
    """goto_start -> square (N laps) -> finish (hover) -> done."""

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
                print(f"[Leader] {self.laps} lap(s) done -> hovering {self.hold_after:.1f} s "
                      f"so the follower can work through the queue")
        else:
            done = (now - self.t_fin) >= self.hold_after
        return self.cmd, done

# =============================================================================
#  COST  (evaluation only, never fed back)
# =============================================================================

def evaluate(f_act, f_vel, sched, v_sched, dist_xy, lag, acc, heading, args):
    """J = Jp + Jv + Je + Ju + Js against the SCHEDULE p_L(t - TAU); Je uses the queue-head lag."""
    mask = np.array([1.0, 1.0, 1.0 if args.follow_z else 0.0])
    nan = float('nan')
    if sched is None:
        e_p, e_v = np.zeros(3), np.zeros(3)
        e_long = e_lat = e_vert = nan
    else:
        e_raw = f_act - sched
        e_p = e_raw * mask
        e_v = (f_vel - v_sched) * mask
        if heading is None:
            e_long = e_lat = nan
        else:
            e_long = float(e_p[:2] @ heading)
            e_lat = float(-e_p[0] * heading[1] + e_p[1] * heading[0])
        e_vert = float(e_raw[2])
    Jp = W_P * float(e_p @ e_p)
    Jv = W_V * float(e_v @ e_v)
    Je = W_E * lag * lag
    Ju = W_U * float(acc @ acc)
    Js = W_S * max(0.0, args.dmin - dist_xy) ** 2
    J = Jp + Jv + Je + Ju + Js
    return dict(e_long=e_long, e_lat=e_lat, e_vert=e_vert,
                Jp=Jp, Jv=Jv, Je=Je, Ju=Ju, Js=Js, J=J, reward=-J)

# =============================================================================
#  LOGGING / SUMMARY
# =============================================================================

def _put(d, name, v):
    d[name + 'x'], d[name + 'y'], d[name + 'z'] = float(v[0]), float(v[1]), float(v[2])


def save_csv(path, keys, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in rows:
            w.writerow([r.get(k, float('nan')) for k in keys])
    print(f"  Saved {path}  ({len(rows)} rows)")


SUMMARY_KEYS = ['time', 'dry', 'laps', 'leader_speed', 'vmax', 'tau', 'tol', 'tg', 'amax',
                'n_checkpoints', 'reached', 'unvisited', 'early', 'timely', 'late', 'late_pct',
                'late_streak', 'err_mean', 'err_rms', 'err_max_late', 'err_max_early',
                'tick_late_pct', 'J_mean', 'J_int', 'G', 'Jp', 'Jv', 'Je', 'Ju', 'Js',
                'min_sep', 'duration', 'log']


def summarise(rows, wq, args, dry, log_path):
    if len(rows) < 5:
        print("\n  [summary] too few follow ticks.")
        return
    col = lambda k: np.array([r[k] for r in rows], dtype=float)
    t = col('t')
    J = col('J')
    dtv = np.diff(t, prepend=t[0] - DT)
    J_int = float(np.sum(J * dtv))
    J_mean = J_int / float(t[-1] - t[0] + DT)
    G = float(np.sum((GAMMA ** np.arange(len(J))) * (-J)))
    comp = {k: float(np.mean(col(k))) for k in ('Jp', 'Jv', 'Je', 'Ju', 'Js')}
    dist = col('dist_xy')

    cps = wq.done
    errs = np.array([c['timing_err'] for c in cps], dtype=float)
    n_r, n_p = len(cps), len(wq.q)
    c = wq.counts
    streak = best = 0
    for cp in cps:
        streak = streak + 1 if cp['status'] == 'LATE' else 0
        best = max(best, streak)
    tick_late = float(np.mean([r['status'] == 'LATE' for r in rows]) * 100)
    late_pct = 100.0 * c['LATE'] / n_r if n_r else float('nan')
    have = n_r > 0
    s = dict(n_checkpoints=wq.n_fed, reached=n_r, unvisited=n_p,
             early=c['EARLY'], timely=c['TIMELY'], late=c['LATE'], late_pct=late_pct,
             late_streak=best,
             err_mean=float(errs.mean()) if have else float('nan'),
             err_rms=float(np.sqrt(np.mean(errs ** 2))) if have else float('nan'),
             err_max_late=float(errs.max()) if have else float('nan'),
             err_max_early=float(errs.min()) if have else float('nan'),
             tick_late_pct=tick_late, J_mean=J_mean, J_int=J_int, G=G,
             min_sep=float(dist.min()), duration=float(t[-1] - t[0]), **comp)

    print("\n" + "=" * 70)
    print("  CHECKPOINT-QUEUE REPORT" + ("   (DRY RUN: both drones simulated)" if dry else ""))
    print("=" * 70)
    print(f"  leader path  : square {args.side} m, {args.laps} lap(s), {args.leader_speed} m/s, "
          f"tau {args.tau} s, timely tolerance +/-{args.tol} s")
    print(f"  checkpoints  : {wq.n_fed} queued (every leader sample), {n_r} reached, "
          f"{n_p} never reached")
    print(f"  status       : TIMELY {c['TIMELY']}   EARLY {c['EARLY']}   LATE {c['LATE']}"
          f"   ({late_pct:.1f}% late, longest late streak {best})")
    print(f"  timing error : mean {s['err_mean']:+.3f} s, RMS {s['err_rms']:.3f} s, "
          f"worst late {s['err_max_late']:+.3f} s, worst early {s['err_max_early']:+.3f} s")
    print(f"  tick status  : {tick_late:.1f}% of 100 ms ticks had the queue head more than "
          f"{args.tol} s behind schedule")
    print(f"  separation   : min {s['min_sep']:.3f} m (D_MIN {args.dmin} m)")
    print(f"  cost         : mean J {J_mean:.4f}, integrated {J_int:.3f}, "
          f"discounted return G {G:.3f}")
    print("                 mean terms  " + "  ".join(f"{k}={v:.4f}" for k, v in comp.items()))
    print("  (the first checkpoints are due while the follower is still flying from its spawn"
          " to the\n   leader's path, so some LATE at the start is normal; filter on idx/due_s in the CSV)")
    print("=" * 70)

    path = os.path.join(args.out, 'queue_runs_summary.csv')
    new = not os.path.exists(path)
    row = dict(s)
    row.update(time=time.strftime('%Y-%m-%d %H:%M:%S'), dry=int(dry), laps=args.laps,
               leader_speed=args.leader_speed, vmax=args.vmax, tau=args.tau, tol=args.tol,
               tg=args.tg, amax=args.amax, log=os.path.basename(log_path))
    with open(path, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(SUMMARY_KEYS)
        w.writerow([('%.5g' % row[k]) if isinstance(row.get(k), float) else row.get(k, '')
                    for k in SUMMARY_KEYS])
    print(f"  Appended run to {path}")


def save_checkpoints(path, wq, t0):
    recs = list(wq.done)
    for rec in wq.q:                                   # queue not drained -> mark them
        recs.append(dict(idx=rec['idx'], leader_s=rec['t'] - t0,
                         x=rec['p'][0], y=rec['p'][1], z=rec['p'][2],
                         due_s=rec['t'] + wq_tau(wq) - t0, status='UNVISITED'))
    save_csv(path, CP_KEYS, recs)


def wq_tau(wq):
    return wq.tau_for_log

# =============================================================================
#  SESSION
# =============================================================================

def prepare(args, L, F, dry):
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

    if dry:
        now = time.time()
        for k in range(int((args.tau + 2.0) / DT), -1, -1):
            L.push(now - k * DT, lcmd0, (0.0, 0.0, 0.0))
    while L.span() < args.tau + 1.0:
        time.sleep(0.1)
    return lcmd0, fcmd0


def flight(args, dry, lcf, fcf, L, F, rows, ctrl, fcmd0, wq, stop_evt):
    """Worker thread: leader parks/lands first; follower drains its queue, then lands."""
    lhc = None if dry else lcf.high_level_commander
    fhc = None if dry else fcf.high_level_commander
    leader_airborne = False
    follower_airborne = False
    leader_landing_t = None
    leader_park_started = False
    leader_park_target = np.array([PARK_X, PARK_Y, PARK_Z], dtype=float)
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
            leader_airborne = True
            follower_airborne = True
            stop_evt.wait(TAKEOFF_DUR + 1.0)

        print(f"[Run] FOLLOWING  tau={args.tau} s  D_MIN={args.dmin} m  vmax={args.vmax} m/s  "
              f"leader {args.leader_speed} m/s  timely tol +/-{args.tol} s")
        print("[Run] every leader sample is a checkpoint; status is report-only "
              "(a late follower is never stopped)")
        tg = TGDPAC(fcmd0, args)
        t_start = time.time()
        wq.start(t_start - args.tau, t_start)           # queue starts at the sample TAU before follow start
        prev_lcmd = ctrl.cmd.copy()
        t_prev = t_start
        tick = 0
        heading = None
        J_cum, G_cum = 0.0, 0.0
        last_cp = None

        while not stop_evt.is_set():
            t_loop = time.time()
            dt_real = DT if tick == 0 else min(t_loop - t_prev, 3 * DT)
            t_prev = t_loop

            if t_loop - t_start > args.duration:
                print(f"[Run] --duration cap {args.duration:.0f} s reached -> landing "
                      f"({len(wq.q)} checkpoint(s) still in the queue)")
                break

            # ---------- leader: read, safety checks, command ----------
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

            lcmd, ldone = ctrl.update(t_loop, dt_real)
            lcmd = np.array(lcmd, dtype=float)

            # Let the final FEED_TAIL_S samples enter the FIFO before moving the
            # leader away. The follower then works through the frozen queue.
            park_after = None if ctrl.t_fin is None else ctrl.t_fin + FEED_TAIL_S
            if (not leader_park_started and park_after is not None and t_loop >= park_after):
                leader_park_started = True
                print(f"[Leader] final queue samples fed -> moving to parking point "
                      f"({PARK_X:.2f}, {PARK_Y:.2f}, {PARK_Z:.2f})")

            if leader_park_started and leader_landing_t is None:
                lcmd = step_toward(prev_lcmd, leader_park_target, args.leader_speed, dt_real)
                if np.linalg.norm(lp[:2] - leader_park_target[:2]) <= 0.12 and abs(lp[2] - leader_park_target[2]) <= 0.15:
                    if not dry:
                        print("[Leader] reached parking point -> landing leader; follower continues")
                        try:
                            lhc.land(0.0, duration_s=LAND_DUR)
                            leader_landing_t = t_loop
                        except Exception as e:
                            print(f"[Leader] parking land command failed: {e} -> landing both")
                            break
                    else:
                        leader_landing_t = t_loop
                        print("[Leader] dry-run reached parking point -> simulated landing")
            elif leader_landing_t is not None:
                lcmd = leader_park_target.copy()
                if t_loop - leader_landing_t >= LAND_DUR:
                    if not dry:
                        try:
                            lhc.stop()
                        except Exception as e:
                            print(f"[Leader] stop after landing warning: {e}")
                    leader_airborne = False
                    print("[Leader] landed; follower is continuing independently")
                    # Keep the timestamp non-None so no further park/land commands are sent.

            if dry:
                lvel = (lcmd - prev_lcmd) / max(dt_real, 1e-3)
                L.push(t_loop, lcmd, lvel)
                lat = L.latest()
                age = t_loop - lat[0]
            elif leader_landing_t is None:
                lhc.go_to(lcmd[0], lcmd[1], lcmd[2], yaw=0.0, duration_s=DT, relative=False)
            prev_lcmd = lcmd.copy()

            # ---------- follower: measured state ----------
            if dry:
                f_act, f_vel, fage = tg.cmd.copy(), tg.v.copy(), 0.0
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
            fresh = age <= args.stale_hold and fage <= args.stale_hold

            # ---------- checkpoint queue ----------
            # leader hover samples after its finish are queued for FEED_TAIL_S, then the queue closes
            t_max = np.inf if ctrl.t_fin is None else ctrl.t_fin + FEED_TAIL_S
            wq.feed(L, t_max)
            popped = wq.pop_reached(t_loop, args.tau, lat[1], f_act) if fresh else []
            if popped:
                last_cp = popped[-1]
            t_ref, head, lag = wq.reference(t_loop, args.tau)

            # ---------- Stage 1: reference, velocity, safety shaping ----------
            if head is not None:
                ref, v_ref = head['p'].copy(), head['v'].copy()
            else:
                ref = L.at(t_ref)
                v_ref = np.array(L.vel_at(t_ref), dtype=float) if ref is not None else np.zeros(3)
            if not args.follow_z:
                v_ref[2] = 0.0

            sched = L.at(t_loop - args.tau)                      # the on-time schedule (for cost)
            v_sched = np.array(L.vel_at(t_loop - args.tau), dtype=float) if sched is not None else np.zeros(3)
            if not args.follow_z:
                v_sched[2] = 0.0
            sp_s = float(np.linalg.norm(v_sched[:2]))
            if sp_s > HEADING_MIN_SPEED:
                heading = v_sched[:2] / sp_s

            tgt = None
            if fresh and ref is not None:
                z = FLY_Z if not args.follow_z else float(np.clip(ref[2], Z_MIN, Z_MAX))
                tgt = compute_target(lat[1], ref, f_act[:2], args.dmin, z)

            v_ff = v_ref
            if tgt is not None and ref is not None and np.linalg.norm(tgt[:2] - ref[:2]) > FF_SHAPED_TOL:
                v_ff = np.zeros(3)

            # ---------- Stages 2-4 ----------
            final_phase = ctrl.t_fin is not None and t_ref >= ctrl.t_fin
            tg_now = args.tg_final if final_phase else args.tg
            info = tg.step(f_act, tgt, v_ff, lat[1][:2], tg_now, dt_real, L.vel_latest())
            fcmd = tg.cmd
            if not dry:
                fhc.go_to(fcmd[0], fcmd[1], fcmd[2], yaw=0.0, duration_s=DT, relative=False)

            # ---------- status + cost (reporting only, never fed back) ----------
            status = classify(lag, args.tol) if wq.q else 'TIMELY'
            dist_xy = float(np.hypot(f_act[0] - lat[1][0], f_act[1] - lat[1][1]))
            ev = evaluate(f_act, f_vel, sched, v_sched, dist_xy, lag, info['acc'], heading, args)
            J_cum += ev['J'] * dt_real
            G_cum += (GAMMA ** tick) * ev['reward']
            for rec in popped:                                   # cost at the moment of each reach
                rec['J_tick'], rec['J_cum'] = ev['J'], J_cum

            nan3 = np.full(3, np.nan)
            row = {'t': float(t_loop), 't_rel': float(t_loop - t_start), 'lphase': ctrl.phase,
                   'leader_age': float(age), 'follower_age': float(fage),
                   'dist_xy': dist_xy,
                   'dist_sched': float(np.hypot(f_act[0] - sched[0], f_act[1] - sched[1])) if sched is not None else np.nan,
                   'dist_target': float(np.hypot(f_act[0] - tgt[0], f_act[1] - tgt[1])) if tgt is not None else np.nan,
                   'acc': float(np.linalg.norm(info['acc'])), 'runaway': int(info['runaway']),
                   'tg': float(tg_now),
                   'q_len': len(wq.q),
                   'head_idx': wq.q[0]['idx'] if wq.q else -1,
                   'head_t_rel': (wq.q[0]['t'] - t_start) if wq.q else np.nan,
                   'n_reached': len(wq.done), 'n_early': wq.counts['EARLY'],
                   'n_timely': wq.counts['TIMELY'], 'n_late': wq.counts['LATE'],
                   'lag_head': float(lag), 'status': status, 'J_cum': J_cum, 'G_cum': G_cum}
            for nm, v in (('l', lat[1]), ('f', f_act), ('s', nan3 if sched is None else sched),
                          ('r', nan3 if ref is None else ref), ('t', nan3 if tgt is None else tgt),
                          ('c', fcmd), ('vs', v_sched), ('vf', f_vel), ('vc', tg.v)):
                _put(row, nm, v)
            row.update(ev)
            rows.append(row)

            if tick % 10 == 0:
                cnt = wq.counts
                print(f"  t={t_loop - t_start:5.1f}s [{ctrl.lap_text()}] "
                      f"leader=({row['lx']:.2f},{row['ly']:.2f}) follower=({row['fx']:.2f},{row['fy']:.2f}) "
                      f"sep={dist_xy:.2f} m  queue={len(wq.q)}  head_lag={lag:+.2f}s [{status}]  "
                      f"reached={len(wq.done)} (E{cnt['EARLY']}/T{cnt['TIMELY']}/L{cnt['LATE']})  "
                      f"J={ev['J']:.3f}" + ("  [HOLD]" if tgt is None else ""))
            tick += 1

            if leader_landing_t is not None and not wq.q:
                print("[Run] leader parked/landed and follower reached every checkpoint -> landing follower")
                break
            stop_evt.wait(max(0.0, DT - (time.time() - t_loop)))

        if stop_evt.is_set():
            print("[Run] stop requested -> landing")

    except Exception:
        print("\n[Run] ERROR in flight loop -> landing")
        traceback.print_exc()
    finally:
        if not dry:
            # If the leader's independent landing was already commanded, allow it
            # to finish instead of issuing a second landing command.
            if leader_landing_t is not None and leader_airborne:
                remaining = max(0.0, LAND_DUR - (time.time() - leader_landing_t))
                if remaining:
                    time.sleep(remaining)
                try:
                    lhc.stop()
                except Exception as e:
                    print(f"[Leader] stop after landing warning: {e}")
                leader_airborne = False

            landing = []
            if leader_airborne:
                landing.append(('Leader', lhc))
            if follower_airborne:
                landing.append(('Follower', fhc))
            if landing:
                print("[Landing] " + " and ".join(name for name, _ in landing))
                for name, h in landing:
                    try:
                        h.land(0.0, duration_s=LAND_DUR)
                    except Exception as e:
                        print(f"[{name}] landing command error: {e}")
                time.sleep(LAND_DUR + 0.5)
                for name, h in landing:
                    try:
                        h.stop()
                    except Exception as e:
                        print(f"[{name}] stop command error: {e}")

# =============================================================================
#  MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description='Leader flies a square; follower visits EVERY leader '
                                             'sample as a queued checkpoint (report-only status).')
    ap.add_argument('--leader-uri', default=LEADER_URI)
    ap.add_argument('--follower-uri', default=FOLLOWER_URI)
    ap.add_argument('--tau', type=float, default=TAU, help='time buffer in seconds')
    ap.add_argument('--tol', type=float, default=TIMELY_TOL_S,
                    help='|timing error| within this is TIMELY, beyond is EARLY/LATE (s)')
    ap.add_argument('--dmin', type=float, default=D_MIN, help='minimum XY distance to leader (m)')
    ap.add_argument('--vmax', type=float, default=V_MAX, help='follower speed limit (m/s)')
    ap.add_argument('--duration', type=float, default=MAX_FLIGHT_S,
                    help='safety cap on the follow phase (s); lateness alone never ends the flight earlier')
    ap.add_argument('--follow-z', action='store_true', help='copy the leader altitude (clipped)')
    ap.add_argument('--tg', type=float, default=TG_CRUISE)
    ap.add_argument('--tg-final', type=float, default=TG_FINAL)
    ap.add_argument('--amax', type=float, default=A_MAX)
    ap.add_argument('--runaway', type=float, default=R_RUNAWAY)
    ap.add_argument('--stale-hold', type=float, default=STALE_HOLD_S)
    ap.add_argument('--leader-lost', type=float, default=LEADER_LOST_S)
    ap.add_argument('--laps', type=int, default=SQ_LAPS)
    ap.add_argument('--side', type=float, default=SQ_SIDE)
    ap.add_argument('--center-x', type=float, default=SQ_CENTER[0])
    ap.add_argument('--center-y', type=float, default=SQ_CENTER[1])
    ap.add_argument('--leader-speed', type=float, default=SQ_SPEED)
    ap.add_argument('--leader-z', type=float, default=LEADER_Z)
    ap.add_argument('--cw', action='store_true', help='fly the square clockwise (default CCW)')
    ap.add_argument('--dry-run', action='store_true', help='no radios, no motors: both drones simulated')
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
    if not (SAFE_X[0] + GEO_MARGIN <= PARK_X <= SAFE_X[1] - GEO_MARGIN and
            SAFE_Y[0] + GEO_MARGIN <= PARK_Y <= SAFE_Y[1] - GEO_MARGIN and
            Z_MIN <= PARK_Z <= Z_MAX):
        raise SystemExit("PARK_X, PARK_Y and PARK_Z must be inside the geofence and altitude limits.")
    if not dry:
        park_xy = np.array([PARK_X, PARK_Y], dtype=float)
        edge_clearances = []
        for i in range(4):
            a, b = path.pts[i], path.pts[(i + 1) % 4]
            ab = b - a
            u = float(np.clip(np.dot(park_xy - a, ab) / max(np.dot(ab, ab), 1e-9), 0.0, 1.0))
            edge_clearances.append(float(np.linalg.norm(park_xy - (a + u * ab))))
        park_clearance = min(edge_clearances)
        if park_clearance < args.dmin + 0.2:
            raise SystemExit(f"Parking point is only {park_clearance:.2f} m from the square path. "
                             f"Choose a point at least {args.dmin + 0.2:.2f} m from every path edge.")
    if args.tg <= 0 or args.tg_final <= 0 or args.amax <= 0 or args.runaway <= 0 or args.tol <= 0:
        raise SystemExit("--tg, --tg-final, --amax, --runaway and --tol must be > 0")
    if args.stale_hold >= args.leader_lost:
        raise SystemExit("--stale-hold must be smaller than --leader-lost")
    if args.leader_speed > args.vmax:
        print(f"[warn] leader speed {args.leader_speed} m/s > follower vmax {args.vmax} m/s: the queue "
              f"will keep growing and the follower will stay LATE (it is not stopped for that; the "
              f"--duration cap {args.duration:.0f} s ends the run).")

    os.makedirs(args.out, exist_ok=True)
    prefix = os.path.join(args.out, f"lfq_{time.strftime('%Y%m%d_%H%M%S')}_")

    L, F, rows = PosStream(), PosStream(), []
    wq = CheckpointQueue(args.tol, args.dmin)
    wq.tau_for_log = args.tau
    stop_evt = threading.Event()
    t_follow0 = [None]
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
                args=(args, dry, lcf, fcf, L, F, rows, ctrl, fcmd0, wq, stop_evt))
            worker.start()
            try:
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
            log_path = prefix + 'log.csv'
            save_csv(log_path, TICK_KEYS, rows)
            save_checkpoints(prefix + 'checkpoints.csv', wq, wq.t0)
            summarise(rows, wq, args, dry, log_path)


if __name__ == '__main__':
    main()

