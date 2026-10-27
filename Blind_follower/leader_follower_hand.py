#!/usr/bin/env python3
"""
leader_follower_hand.py - hand-carried leader, flying follower (communication test).

LEADER   : powered, NOT armed, carried by hand. Only its LPS position
           (kalman.stateX/Y/Z, 10 Hz) is read and timestamped on the PC.
FOLLOWER : takes off, then every 100 ms flies to the leader's position from
           TAU seconds ago (rolling buffer), but never closer than D_MIN
           (horizontal) to the leader's CURRENT position.

    reference   D(t)  = p_leader(t - TAU)
    keep-out    if |D - p_leader(t)|_xy < D_MIN  ->  push D out to D_MIN from the leader
    target      geofence-clamped, speed-limited, sent with go_to()

Run the no-risk version first (follower is not connected, no motors):
    python leader_follower_hand.py --dry-run
Then the real thing:
    python leader_follower_hand.py

Ctrl+C at any time -> the follower lands.
"""

import argparse
import csv
import math
import os
import threading
import time
from contextlib import ExitStack

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

LEADER_URI   = 'radio://0/80/2M/E7E7E7E705'     # Drone 7: carried by hand
FOLLOWER_URI = 'radio://0/80/2M/E7E7E7E701'     # Drone 2: flies

SAFE_X = (0.0, 3.0)
SAFE_Y = (1.0, 4.0)
GEO_MARGIN = 0.10          # follower targets stay this far inside the safe zone
FOLLOWER_BREACH = 0.15     # follower measured this far OUTSIDE the zone -> land

FLY_Z = 1.0                # follower altitude (m)
Z_MIN, Z_MAX = 0.5, 1.5    # altitude limits when --follow-z is used

TAU    = 3.0               # s   time buffer (follower uses leader position from TAU ago)
D_MIN  = 0.5               # m   minimum horizontal distance to the leader
V_MAX  = 0.8               # m/s follower speed limit
V_ESCAPE = 1.2             # m/s speed limit while inside the keep-out circle (move away faster)

DT           = 0.1         # s   control period (10 Hz)
TAKEOFF_DUR  = 2.5
LAND_DUR     = 3.0
LPS_SETTLE   = 10.0        # s   wait for the Kalman filters after connecting
STALE_HOLD_S = 0.5         # leader data older than this -> follower holds
LEADER_LOST_S = 3.0        # leader data older than this -> follower lands
MAX_FLIGHT_S = 120.0       # follow phase is cut off after this long

# Pass criteria from the progress document
RATE_MIN_HZ, GAP_MAX_S = 9.0, 0.5
LAG_TOL_S = 0.3
SEP_TOL_M = 0.05           # tolerance below D_MIN when checking "separation >= D_MIN"

ROW_KEYS = ['t', 'phase', 'lx', 'ly', 'lz', 'leader_age', 'rx', 'ry', 'rz',
            'tx', 'ty', 'tz', 'cx', 'cy', 'cz', 'fx', 'fy', 'fz',
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
#  GEOMETRY: reference -> target -> command
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

# =============================================================================
#  ROWS / ANALYSIS
# =============================================================================

def make_row(t, phase, lat, age, ref, tgt, cmd, f):
    nan = np.full(3, np.nan)
    ref_ = nan if ref is None else ref
    tgt_ = nan if tgt is None else tgt
    d = {'t': float(t), 'phase': phase, 'leader_age': float(age)}
    for name, v in (('l', lat), ('r', ref_), ('t', tgt_), ('c', cmd), ('f', f)):
        d[name + 'x'], d[name + 'y'], d[name + 'z'] = (float(v[0]), float(v[1]), float(v[2]))
    d['dist_xy'] = float(np.hypot(f[0] - lat[0], f[1] - lat[1]))
    d['dist_ref'] = float(np.hypot(f[0] - ref_[0], f[1] - ref_[1]))
    d['ex'], d['ey'], d['ez'] = (float(f[0] - tgt_[0]), float(f[1] - tgt_[1]), float(f[2] - tgt_[2]))
    return d


def estimate_lag(Lt, Lp, Ft, Fp, max_lag=6.0, step=0.1):
    """Lag L that best explains follower(t) ~ leader(t - L) in XY. None if the
    leader barely moved (flat curve -> lag not observable)."""
    if np.ptp(Lp[:, 0]) + np.ptp(Lp[:, 1]) < 0.8:
        return None
    lags = np.arange(0.0, max_lag + 1e-9, step)
    m = Ft >= Lt[0] + max_lag
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
    lag = estimate_lag(Lt, Lp, t, Fp)

    below = float(np.mean(dist < args.dmin - SEP_TOL_M) * 100)
    s = dict(n=len(follow), duration=float(t[-1] - t[0]), rate=rate, max_gap=max_gap,
             min_sep=float(dist.min()), mean_sep=float(dist.mean()),
             pct_below=below, track_rms=track_rms, lag=lag)

    ok_rate = rate >= RATE_MIN_HZ and max_gap <= GAP_MAX_S
    ok_lag = None if lag is None else abs(lag - args.tau) <= LAG_TOL_S
    ok_sep = s['min_sep'] >= args.dmin - SEP_TOL_M
    tag = lambda ok: 'n/a (leader barely moved)' if ok is None else ('PASS' if ok else 'CHECK')

    print("\n" + "=" * 62)
    print("  LEADER-FOLLOWER REPORT" + ("   (DRY RUN: follower = commanded setpoint)" if dry else ""))
    print("=" * 62)
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


def save_plots(rows, args, prefix):
    follow = [r for r in rows if r['phase'] == 'follow']
    if len(follow) < 5:
        return
    col = lambda k: np.array([r[k] for r in follow])
    t = col('t') - follow[0]['t']

    # ---- XY ----
    fig, ax = plt.subplots(figsize=(7, 7))
    bx = [SAFE_X[0], SAFE_X[1], SAFE_X[1], SAFE_X[0], SAFE_X[0]]
    by = [SAFE_Y[0], SAFE_Y[0], SAFE_Y[1], SAFE_Y[1], SAFE_Y[0]]
    ax.plot(bx, by, 'r--', lw=1.1, alpha=0.6, label='Safety boundary')
    ax.plot(col('lx'), col('ly'), color='#3266ad', lw=1.6, label='Leader (hand-carried)')
    ax.plot(col('tx'), col('ty'), '--', color='#3a8a5a', lw=1.1, label=f'Follower target (t-{args.tau:.0f}s, keep-out)')
    ax.plot(col('fx'), col('fy'), color='#c94a2a', lw=1.4, label='Follower actual')
    ax.plot(col('lx')[0], col('ly')[0], 'o', color='#3266ad')
    ax.plot(col('fx')[0], col('fy')[0], 'o', color='#c94a2a')
    ax.add_patch(plt.Circle((col('lx')[-1], col('ly')[-1]), args.dmin, fill=False,
                            ls=':', color='#888780', label=f'D_MIN={args.dmin} m (at end)'))
    ax.set_aspect('equal')
    ax.set_xlim(SAFE_X[0] - 0.2, SAFE_X[1] + 0.2)
    ax.set_ylim(SAFE_Y[0] - 0.2, SAFE_Y[1] + 0.2)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_title('Top-down: leader path vs delayed follower'); ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc='upper right')
    fig.savefig(prefix + 'xy.png', dpi=130, bbox_inches='tight'); plt.close(fig)

    # ---- time series ----
    fig, axs = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    dist = col('dist_xy')
    axs[0].plot(t, dist, color='#3266ad', label='XY distance follower-leader')
    axs[0].axhline(args.dmin, color='#c94a2a', ls='--', label=f'D_MIN = {args.dmin} m')
    axs[0].fill_between(t, 0, np.minimum(dist, args.dmin), color='#c94a2a', alpha=0.15)
    axs[0].set_ylabel('Separation (m)'); axs[0].legend(fontsize=8)
    axs[0].set_title(f"min {dist.min():.3f} m   mean {dist.mean():.3f} m")
    for ax, ck, tk, fk, name in ((axs[1], 'lx', 'tx', 'fx', 'X'), (axs[2], 'ly', 'ty', 'fy', 'Y')):
        ax.plot(t, col(ck), color='#3266ad', label='Leader')
        ax.plot(t, col(tk), '--', color='#3a8a5a', label=f'Target (t-{args.tau:.0f}s)')
        ax.plot(t, col(fk), color='#c94a2a', label='Follower')
        ax.set_ylabel(f'{name} (m)'); ax.legend(fontsize=8, ncol=3, loc='upper right')
    axs[2].set_xlabel('Time since follow start (s)')
    for ax in axs:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(prefix + 'time.png', dpi=130); plt.close(fig)
    print(f"  Saved {prefix}xy.png, {prefix}time.png")

# =============================================================================
#  SESSION
# =============================================================================

def run_session(args, fcf, L, F, rows):
    """fcf: follower Crazyflie object (None in dry-run)."""
    dry = fcf is None
    hlc = None if dry else fcf.high_level_commander
    airborne = False

    try:
        print(f"\n[LPS] Waiting {LPS_SETTLE:.0f} s for the estimators to settle ...")
        time.sleep(LPS_SETTLE)

        lat = L.latest()
        if lat is None:
            raise RuntimeError("No position from the LEADER. Check its LPS deck / radio.")
        if not in_zone(lat[1][:2], 0.0):
            print(f"[warn] leader is outside the safe zone: ({lat[1][0]:.2f}, {lat[1][1]:.2f})")
        print(f"[Leader]   at ({lat[1][0]:.2f}, {lat[1][1]:.2f}, {lat[1][2]:.2f})")

        if dry:
            cmd = np.array([(SAFE_X[0] + SAFE_X[1]) / 2, SAFE_Y[0] + 0.5, FLY_Z])
            print(f"[Follower] DRY RUN - virtual follower starts at ({cmd[0]:.2f}, {cmd[1]:.2f})")
        else:
            fl = F.latest()
            if fl is None:
                raise RuntimeError("No position from the FOLLOWER. Check its LPS deck / radio.")
            sp = fl[1]
            print(f"[Follower] at ({sp[0]:.2f}, {sp[1]:.2f}, {sp[2]:.2f})")
            if not in_zone(sp[:2], 0.1):
                raise RuntimeError("Follower spawn is outside / too close to the safe zone. Aborting.")
            cmd = np.array([sp[0], sp[1], FLY_Z])

        input("\nPlace the leader where you want to start, then press ENTER to "
              + ("start (dry run) ... " if dry else "TAKE OFF ... "))

        while L.span() < args.tau + 1.0:        # buffer must reach back TAU seconds
            time.sleep(0.1)

        if not dry:
            fcf.param.set_value('commander.enHighLevel', '1')
            time.sleep(0.3)
            try:
                fcf.supervisor.send_arming_request(True)
            except AttributeError:
                pass
            time.sleep(0.5)
            print(f"[Follower] TAKEOFF -> z={FLY_Z} m")
            hlc.takeoff(FLY_Z, TAKEOFF_DUR)
            airborne = True
            time.sleep(TAKEOFF_DUR + 1.0)

        print(f"[Follower] FOLLOWING  tau={args.tau} s  D_MIN={args.dmin} m  "
              f"vmax={args.vmax} m/s   (Ctrl+C to land)")
        t_start = time.time()
        tick = 0
        while True:
            t_loop = time.time()
            if t_loop - t_start > args.duration:
                print("[Follower] max duration reached -> landing")
                break

            lat = L.latest()
            age = t_loop - lat[0]
            if age > LEADER_LOST_S:
                print(f"[Follower] leader data lost for {age:.1f} s -> landing")
                break

            if dry:
                f_act = cmd.copy()
            else:
                f_act = F.latest()[1]
                if not (SAFE_X[0] - FOLLOWER_BREACH <= f_act[0] <= SAFE_X[1] + FOLLOWER_BREACH and
                        SAFE_Y[0] - FOLLOWER_BREACH <= f_act[1] <= SAFE_Y[1] + FOLLOWER_BREACH):
                    print(f"[Follower] left the safe zone at ({f_act[0]:.2f}, {f_act[1]:.2f}) -> landing")
                    break

            ref = L.at(t_loop - args.tau)
            tgt = None
            if age <= STALE_HOLD_S and ref is not None:
                z = FLY_Z if not args.follow_z else float(np.clip(ref[2], Z_MIN, Z_MAX))
                tgt = compute_target(lat[1], ref, f_act[:2], args.dmin, z)
                if tgt is not None:
                    cmd = slew(cmd, tgt, lat[1], args.dmin, DT, args.vmax, V_ESCAPE)
            # (tgt is None -> hold the last command)

            if not dry:
                hlc.go_to(cmd[0], cmd[1], cmd[2], yaw=0.0, duration_s=DT, relative=False)

            rows.append(make_row(t_loop, 'follow', lat[1], age, ref, tgt, cmd, f_act))

            if tick % 10 == 0:
                r = rows[-1]
                print(f"  t={t_loop - t_start:5.1f}s  leader=({r['lx']:.2f},{r['ly']:.2f})  "
                      f"target=({r['tx']:.2f},{r['ty']:.2f})  follower=({r['fx']:.2f},{r['fy']:.2f})  "
                      f"sep={r['dist_xy']:.2f} m  leader_age={age:.2f}s"
                      + ("  [HOLD]" if tgt is None else ""))
            tick += 1
            time.sleep(max(0.0, DT - (time.time() - t_loop)))

    except KeyboardInterrupt:
        print("\n[Follower] Ctrl+C -> landing")
    finally:
        if airborne:
            try:
                print("[Follower] LANDING")
                hlc.land(0.0, duration_s=LAND_DUR)
                time.sleep(LAND_DUR + 0.5)
                hlc.stop()
            except Exception as e:
                print(f"[Follower] landing command error: {e}")

# =============================================================================
#  MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description='Hand-carried leader, delayed follower.')
    ap.add_argument('--leader-uri', default=LEADER_URI)
    ap.add_argument('--follower-uri', default=FOLLOWER_URI)
    ap.add_argument('--tau', type=float, default=TAU, help='time buffer in seconds')
    ap.add_argument('--dmin', type=float, default=D_MIN, help='minimum XY distance to leader (m)')
    ap.add_argument('--vmax', type=float, default=V_MAX, help='follower speed limit (m/s)')
    ap.add_argument('--duration', type=float, default=MAX_FLIGHT_S, help='max follow time (s)')
    ap.add_argument('--follow-z', action='store_true', help='copy the leader altitude (clipped)')
    ap.add_argument('--dry-run', action='store_true',
                    help='leader only: no follower connection, no motors')
    ap.add_argument('--out', default='./lf_runs')
    args = ap.parse_args()

    if not CF_AVAILABLE:
        raise SystemExit("cflib not found.  pip install cflib")
    cflib.crtp.init_drivers()
    os.makedirs(args.out, exist_ok=True)
    prefix = os.path.join(args.out, f"lf_{time.strftime('%Y%m%d_%H%M%S')}_")

    L, F, rows = PosStream(), PosStream(), []
    blocks = []
    try:
        with ExitStack() as stack:
            lscf = stack.enter_context(SyncCrazyflie(args.leader_uri, cf=Crazyflie(rw_cache='./cache_leader')))
            blocks.append(start_pos_logging(lscf.cf, 'LeaderPos', L))
            fcf = None
            if not args.dry_run:
                fscf = stack.enter_context(SyncCrazyflie(args.follower_uri, cf=Crazyflie(rw_cache='./cache_follower')))
                fcf = fscf.cf
                blocks.append(start_pos_logging(fcf, 'FollowerPos', F))
            run_session(args, fcf, L, F, rows)
            for lc in blocks:
                try:
                    lc.stop()
                except Exception:
                    pass
    finally:
        if rows:
            save_csv(rows, prefix + 'log.csv')
            analyse(rows, L, args, args.dry_run)
            save_plots(rows, args, prefix)


if __name__ == '__main__':
    main()
