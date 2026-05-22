#!/usr/bin/env python3
"""
real_boids_swarm.py
====================
Real-hardware Boids swarm for TWO Crazyflie drones via LPS positioning.
Modified to:
  1. Fly from Point A (0,1) to Point B (3,4) with a 0.3m height separation.
  2. Run the dynamic Boids loop for 60 seconds.
  3. Navigate to and land at dedicated target locations:
     - Drone 7 lands around (2.2, 2.0)
     - Drone 2 lands around (2.0, 3.0)

Hardware config
---------------
  Drone 1 : radio://0/80/2M/E7E7E7E7E7   (URI ending in 7)
  Drone 2 : radio://0/80/2M/E7E7E7E7E2   (URI ending in 2)
  Channel : 80 / 2M  (both share the same radio channel)

Safe zone (XY)
--------------
  Corners : (0,1) → (0,4) → (3,4) → (3,1)
  X range : [0.0, 3.0]
  Y range : [1.0, 4.0]
  Z range : [0.0, 1.2]  (max height expanded to allow Z-separation)
"""

import time
import threading
import numpy as np
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig

# ─────────────────────────────────────────────────────────────────────────────
#  RADIO URIs  (channel 80 / 2M bandwidth)
# ─────────────────────────────────────────────────────────────────────────────
URI_7 = 'radio://0/80/2M/E7E7E7E7E7'   # drone whose address ends in 7
URI_2 = 'radio://0/80/2M/E7E7E7E7E2'   # drone whose address ends in 2

# ─────────────────────────────────────────────────────────────────────────────
#  SAFE ZONE  (lab absolute coordinates, metres)
# ─────────────────────────────────────────────────────────────────────────────
SAFE_X_MIN, SAFE_X_MAX = 0.0, 3.0
SAFE_Y_MIN, SAFE_Y_MAX = 1.0, 4.0
SAFE_Z_MIN, SAFE_Z_MAX = 0.0, 1.2

# Inner margin used for hard clamping of commanded positions
CLAMP_MARGIN  = 0.15   # m

# Distance from a wall at which repulsion force starts
WALL_MARGIN   = 0.35   # m

# ─────────────────────────────────────────────────────────────────────────────
#  FLIGHT PARAMETERS & ALTITUDE SEPARATION
# ─────────────────────────────────────────────────────────────────────────────
CRUISE_Z_7    = 0.6    # m – Drone 7 Cruise Altitude
CRUISE_Z_2    = 0.9    # m – Drone 2 Cruise Altitude (0.3m higher than Drone 7)
TAKEOFF_DUR   = 2.5    # s – takeoff ramp time
TRANSIT_DUR   = 6.0    # s – time allocated to fly from Point A to Point B
LPS_SETTLE    = 4.0    # s – wait for Kalman filter after connect

# ─────────────────────────────────────────────────────────────────────────────
#  BOIDS PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
SEP_DIST      = 0.8    # m – separation threshold
SEP_WEIGHT    = 1.5    # separation steering weight
COH_WEIGHT    = 0.4    # cohesion steering weight
WALL_WEIGHT   = 2.0    # wall-repulsion weight
MAX_STEP      = 0.15   # m – maximum displacement per tick
BOIDS_DT      = 0.20   # s – boids tick period (~5 Hz)
MISSION_TIME  = 60.0   # s – total boids flight time before auto-land

# Radio stagger: offset in seconds between drone-7 and drone-2 commands.
# Prevents simultaneous go_to packets on the shared channel 80/2M.
RADIO_STAGGER = 0.10   # s  (drone-2 waits this long before each tick)

# ─────────────────────────────────────────────────────────────────────────────
#  SHARED POSITION STATE  (updated by per-drone logging callbacks)
# ─────────────────────────────────────────────────────────────────────────────
_state_lock = threading.Lock()
_drone_pos  = {
    URI_7: np.array([1.0, 2.0, 0.0]),   
    URI_2: np.array([2.0, 3.0, 0.0]),
}


# ─────────────────────────────────────────────────────────────────────────────
#  UTILITY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _clip_to_safe(pos: np.ndarray, target_z: float) -> np.ndarray:
    """Hard-clamp a 3-D position to the interior of the safe zone."""
    cm = CLAMP_MARGIN
    return np.array([
        _clamp(pos[0], SAFE_X_MIN + cm, SAFE_X_MAX - cm),
        _clamp(pos[1], SAFE_Y_MIN + cm, SAFE_Y_MAX - cm),
        _clamp(pos[2], target_z - 0.05, target_z + 0.05), # Force retention of specific plane
    ])


def _wall_repulsion(pos: np.ndarray) -> np.ndarray:
    """Return a 3-D force vector pushing away from boundaries."""
    force = np.zeros(3)
    wm = WALL_MARGIN

    # X walls
    excess = (SAFE_X_MIN + wm) - pos[0]
    if excess > 0: force[0] += excess          
    excess = pos[0] - (SAFE_X_MAX - wm)
    if excess > 0: force[0] -= excess          

    # Y walls
    excess = (SAFE_Y_MIN + wm) - pos[1]
    if excess > 0: force[1] += excess
    excess = pos[1] - (SAFE_Y_MAX - wm)
    if excess > 0: force[1] -= excess

    return force


# ─────────────────────────────────────────────────────────────────────────────
#  BOIDS CORE
# ─────────────────────────────────────────────────────────────────────────────

def _boids_next_target(my_uri: str, other_uri: str, my_target_z: float) -> np.ndarray:
    """Compute the next waypoint using Boids rules, locked to specific altitude."""
    with _state_lock:
        me    = _drone_pos[my_uri].copy()
        other = _drone_pos[other_uri].copy()

    # Calculate horizontal separation only to keep the 0.3m vertical safety gap independent
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

    # Combine forces
    net = (SEP_WEIGHT  * sep_force +
           COH_WEIGHT  * coh_force +
           WALL_WEIGHT * wall_force)
    net[2] = 0.0 

    # Limit maximum step per tick
    norm = np.linalg.norm(net)
    if norm > MAX_STEP:
        net = net / norm * MAX_STEP

    target = _clip_to_safe(me + net, my_target_z)
    return target


# ─────────────────────────────────────────────────────────────────────────────
#  LPS POSITION READ
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
#  CONTINUOUS POSITION TRACKING
# ─────────────────────────────────────────────────────────────────────────────

def _start_position_tracking(cf, uri: str) -> LogConfig:
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


# ─────────────────────────────────────────────────────────────────────────────
#  PER-DRONE FLIGHT THREAD
# ─────────────────────────────────────────────────────────────────────────────

def _drone_thread(uri: str,
                  other_uri: str,
                  ready_evt: threading.Event,
                  go_evt:    threading.Event,
                  stop_evt:  threading.Event):
    
    label = uri[-1]   # '7' or '2'
    
    # Assign altitude planes based on active drone ID
    my_cruise_z = CRUISE_Z_7 if uri == URI_7 else CRUISE_Z_2

    # Define requested landing zones
    if uri == URI_7:
        land_x, land_y = 2.2, 2.0
    else:
        land_x, land_y = 2.0, 3.0

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

            # Read absolute LPS placement (wherever the drone was physically put)
            spawn = _read_lps_position(cf)
            with _state_lock:
                _drone_pos[uri] = spawn
            print(f"[Drone {label}] Spawned at: x={spawn[0]:.2f}  y={spawn[1]:.2f}  z={spawn[2]:.2f}")

            # ── Check if spawn is outside safe zone ────────────────────────
            # We do NOT abort. Instead we take off and fly to the nearest
            # safe-zone entry point before the main mission begins.
            m = CLAMP_MARGIN
            spawn_outside = not (
                SAFE_X_MIN + m <= spawn[0] <= SAFE_X_MAX - m and
                SAFE_Y_MIN + m <= spawn[1] <= SAFE_Y_MAX - m
            )
            if spawn_outside:
                # Clamp the spawn XY to the nearest safe interior point
                entry_x = _clamp(spawn[0], SAFE_X_MIN + m, SAFE_X_MAX - m)
                entry_y = _clamp(spawn[1], SAFE_Y_MIN + m, SAFE_Y_MAX - m)
                print(f"[Drone {label}] ⚠ Spawn is OUTSIDE safe zone – "
                      f"will enter at ({entry_x:.2f}, {entry_y:.2f}) after takeoff.")
            else:
                entry_x, entry_y = spawn[0], spawn[1]
                print(f"[Drone {label}] ✓ Spawn is inside safe zone.")

            pos_log = _start_position_tracking(cf, uri)

            print(f"[Drone {label}] Ready – waiting for partner drone ...")
            ready_evt.set()
            go_evt.wait()          

            # ── Step 1: Lift-off to Assigned Altitude Plane ──────────────────
            print(f"[Drone {label}] TAKEOFF  ↑  {spawn[0]:.2f},{spawn[1]:.2f},{spawn[2]:.2f}"
                  f"  →  z={my_cruise_z:.2f} m")
            hlc.takeoff(my_cruise_z, TAKEOFF_DUR)
            time.sleep(TAKEOFF_DUR + 1.0)

            # ── Step 2: If spawned outside, enter the safe zone first ─────────
            if spawn_outside:
                dur_entry = max(3.0, abs(entry_x - spawn[0]) / 0.4 +
                                      abs(entry_y - spawn[1]) / 0.4)
                print(f"[Drone {label}] ENTRY    "
                      f"({spawn[0]:.2f},{spawn[1]:.2f}) → ({entry_x:.2f},{entry_y:.2f})"
                      f"  dur={dur_entry:.1f}s")
                hlc.go_to(entry_x, entry_y, my_cruise_z,
                          yaw=0.0, duration_s=dur_entry, relative=False)
                time.sleep(dur_entry + 0.8)

            # ── Step 3: Route to Point A (safe-zone corner near 0,1) ──────────
            pt_a_x = SAFE_X_MIN + CLAMP_MARGIN   # 0.15
            pt_a_y = SAFE_Y_MIN + CLAMP_MARGIN   # 1.15
            print(f"[Drone {label}] POINT-A  "
                  f"({entry_x:.2f},{entry_y:.2f}) → ({pt_a_x:.2f},{pt_a_y:.2f})  "
                  f"z={my_cruise_z:.2f}")
            hlc.go_to(pt_a_x, pt_a_y, my_cruise_z, yaw=0.0, duration_s=4.0, relative=False)
            time.sleep(4.5)

            # ── Step 4: Transit Point A → Point B (safe-zone far corner) ─────
            pt_b_x = SAFE_X_MAX - CLAMP_MARGIN   # 2.85
            pt_b_y = SAFE_Y_MAX - CLAMP_MARGIN   # 3.85
            print(f"[Drone {label}] POINT-B  "
                  f"({pt_a_x:.2f},{pt_a_y:.2f}) → ({pt_b_x:.2f},{pt_b_y:.2f})  "
                  f"z={my_cruise_z:.2f}  dur={TRANSIT_DUR:.0f}s")
            hlc.go_to(pt_b_x, pt_b_y, my_cruise_z, yaw=0.0,
                      duration_s=TRANSIT_DUR, relative=False)
            time.sleep(TRANSIT_DUR + 0.5)

            # ── Step 5: Boids swarm loop (from Point B) ───────────────────────
            # Drone-2 starts each tick RADIO_STAGGER seconds after Drone-7
            # so their go_to packets never hit the shared radio simultaneously.
            if uri == URI_2:
                time.sleep(RADIO_STAGGER)

            print(f"[Drone {label}] BOIDS START  pos=({pt_b_x:.2f},{pt_b_y:.2f})  "
                  f"z={my_cruise_z:.2f}  mission={MISSION_TIME:.0f}s")
            t_start = time.time()
            tick    = 0
            prev_tx, prev_ty = pt_b_x, pt_b_y

            while not stop_evt.is_set():
                elapsed = time.time() - t_start
                if elapsed > MISSION_TIME:
                    break

                target = _boids_next_target(uri, other_uri, my_cruise_z)
                tx, ty, tz = float(target[0]), float(target[1]), float(target[2])

                # Each drone issues its command at a different moment in the tick,
                # eliminating simultaneous radio contention on channel 80.
                hlc.go_to(tx, ty, tz, yaw=0.0,
                          duration_s=BOIDS_DT * 1.6, relative=False)

                tick += 1
                if tick % 10 == 0:   # print every 2 s
                    print(f"[Drone {label}] BOIDS  t={elapsed:5.1f}s  "
                          f"({prev_tx:.2f},{prev_ty:.2f}) → ({tx:.2f},{ty:.2f})  "
                          f"z={tz:.2f}")
                prev_tx, prev_ty = tx, ty

                # Drone-2 stagger: sleep slightly longer so next tick is offset
                if uri == URI_2:
                    time.sleep(BOIDS_DT)
                else:
                    time.sleep(BOIDS_DT)

            # ── Step 6: Fly to assigned landing pad ───────────────────────────
            with _state_lock:
                cur = _drone_pos[uri].copy()
            print(f"[Drone {label}] LAND-NAV  "
                  f"({cur[0]:.2f},{cur[1]:.2f}) → ({land_x:.2f},{land_y:.2f})  "
                  f"z={my_cruise_z:.2f}  dur=4s")
            hlc.go_to(land_x, land_y, my_cruise_z, yaw=0.0,
                      duration_s=4.0, relative=False)
            time.sleep(4.5)

            # ── Step 7: Controlled vertical landing ───────────────────────────
            print(f"[Drone {label}] LANDING ↓  ({land_x:.2f},{land_y:.2f})  "
                  f"z={my_cruise_z:.2f} → 0.0")
            hlc.land(0.0, duration_s=4.0)
            time.sleep(4.5)
            hlc.stop()

            pos_log.stop()
            print(f"[Drone {label}] Disconnected safely.")

    except Exception as exc:
        print(f"[Drone {label}] ERROR: {exc}")
        stop_evt.set()          
        ready_evt.set()         
        raise


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN SYSTEM ENTRY
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Real-Hardware Swarm: Point A -> Point B with Target Landing Pads")
    print("=" * 60)
    print(f"  Drone 7 Target Plane: {CRUISE_Z_7} m  | Landing Spot: (2.2, 2.0)")
    print(f"  Drone 2 Target Plane: {CRUISE_Z_2} m  | Landing Spot: (2.0, 3.0)")
    print(f"  Safe bounds: X [{SAFE_X_MIN}, {SAFE_X_MAX}] Y [{SAFE_Y_MIN}, {SAFE_Y_MAX}]")
    print("=" * 60)
    input("\nPress ENTER to verify anchors, spawn positions, and initialize flight: ")

    cflib.crtp.init_drivers()

    stop_evt  = threading.Event()
    go_evt    = threading.Event()
    ready_7   = threading.Event()
    ready_2   = threading.Event()

    t7 = threading.Thread(target=_drone_thread, args=(URI_7, URI_2, ready_7, go_evt, stop_evt), daemon=True)
    t2 = threading.Thread(target=_drone_thread, args=(URI_2, URI_7, ready_2, go_evt, stop_evt), daemon=True)

    t7.start()
    t2.start()

    ready_7.wait(timeout=90)
    ready_2.wait(timeout=90)

    if not ready_7.is_set() or not ready_2.is_set() or stop_evt.is_set():
        print("[Main] Pre-flight systems timeout or error. Terminating.")
        stop_evt.set()
        return

    print("[Main] Flight authorization verified. Synchronizing departure in 3s...")
    time.sleep(3.0)
    go_evt.set()

    try:
        t7.join()
        t2.join()
    except KeyboardInterrupt:
        print("\n[Main] User abort signaled -- forcing landing protocols...")
        stop_evt.set()
        t7.join(timeout=5)
        t2.join(timeout=5)

    print("\n[Main] Operations concluded successfully.")


if __name__ == '__main__':
    main()