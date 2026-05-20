#!/usr/bin/env python3

import time
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper
from cflib.utils.reset_estimator import reset_estimator

# URI to the Crazyflie
uri = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E2')

# -------- Circle Parameters --------
# Safe boundary: square with vertices (0,1,0), (0,4,0), (3,1,0), (3,4,0)
#   X range: 0 → 3  (width  = 3 m, half = 1.5 m)
#   Y range: 1 → 4  (height = 3 m, half = 1.5 m)
#   Center  = (1.5, 2.5)
#
# Maximum inscribed circle radius = 1.5 m.
# We use 1.2 m to keep a 0.3 m safety margin on every side.
#
# Boundary check at any angle θ:
#   x(θ) = 1.5 + 1.2·cos(θ)  ∈ [0.3, 2.7]  ✓ inside [0, 3]
#   y(θ) = 2.5 + 1.2·sin(θ)  ∈ [1.3, 3.7]  ✓ inside [1, 4]

RADIUS = 1.2        # metres (1.5 m half-side − 0.3 m margin)
OMEGA  = 0.7        # rad/s  (~one lap every 8.98 s)

# 20 s ≈ 2.2 full laps — enough to observe tracking behaviour
CIRCLE_DURATION = 20.0   # seconds

# Waypoint step — 10 Hz keeps go_to chain smooth
DT = 0.1

# Circle center expressed as displacement from takeoff origin (0, 0)
CENTER_REL_X = 1.5   # world-frame X of circle center
CENTER_REL_Y = 2.5   # world-frame Y of circle center
FLY_Z        = 0.6   # cruise altitude (m)

# -------- Logging globals --------
log_data_list = []
log_conf = None


# -------- LOG CALLBACK ----------

def log_callback(timestamp, data, logconf):
    log_data_list.append({
        'time':       timestamp,
        'actual_x':   data.get('kalman.stateX',  0),
        'actual_y':   data.get('kalman.stateY',  0),
        'actual_z':   data.get('kalman.stateZ',  0),
        'expected_x': data.get('ctrltarget.x',   0),
        'expected_y': data.get('ctrltarget.y',   0),
        'expected_z': data.get('ctrltarget.z',   0),
    })


# -------- LOG SETUP ----------

def setup_logging(cf):
    global log_conf

    log_conf = LogConfig(name='TrajectoryLog', period_in_ms=100)
    log_conf.add_variable('kalman.stateX', 'float')
    log_conf.add_variable('kalman.stateY', 'float')
    log_conf.add_variable('kalman.stateZ', 'float')
    log_conf.add_variable('ctrltarget.x', 'float')
    log_conf.add_variable('ctrltarget.y', 'float')
    log_conf.add_variable('ctrltarget.z', 'float')

    cf.log.add_config(log_conf)
    log_conf.data_received_cb.add_callback(log_callback)


# -------- SAVE CSV ----------

def save_log_to_csv(log_data, filename='trajectory_log.csv'):
    if not log_data:
        print("No logs to save!")
        return

    keys = ['time', 'actual_x', 'actual_y', 'actual_z',
            'expected_x', 'expected_y', 'expected_z']

    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in log_data:
            writer.writerow(row)

    print(f"Saved CSV: {filename}")


# -------- PLOTTING ----------

def plot_and_save_data(log_data):
    if not log_data:
        print('No data to plot.')
        return

    t  = np.array([d['time']       for d in log_data]) / 1000.0
    t -= t[0]

    ax_arr = np.array([d['actual_x']   for d in log_data])
    ay_arr = np.array([d['actual_y']   for d in log_data])
    az_arr = np.array([d['actual_z']   for d in log_data])

    ex_arr = np.array([d['expected_x'] for d in log_data])
    ey_arr = np.array([d['expected_y'] for d in log_data])
    ez_arr = np.array([d['expected_z'] for d in log_data])

    # Draw safety boundary on XY plot
    bx = [0, 3, 3, 0, 0]
    by = [1, 1, 4, 4, 1]

    plt.figure()
    plt.plot(bx, by, 'r--', linewidth=1.5, label='Safety boundary')
    plt.plot(ax_arr, ay_arr, label='Actual')
    plt.plot(ex_arr, ey_arr, '--', label='Expected')
    plt.title("Top-down XY — circle inside safe square")
    plt.xlabel('X (m)'); plt.ylabel('Y (m)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig("trajectory_xy_topdown.png")
    print("Saved trajectory_xy_topdown.png")

    # 3D plot
    fig = plt.figure()
    ax3 = fig.add_subplot(111, projection='3d')
    ax3.plot(ax_arr, ay_arr, az_arr, label='Actual')
    ax3.set_xlabel('X'); ax3.set_ylabel('Y'); ax3.set_zlabel('Z')
    ax3.legend()
    plt.savefig("trajectory_3d.png")
    print("Saved trajectory_3d.png")


# -------- MAIN SEQUENCE ----------

def run_sequence(cf):
    commander = cf.high_level_commander

    log_conf.start()

    # Arm
    try:
        cf.supervisor.send_arming_request(True)
    except AttributeError:
        cf.platform.send_arming_request(True)
    time.sleep(1.0)

    # ------------------------------------------------------------------
    # PHASE 1 — TAKEOFF  (0 – 3.5 s)
    # Drone lifts vertically to FLY_Z from the takeoff origin (0, 0).
    # ------------------------------------------------------------------
    print("Phase 1: Taking off to 0.6 m ...")
    commander.takeoff(FLY_Z, 2.5)
    time.sleep(3.5)

    # ------------------------------------------------------------------
    # PHASE 2 — MOVE TO CIRCLE CENTER  (3.5 – 8.5 s)
    # Translate from (0, 0) to (1.5, 2.5) — the center of the safe square.
    # relative=True means the offset is added to the current position.
    # ------------------------------------------------------------------
    print(f"Phase 2: Moving to circle center ({CENTER_REL_X}, {CENTER_REL_Y}) ...")
    commander.go_to(CENTER_REL_X, CENTER_REL_Y, FLY_Z,
                    yaw=0.0, duration_s=4.0, relative=True)
    time.sleep(4.5)

    # ------------------------------------------------------------------
    # PHASE 3 — CIRCLE  (8.5 – 28.5 s = 20 s)
    #
    # World-frame trajectory:
    #   x(t) = 1.5 + 1.2·cos(0.7·t)   ∈ [0.3, 2.7]  — 0.3 m inside X walls
    #   y(t) = 2.5 + 1.2·sin(0.7·t)   ∈ [1.3, 3.7]  — 0.3 m inside Y walls
    #
    # go_to waypoints are issued every DT=0.1 s with relative=False so the
    # high-level commander tracks explicit absolute positions.
    # ------------------------------------------------------------------
    print(f"Phase 3: Circle — r={RADIUS} m, ω={OMEGA} rad/s, {CIRCLE_DURATION} s ...")

    cx = CENTER_REL_X   # 1.5
    cy = CENTER_REL_Y   # 2.5

    steps = int(CIRCLE_DURATION / DT)
    theta = 0.0   # start angle → first waypoint at (cx+r, cy) = (2.7, 2.5)

    for _ in range(steps):
        wx = cx + RADIUS * math.cos(theta)
        wy = cy + RADIUS * math.sin(theta)

        # Safety guard — should never trigger given the geometry above
        wx = max(0.05, min(2.95, wx))
        wy = max(1.05, min(3.95, wy))

        commander.go_to(wx, wy, FLY_Z, yaw=0.0, duration_s=DT, relative=False)

        theta += OMEGA * DT
        time.sleep(DT)

    # ------------------------------------------------------------------
    # PHASE 4 — LAND  (28.5 – 32 s)
    # ------------------------------------------------------------------
    print("Phase 4: Landing ...")
    commander.land(0.0, duration_s=3.0)
    time.sleep(3.5)

    commander.stop()
    log_conf.stop()
    print("Sequence complete.")


# -------- MAIN ----------

if __name__ == '__main__':
    cflib.crtp.init_drivers()

    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
        cf = scf.cf

        setup_logging(cf)
        reset_estimator(cf)
        run_sequence(cf)

    save_log_to_csv(log_data_list)
    plot_and_save_data(log_data_list)
