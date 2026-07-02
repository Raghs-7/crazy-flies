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

# NOTE: reset_estimator is intentionally NOT imported.
# With the LPS deck the Kalman filter already knows the drone's
# absolute lab position — resetting it would destroy that information.

uri = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E701')

# -------- Safe zone (your lab square) --------
# Vertices: (0,1), (0,4), (3,1), (3,4)
SAFE_X_MIN, SAFE_X_MAX = 0.0, 3.0
SAFE_Y_MIN, SAFE_Y_MAX = 1.0, 4.0

# -------- Circle parameters --------
# Centre of the safe square in absolute lab coordinates
CIRCLE_CENTER_X = 1.5
CIRCLE_CENTER_Y = 2.5

RADIUS = 1.2    # metres  (0.3 m margin from every wall)
OMEGA  = 0.7    # rad/s
CIRCLE_DURATION = 20.0   # seconds
DT     = 0.1    # seconds per waypoint (10 Hz)
FLY_Z  = 0.6    # cruise altitude (m)

# How long to wait for the LPS estimator to converge at boot
LPS_SETTLE_TIME = 10.0   # seconds

# -------- Logging globals --------
log_data_list = []
log_conf      = None

# Spawn position read from LPS (filled before flight)
spawn_x = None
spawn_y = None
spawn_z = None


# -------- POSITION READING -------

def read_lps_position(cf):
    """
    Block until the Kalman filter has a stable LPS fix,
    then return (x, y, z) in absolute lab coordinates.

    kalman.stateX/Y/Z are the filtered position estimates that the
    LPS anchors continuously correct.  We sample them once after
    letting the filter settle for LPS_SETTLE_TIME seconds.
    """
    print(f"Waiting {LPS_SETTLE_TIME} s for LPS estimator to settle ...")
    time.sleep(LPS_SETTLE_TIME)

    pos = {}
    done = [False]

    def _cb(timestamp, data, logconf):
        if not done[0]:
            pos['x'] = data.get('kalman.stateX', 0)
            pos['y'] = data.get('kalman.stateY', 0)
            pos['z'] = data.get('kalman.stateZ', 0)
            done[0] = True

    lc = LogConfig(name='SpawnPos', period_in_ms=100)
    lc.add_variable('kalman.stateX', 'float')
    lc.add_variable('kalman.stateY', 'float')
    lc.add_variable('kalman.stateZ', 'float')
    cf.log.add_config(lc)
    lc.data_received_cb.add_callback(_cb)
    lc.start()

    timeout = 5.0
    t0 = time.time()
    while not done[0] and (time.time() - t0) < timeout:
        time.sleep(0.05)

    lc.stop()

    if not done[0]:
        raise RuntimeError("Could not read LPS position within timeout. "
                           "Check that the LPS deck is active and anchors are on.")

    return pos['x'], pos['y'], pos['z']


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

def save_log_to_csv(log_data, filename='/home/dewang/trajectory_log.csv'):
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


# -------- PLOTTING & ERROR ANALYSIS ----------

def plot_and_save_data(log_data):
    if not log_data:
        print('No data to plot.')
        return

    # Convert timestamps to relative seconds
    t  = np.array([d['time']       for d in log_data]) / 1000.0
    t -= t[0]

    # Extract positions into numpy arrays
    ax_arr = np.array([d['actual_x']   for d in log_data])
    ay_arr = np.array([d['actual_y']   for d in log_data])
    az_arr = np.array([d['actual_z']   for d in log_data])
    
    ex_arr = np.array([d['expected_x'] for d in log_data])
    ey_arr = np.array([d['expected_y'] for d in log_data])
    ez_arr = np.array([d['expected_z'] for d in log_data])

    # -------------------------------------------------------------
    # 1. ERROR CALCULATION
    # -------------------------------------------------------------
    # Mean Absolute Error (MAE) per axis
    error_x = np.mean(np.abs(ax_arr - ex_arr))
    error_y = np.mean(np.abs(ay_arr - ey_arr))
    error_z = np.mean(np.abs(az_arr - ez_arr))

    print("\n" + "="*40)
    print("      AXIS-WISE TRACKING AVERAGE ERROR      ")
    print("="*40)
    print(f"X-Axis Mean Error: {error_x:.4f} meters")
    print(f"Y-Axis Mean Error: {error_y:.4f} meters")
    print(f"Z-Axis Mean Error: {error_z:.4f} meters")
    print("="*40 + "\n")

    # -------------------------------------------------------------
    # 2. PLOT AXIS-WISE TIMELINE COMPARISON (NEW)
    # -------------------------------------------------------------
    fig_axis, axs = plt.subplots(3, 1, figsize=(11, 11), sharex=True)

    # X-axis timeline
    axs[0].plot(t, ex_arr, 'g--', label='Expected X (Target)', linewidth=2)
    axs[0].plot(t, ax_arr, 'r-', label='Actual X (LPS Feedback)', alpha=0.8)
    axs[0].set_ylabel('X Position (m)')
    axs[0].set_title(f'X-Axis Tracking Timeline (Avg Error: {error_x:.3f}m)')
    axs[0].grid(True)
    axs[0].legend(loc='upper right')

    # Y-axis timeline
    axs[1].plot(t, ey_arr, 'g--', label='Expected Y (Target)', linewidth=2)
    axs[1].plot(t, ay_arr, 'r-', label='Actual Y (LPS Feedback)', alpha=0.8)
    axs[1].set_ylabel('Y Position (m)')
    axs[1].set_title(f'Y-Axis Tracking Timeline (Avg Error: {error_y:.3f}m)')
    axs[1].grid(True)
    axs[1].legend(loc='upper right')

    # Z-axis timeline
    axs[2].plot(t, ez_arr, 'g--', label='Expected Z (Target)', linewidth=2)
    axs[2].plot(t, az_arr, 'r-', label='Actual Z (LPS Feedback)', alpha=0.8)
    axs[2].set_xlabel('Time (seconds)')
    axs[2].set_ylabel('Z Position (m)')
    axs[2].set_title(f'Z-Axis Tracking Timeline (Avg Error: {error_z:.3f}m)')
    axs[2].grid(True)
    axs[2].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig("/home/dewang/trajectory_axis_wise.png")
    print("Saved trajectory_axis_wise.png")

    # -------------------------------------------------------------
    # 3. ORIGINAL TOP-DOWN XY PLOT
    # -------------------------------------------------------------
    bx = [SAFE_X_MIN, SAFE_X_MAX, SAFE_X_MAX, SAFE_X_MIN, SAFE_X_MIN]
    by = [SAFE_Y_MIN, SAFE_Y_MIN, SAFE_Y_MAX, SAFE_Y_MAX, SAFE_Y_MIN]

    plt.figure()
    plt.plot(bx, by, 'r--', linewidth=1.5, label='Safety boundary')
    if spawn_x is not None:
        plt.plot(spawn_x, spawn_y, 'ko', markersize=8, label=f'Spawn ({spawn_x:.2f},{spawn_y:.2f})')
    plt.plot(CIRCLE_CENTER_X, CIRCLE_CENTER_Y, 'g+', markersize=12,
             markeredgewidth=2, label='Circle centre')
    plt.plot(ax_arr, ay_arr, label='Actual')
    plt.plot(ex_arr, ey_arr, '--', label='Expected')
    plt.title("Top-down XY — LPS absolute coordinates")
    plt.xlabel('X (m)'); plt.ylabel('Y (m)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig("/home/dewang/trajectory_xy_topdown.png")
    print("Saved trajectory_xy_topdown.png")

    # -------------------------------------------------------------
    # 4. ORIGINAL 3D PLOT
    # -------------------------------------------------------------
    fig_3d = plt.figure()
    ax3 = fig_3d.add_subplot(111, projection='3d')
    ax3.plot(ax_arr, ay_arr, az_arr, label='Actual', color='red')
    ax3.plot(ex_arr, ey_arr, ez_arr, 'g--', label='Expected')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_zlabel('Z (m)')
    ax3.set_title("3D Trajectory Comparison")
    ax3.legend()
    plt.savefig("/home/dewang/trajectory_3d.png")
    print("Saved trajectory_3d.png")
    
    # Displays all interactive plots cleanly if window environments allow
    plt.show()


# -------- MAIN SEQUENCE ----------

def run_sequence(cf):
    global spawn_x, spawn_y, spawn_z

    commander = cf.high_level_commander

    # READ SPAWN POSITION FROM LPS
    spawn_x, spawn_y, spawn_z = read_lps_position(cf)
    print(f"LPS spawn position: x={spawn_x:.3f}  y={spawn_y:.3f}  z={spawn_z:.3f}")

    # Safety check — make sure the drone is actually inside the safe zone
    margin = 0.1
    if not (SAFE_X_MIN + margin <= spawn_x <= SAFE_X_MAX - margin and
            SAFE_Y_MIN + margin <= spawn_y <= SAFE_Y_MAX - margin):
        raise RuntimeError(
            f"Spawn position ({spawn_x:.2f}, {spawn_y:.2f}) is outside or "
            f"too close to the safe zone boundary. Aborting."
        )

    log_conf.start()

    # Arm
    try:
        cf.supervisor.send_arming_request(True)
    except AttributeError:
        cf.platform.send_arming_request(True)
    time.sleep(1.0)

    print("Phase 1: Taking off ...")
    commander.takeoff(FLY_Z, 2.5)
    time.sleep(3.5)

    dist = math.sqrt((CIRCLE_CENTER_X - spawn_x)**2 +
                     (CIRCLE_CENTER_Y - spawn_y)**2)
    travel_time = max(3.0, dist / 0.5)

    print(f"Phase 2: Flying to circle centre (dist={dist:.2f} m, t={travel_time:.1f} s) ...")
    commander.go_to(CIRCLE_CENTER_X, CIRCLE_CENTER_Y, FLY_Z,
                    yaw=0.0, duration_s=travel_time, relative=False)
    time.sleep(travel_time + 0.5)

    print(f"Phase 3: Circle — r={RADIUS} m, ω={OMEGA} rad/s, {CIRCLE_DURATION} s ...")

    steps = int(CIRCLE_DURATION / DT)
    theta = 0.0

    for _ in range(steps):
        wx = CIRCLE_CENTER_X + RADIUS * math.cos(theta)
        wy = CIRCLE_CENTER_Y + RADIUS * math.sin(theta)

        # Hard clamp inside safe zone
        wx = max(SAFE_X_MIN + 0.05, min(SAFE_X_MAX - 0.05, wx))
        wy = max(SAFE_Y_MIN + 0.05, min(SAFE_Y_MAX - 0.05, wy))

        commander.go_to(wx, wy, FLY_Z, yaw=0.0, duration_s=DT, relative=False)
        theta += OMEGA * DT
        time.sleep(DT)

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
        run_sequence(cf)

    save_log_to_csv(log_data_list)
    plot_and_save_data(log_data_list)
