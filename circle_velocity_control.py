#!/usr/bin/env python3
"""
Single drone smooth circle flight using send_full_state_setpoint.
Position + velocity + acceleration feedforward reduces PID tracking error.
Firmware handles the actual PID (cascaded PID/PD per Snider 2025).
"""

import time
import csv
import math
import numpy as np
import matplotlib.pyplot as plt

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E701')

# -------- Circle parameters --------
CIRCLE_CENTER_X = 1.5
CIRCLE_CENTER_Y = 2.5
RADIUS          = 1.0    # metres
FLY_Z           = 0.6    # cruise altitude (m)

# -------- Speed & Timing --------
# Keep SPEED <= 0.6 m/s to stay within inner loop bandwidth (~6-8 rad/s).
# Faster speeds cause the inner loop to lag the outer loop setpoint.
SPEED   = 0.5
LOOP_HZ = 50
LOOP_DT = 1.0 / LOOP_HZ

OMEGA        = SPEED / RADIUS
LAP_DURATION = (2.0 * math.pi * RADIUS) / SPEED
N_LAPS       = 2

# Ramp omega 0 -> OMEGA over 2s to avoid entry jerk.
# Important: avoids saturating the Z thrust PD at circle entry.
RAMP_DURATION = 2.0
RAMP_STEPS    = int(RAMP_DURATION * LOOP_HZ)

LPS_SETTLE_TIME = 3.0

# -------- Logging --------
log_data_list = []
log_conf      = None


# ================================================================
# HELPERS
# ================================================================
def yaw_to_quaternion(yaw_rad):
    half = yaw_rad * 0.5
    return [0.0, 0.0, math.sin(half), math.cos(half)]


def smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


# ================================================================
# LPS
# ================================================================
def read_lps_position(cf):
    print(f"Waiting {LPS_SETTLE_TIME}s for LPS estimator to settle...")
    time.sleep(LPS_SETTLE_TIME)

    pos  = {}
    done = [False]

    def _cb(timestamp, data, logconf):
        if not done[0]:
            pos['x'] = data.get('kalman.stateX', 0)
            pos['y'] = data.get('kalman.stateY', 0)
            pos['z'] = data.get('kalman.stateZ', 0)
            done[0]  = True

    lc = LogConfig(name='SpawnPos', period_in_ms=100)
    lc.add_variable('kalman.stateX', 'float')
    lc.add_variable('kalman.stateY', 'float')
    lc.add_variable('kalman.stateZ', 'float')
    cf.log.add_config(lc)
    lc.data_received_cb.add_callback(_cb)
    lc.start()
    t0 = time.time()
    while not done[0] and (time.time() - t0) < 5.0:
        time.sleep(0.05)
    lc.stop()
    if not done[0]:
        raise RuntimeError("LPS position read timeout.")
    return pos['x'], pos['y'], pos['z']


# ================================================================
# LOGGING
# ================================================================
def log_callback(timestamp, data, logconf):
    log_data_list.append({
        'time':     timestamp,
        'actual_x': data.get('kalman.stateX', 0),
        'actual_y': data.get('kalman.stateY', 0),
        'actual_z': data.get('kalman.stateZ', 0),
    })


def setup_logging(cf):
    global log_conf
    log_conf = LogConfig(name='TrajectoryLog', period_in_ms=20)
    log_conf.add_variable('kalman.stateX', 'float')
    log_conf.add_variable('kalman.stateY', 'float')
    log_conf.add_variable('kalman.stateZ', 'float')
    cf.log.add_config(log_conf)
    log_conf.data_received_cb.add_callback(log_callback)


# ================================================================
# FULL STATE SETPOINT
# ================================================================
def compute_circle_state(theta, omega):
    """
    Position  : on the circle at angle theta
    Velocity  : tangent to circle (feedforward pre-cancels outer PID lag)
    Accel     : centripetal (feedforward pre-cancels inner PD lag)
    Yaw       : nose along velocity vector

    With these three feedforwards the firmware PID only corrects
    small residual disturbances rather than chasing a moving target.
    """
    x  = CIRCLE_CENTER_X + RADIUS * math.cos(theta)
    y  = CIRCLE_CENTER_Y + RADIUS * math.sin(theta)
    z  = FLY_Z

    vx = -RADIUS * omega * math.sin(theta)
    vy =  RADIUS * omega * math.cos(theta)
    vz = 0.0

    # Centripetal acceleration inward toward centre
    ax = -RADIUS * omega * omega * math.cos(theta)
    ay = -RADIUS * omega * omega * math.sin(theta)
    az = 0.0

    yaw_rad = math.atan2(vy, vx)
    quat    = yaw_to_quaternion(yaw_rad)

    return [x, y, z], [vx, vy, vz], [ax, ay, az], quat


# ================================================================
# MAIN SEQUENCE
# ================================================================
def run_sequence(cf):
    commander    = cf.commander
    hl_commander = cf.high_level_commander

    spawn_x, spawn_y, spawn_z = read_lps_position(cf)
    print(f"LPS spawn: x={spawn_x:.3f}  y={spawn_y:.3f}  z={spawn_z:.3f}")

    log_conf.start()

    try:
        cf.supervisor.send_arming_request(True)
    except AttributeError:
        cf.platform.send_arming_request(True)
    time.sleep(1.0)

    try:
        # Phase 1: Takeoff
        print("Phase 1: Takeoff...")
        hl_commander.takeoff(FLY_Z, 2.5)
        time.sleep(3.5)

        # Phase 2: Fly to circle entry point theta=0 => (cx+R, cy)
        # entry_yaw=90deg: nose pointing +Y (tangent direction at theta=0)
        entry_x   = CIRCLE_CENTER_X + RADIUS
        entry_y   = CIRCLE_CENTER_Y
        entry_yaw = 90.0

        dist        = math.hypot(entry_x - spawn_x, entry_y - spawn_y)
        travel_time = max(3.5, dist / 0.35)

        print(f"Phase 2: Flying to entry ({entry_x:.2f}, {entry_y:.2f})...")
        hl_commander.go_to(entry_x, entry_y, FLY_Z,
                           yaw=entry_yaw,
                           duration_s=travel_time,
                           relative=False)
        time.sleep(travel_time + 1.0)

        # Phase 3: Circle with full state feedforward
        print(f"Phase 3: Circle  R={RADIUS}m  speed={SPEED}m/s  {N_LAPS} lap(s)...")
        total_steps = RAMP_STEPS + int(LAP_DURATION * N_LAPS * LOOP_HZ)
        start_time  = time.time()
        theta       = 0.0

        for step in range(total_steps):
            if step < RAMP_STEPS:
                current_omega = OMEGA * smoothstep(step / RAMP_STEPS)
            else:
                current_omega = OMEGA

            theta += current_omega * LOOP_DT

            pos_vec, vel_vec, acc_vec, quat_vec = compute_circle_state(theta, current_omega)

            commander.send_full_state_setpoint(
                pos_vec,
                vel_vec,
                acc_vec,
                quat_vec,
                0.0, 0.0, 0.0   # body roll/pitch/yaw rates
            )

            expected_wake = start_time + (step + 1) * LOOP_DT
            sleep_time    = expected_wake - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Phase 4: Hold position at entry point
        print("Phase 4: Holding position...")
        for _ in range(int(LOOP_HZ * 1.5)):
            commander.send_position_setpoint(entry_x, entry_y, FLY_Z, entry_yaw)
            time.sleep(LOOP_DT)

    except Exception as e:
        print(f"[ERROR] {e}")
    except KeyboardInterrupt:
        print("[ABORT] Manual interrupt.")

    finally:
        print("Landing...")
        hl_commander.land(0.0, duration_s=3.0)
        time.sleep(3.5)
        hl_commander.stop()
        log_conf.stop()
        print("On ground.")


# ================================================================
# SAVE CSV
# ================================================================
def save_log_to_csv(log_data, filename='/home/dewang/trajectory_log.csv'):
    if not log_data:
        return
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['time','actual_x','actual_y','actual_z'])
        writer.writeheader()
        writer.writerows(log_data)
    print(f"CSV saved: {filename}")


# ================================================================
# PLOT
# ================================================================
def plot_and_save(log_data):
    if not log_data:
        print("No log data.")
        return

    ax_arr = np.array([d['actual_x'] for d in log_data])
    ay_arr = np.array([d['actual_y'] for d in log_data])

    angles  = np.linspace(0, 2 * math.pi, 300)
    ideal_x = CIRCLE_CENTER_X + RADIUS * np.cos(angles)
    ideal_y = CIRCLE_CENTER_Y + RADIUS * np.sin(angles)

    plt.figure(figsize=(8, 8))
    plt.plot(ideal_x, ideal_y, 'g--', linewidth=1.5, label=f'Ideal (R={RADIUS}m)')
    plt.plot(ax_arr,  ay_arr,  'b-',  linewidth=1.5, alpha=0.85, label='Actual')

    if len(ax_arr) > 1:
        ref_x = CIRCLE_CENTER_X + RADIUS * np.cos(
            np.linspace(0, 2 * math.pi * N_LAPS, len(ax_arr)))
        ref_y = CIRCLE_CENTER_Y + RADIUS * np.sin(
            np.linspace(0, 2 * math.pi * N_LAPS, len(ax_arr)))
        rms = np.sqrt(np.mean((ax_arr - ref_x)**2 + (ay_arr - ref_y)**2))
        print(f"RMS tracking error: {rms*100:.1f} cm")
        plt.title(f'Circle — Full State Feedforward\nRMS error: {rms*100:.1f} cm')
    else:
        plt.title('Circle — Full State Feedforward')

    plt.plot(ax_arr[0],  ay_arr[0],  'ko', markersize=7, label='Start')
    plt.plot(ax_arr[-1], ay_arr[-1], 'rs', markersize=7, label='End')
    plt.plot(CIRCLE_CENTER_X, CIRCLE_CENTER_Y, 'g+', markersize=14, markeredgewidth=2)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('/home/dewang/trajectory_xy.png', dpi=300)
    print("Plot saved: /home/dewang/trajectory_xy.png")


# ================================================================
# ENTRY POINT
# ================================================================
if __name__ == '__main__':
    cflib.crtp.init_drivers()
    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        setup_logging(scf.cf)
        run_sequence(scf.cf)
    save_log_to_csv(log_data_list)
    plot_and_save(log_data_list)